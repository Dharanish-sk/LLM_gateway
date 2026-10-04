from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from redis.asyncio import Redis
from redis.exceptions import RedisError

from gateway.auth import KeyAuth
from gateway.backends import BackendError, make_backend
from gateway.cache import ResponseCache
from gateway.capabilities import CapabilityRegistry, SelectionError, effective_capabilities
from gateway.config import Settings, load_settings
from gateway.embeddings import DemoEmbedder, OllamaEmbedder
from gateway.models import ChatRequest
from gateway.protocol import Accumulator, chunk, sse
from gateway.reliability import BackendPool
from gateway.router import Router
from gateway.state import MemoryState, RedisState, retry_seconds
from gateway.telemetry import Telemetry

logger = logging.getLogger("gateway")


def error(message: str, code: str, status: int, headers=None, details=None):
    body = {"message": message, "type": "gateway_error", "code": code, "param": None}
    if details is not None:
        body["details"] = details
    return JSONResponse({"error": body}, status_code=status, headers=headers)


class BodyLimit:
    def __init__(self, app, maximum: int):
        self.app, self.maximum = app, maximum

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.maximum:
                return await error("Request body too large", "request_too_large", 413)(scope, receive, send)
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def create_app(settings: Settings | None = None, *, state=None, backends=None, embedder=None):
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app):
        client = httpx.AsyncClient(limits=httpx.Limits(max_connections=settings.max_concurrent_requests * 3), trust_env=False)
        store = state or (MemoryState() if settings.state == "memory" else RedisState(Redis.from_url(
            os.getenv(settings.redis_url_env, "redis://localhost:6379/0"), decode_responses=True,
            socket_connect_timeout=2, socket_timeout=2,
        )))
        telemetry = Telemetry()
        try:
            app.state.auth = KeyAuth(settings.keys_file, settings.mode == "demo")
            embedding = embedder or (DemoEmbedder() if settings.mode == "demo" else OllamaEmbedder(
                client, settings.cache.embedding_url, settings.cache.embedding_model, settings.cache.embedding_timeout
            ))
            app.state.cache = ResponseCache(store, embedding, settings.cache, settings.policy_hash())
            app.state.capabilities = CapabilityRegistry(settings.tiers, settings.reliability, settings.router)
            app.state.router = Router(settings.router, embedding, app.state.capabilities)
            artifact = app.state.router.artifact
            if artifact:
                models = {tier.name: tier.model for tier in settings.tiers}
                if artifact.get("small_model") != models[settings.router.small_tier] or artifact.get("large_model") != models[settings.router.large_tier]:
                    raise ValueError("classifier was trained for different tier models")
            app.state.pool = BackendPool(settings.tiers, backends or {t.name: make_backend(t, client) for t in settings.tiers}, settings.reliability, telemetry, app.state.capabilities)
            app.state.store, app.state.telemetry = store, telemetry
            app.state.active = 0
            yield
        finally:
            await client.aclose()
            await store.close()
            await asyncio.to_thread(telemetry.close)

    app = FastAPI(title="Intelligent LLM Gateway", version="0.1.0", lifespan=lifespan)
    app.add_middleware(BodyLimit, maximum=settings.max_request_bytes)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        details = "; ".join(".".join(map(str, item["loc"])) + ": " + item["msg"] for item in exc.errors())
        return error(details, "invalid_request", 400)

    @app.get("/health/live")
    async def live():
        return {"status": "ok", "mode": settings.mode}

    @app.get("/health/ready")
    async def ready():
        try:
            await app.state.store.ping()
        except RedisError:
            return error("State store unavailable", "not_ready", 503)
        return {"status": "ready", "mode": settings.mode, "backends_probed": False}

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        return Response(generate_latest(app.state.telemetry.registry), headers={"Content-Type": CONTENT_TYPE_LATEST})

    @app.get("/v1/models")
    async def models(request: Request):
        if not app.state.auth.authenticate(request.headers.get("authorization")):
            return error("Invalid API key", "invalid_api_key", 401, {"WWW-Authenticate": "Bearer"})
        data = [{"id": "auto", "object": "model", "created": 0, "owned_by": "gateway"}]
        data += [{"id": tier.name, "object": "model", "created": 0, "owned_by": "gateway", "capabilities": effective_capabilities(tier)} for tier in settings.tiers]
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions", response_model=None)
    async def completions(body: ChatRequest, request: Request):
        telemetry = app.state.telemetry
        started = time.monotonic()
        request_id, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())
        root = telemetry.tracer.start_span("chat.completions", attributes={"request.id": request_id, "gateway.mode": settings.mode})
        selected, outcome, charged, admitted = "none", "error", False, False
        accumulator = Accumulator()
        iterator = None

        async def finalize():
            nonlocal charged
            if charged:
                return
            charged = True
            try:
                if iterator:
                    await iterator.aclose()
            finally:
                if admitted:
                    app.state.active -= 1
                if selected in app.state.pool.tiers and outcome != "cache":
                    usage, source = accumulator.effective_usage(body.prompt_text())
                    telemetry.charge(app.state.pool.tiers[selected], usage, source)
                telemetry.requests.labels(selected, outcome).inc()
                telemetry.latency.labels(selected).observe(time.monotonic() - started)
                root.set_attribute("outcome", outcome)
                root.end()

        try:
            with telemetry.span("auth", root):
                identity = app.state.auth.authenticate(request.headers.get("authorization"))
            if not identity:
                outcome = "unauthorized"
                await finalize()
                return error("Invalid API key", "invalid_api_key", 401, {"WWW-Authenticate": "Bearer"})
            if body.model != "auto" and body.model not in app.state.pool.tiers:
                outcome = "invalid_model"
                await finalize()
                return error("Unknown model alias; see /v1/models", "model_not_found", 404)
            with telemetry.span("rate_limit", root):
                bucket = await app.state.store.consume(hashlib.sha256(identity.key_id.encode()).hexdigest(), settings.bucket_capacity, settings.bucket_refill_rate)
            if not bucket.allowed:
                telemetry.rejections.inc()
                outcome = "rate_limited"
                await finalize()
                return error("Rate limit exceeded", "rate_limit_exceeded", 429, {"Retry-After": retry_seconds(bucket)})
            if app.state.active >= settings.max_concurrent_requests:
                outcome = "overloaded"
                await finalize()
                return error("Gateway concurrency limit reached", "overloaded", 503, {"Retry-After": "1"})
            app.state.active += 1
            admitted = True
            with telemetry.span("capabilities", root) as span:
                report = app.state.capabilities.preflight(body)
                span.set_attribute("eligible.tiers", list(report.eligible))
                span.set_attribute("input_tokens.estimated", report.needs.input_tokens)
                for name, reasons in report.rejected.items():
                    for reason in reasons:
                        telemetry.capability_rejections.labels(name, reason).inc()
            with telemetry.span("cache", root) as span:
                try:
                    cached, cache_status, context = await app.state.cache.lookup(body, identity.key_id, set(report.eligible))
                except (httpx.HTTPError, ValueError):
                    cached, cache_status, context = None, "embedding_error", app.state.cache.context(body, identity.key_id)
                span.set_attribute("cache.result", cache_status)
                telemetry.cache.labels(cache_status).inc()
            headers = {"X-Request-ID": request_id, "X-Cache": cache_status, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
            if cached:
                selected, outcome = cached["model"], "cache"
                cached = {**cached, "id": request_id, "created": created}
                if not body.stream:
                    await finalize()
                    return JSONResponse(cached, headers=headers)

                async def cached_stream():
                    try:
                        message = dict(cached["choices"][0]["message"])
                        if "tool_calls" in message:
                            message["tool_calls"] = [{"index": i, **call} for i, call in enumerate(message["tool_calls"])]
                        yield sse(chunk(request_id, created, selected, message))
                        yield sse(chunk(request_id, created, selected, reason=cached["choices"][0]["finish_reason"]))
                        if body.stream_options and body.stream_options.include_usage:
                            yield sse(chunk(request_id, created, selected, usage=cached["usage"]))
                        yield sse("[DONE]")
                    finally:
                        await finalize()

                return StreamingResponse(cached_stream(), media_type="text/event-stream", headers=headers)
            with telemetry.span("router", root) as span:
                decision = await app.state.router.choose(body, report)
                span.set_attribute("router.tier", decision.tier)
                span.set_attribute("router.reason", decision.reason)
                span.set_attribute("router.preferred_tier", decision.preferred_tier)
                if decision.tier != decision.preferred_tier:
                    telemetry.capability_reroutes.labels(decision.preferred_tier, decision.tier).inc()
                if decision.probability is not None:
                    span.set_attribute("router.probability", decision.probability)
            iterator = app.state.pool.complete(body, decision.tier, root)
            # Prefetch before sending HTTP headers, so pre-output errors retain HTTP status.
            selected, first = await anext(iterator)

            async def events():
                yield first
                async for _, event in iterator:
                    yield event

            async def finish_success():
                nonlocal outcome
                if not accumulator.finished:
                    raise BackendError("upstream_missing_finish")
                result = accumulator.response(request_id, created, selected, body.prompt_text())
                try:
                    await app.state.cache.store(context, result)
                except RedisError:
                    telemetry.cache.labels("write_error").inc()
                outcome = "success"
                return result

            if not body.stream:
                async for event in events():
                    accumulator.add(event)
                result = await finish_success()
                await finalize()
                return JSONResponse(result, headers={**headers, "X-Gateway-Tier": selected})

            async def stream():
                nonlocal outcome
                try:
                    yield sse(chunk(request_id, created, selected, {"role": "assistant", "content": ""}))
                    async for event in events():
                        accumulator.add(event)
                        if event.delta:
                            yield sse(chunk(request_id, created, selected, event.delta))
                    await finish_success()
                    yield sse(chunk(request_id, created, selected, reason=accumulator.reason))
                    if body.stream_options and body.stream_options.include_usage:
                        yield sse(chunk(request_id, created, selected, usage=accumulator.effective_usage(body.prompt_text())[0]))
                    yield sse("[DONE]")
                except BackendError as exc:
                    outcome = "stream_error"
                    yield sse({"error": {"message": "Backend failed after output began", "type": "gateway_error", "code": exc.kind}})
                    yield sse("[DONE]")
                except asyncio.CancelledError:
                    outcome = "cancelled"
                    raise
                finally:
                    await finalize()

            return StreamingResponse(stream(), media_type="text/event-stream", headers={**headers, "X-Gateway-Tier": selected})
        except SelectionError as exc:
            outcome = "incompatible_request" if exc.code == "no_compatible_tier" else "invalid_request"
            await finalize()
            return error(exc.message, exc.code, exc.status, {"X-Request-ID": request_id}, exc.details)
        except BackendError as exc:
            await finalize()
            return error("Backend request failed", exc.kind, exc.status, {"X-Request-ID": request_id})
        except RedisError:
            outcome = "state_unavailable"
            await finalize()
            return error("State store unavailable", "state_unavailable", 503)
        except BaseException:
            await finalize()
            raise

    return app
