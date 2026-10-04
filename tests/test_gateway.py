import asyncio
import json

import httpx
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from gateway.app import create_app
from gateway.backends import BackendError, BackendEvent, MockBackend
from gateway.reliability import CircuitBreaker
from gateway.state import MemoryState

AUTH = {"Authorization": "Bearer demo-key-change-me"}
BODY = {"model": "local", "messages": [{"role": "user", "content": "Hello"}]}


class FaultBackend:
    def __init__(self, mode):
        self.mode, self.calls, self.closed = mode, 0, False

    async def complete(self, request):
        self.calls += 1
        try:
            if self.mode == "timeout":
                await asyncio.sleep(1)
            if self.mode == "partial":
                yield BackendEvent({"content": "partial"})
            raise BackendError("injected_500", retryable=self.mode != "400")
        finally:
            self.closed = True


async def test_auth_protocol_cache_and_metrics(settings):
    app = create_app(settings)
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.post("/v1/chat/completions", json=BODY)).status_code == 401
        assert (await client.get("/health/ready")).status_code == 200
        assert (await client.get("/docs")).status_code == 200
        assert (await client.post("/v1/chat/completions", json={**BODY, "n": 2}, headers=AUTH)).status_code == 400
        response = await client.post("/v1/chat/completions", json=BODY, headers=AUTH)
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "Demo response from local: Hello"
        assert response.headers["x-cache"] == "miss"
        streamed = await client.post("/v1/chat/completions", json={**BODY, "stream": True, "stream_options": {"include_usage": True}}, headers=AUTH)
        assert streamed.headers["x-cache"] == "exact"
        frames = [line[6:] for line in streamed.text.splitlines() if line.startswith("data: ")]
        assert frames[-1] == "[DONE]"
        assert json.loads(frames[-2])["usage"]["total_tokens"] > 0
        assert app.state.pool.backends["local"].calls == 1
        assert app.state.active == 0
        metrics = (await client.get("/metrics")).text
        assert 'gateway_cache_total{result="exact"} 1.0' in metrics
        assert 'gateway_requests_total{outcome="success",tier="local"} 1.0' in metrics


@pytest.mark.parametrize("mode", ["500", "timeout", "partial", "400"])
async def test_fallback_only_before_first_output(settings, mode):
    settings.cache.enabled = False
    settings.tiers[0].ttft_timeout = 0.01
    faulty = FaultBackend(mode)
    backends = {tier.name: MockBackend(tier) for tier in settings.tiers}
    backends["local"] = faulty
    app = create_app(settings, backends=backends)
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json={**BODY, "stream": True}, headers=AUTH)
        if mode in {"500", "timeout"}:
            assert response.status_code == 200
            assert response.headers["x-gateway-tier"] == "large"
            assert backends["large"].calls == 1
        elif mode == "partial":
            assert response.status_code == 200
            assert '"content":"partial"' in response.text
            assert '"error"' in response.text
            assert backends["large"].calls == 0
            assert not app.state.store.entries
        else:
            assert response.status_code == 502
            assert backends["large"].calls == 0
        assert faulty.closed
        assert app.state.active == 0


async def test_retry_budget(settings):
    settings.reliability.max_retries = 1
    backends = {tier.name: FaultBackend("500") for tier in settings.tiers}
    app = create_app(settings, backends=backends)
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json=BODY, headers=AUTH)
        assert response.status_code == 502
        assert sum(b.calls for b in backends.values()) == 2


async def test_circuit_breaker_single_probe_and_recovery():
    now = [0]
    breaker = CircuitBreaker(2, 5, lambda: now[0])
    await breaker.failure("local")
    assert await breaker.acquire("local")
    await breaker.failure("local")
    assert not await breaker.acquire("local")
    now[0] = 5
    probes = await asyncio.gather(*(breaker.acquire("local") for _ in range(20)))
    assert sum(probes) == 1
    await breaker.success("local")
    assert await breaker.acquire("local")


async def test_redis_failure_is_closed(settings):
    class BrokenState(MemoryState):
        async def consume(self, *args):
            raise RedisConnectionError("down")

    app = create_app(settings, state=BrokenState())
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json=BODY, headers=AUTH)
        assert response.status_code == 503
        assert app.state.pool.backends["local"].calls == 0


async def test_rate_limit_and_body_limit(settings):
    settings.bucket_capacity, settings.bucket_refill_rate = 1, 0.001
    settings.max_request_bytes = 1024
    app = create_app(settings)
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.post("/v1/chat/completions", json=BODY, headers=AUTH)).status_code == 200
        response = await client.post("/v1/chat/completions", json=BODY, headers=AUTH)
        assert response.status_code == 429 and int(response.headers["retry-after"]) > 0
        assert (await client.post("/v1/chat/completions", content="x" * 1025, headers=AUTH)).status_code == 413


async def test_stream_disconnect_closes_upstream_and_releases_admission(settings):
    from experiments.load import serve_app

    class LongBackend:
        def __init__(self):
            self.closed = asyncio.Event()

        async def complete(self, request):
            try:
                while True:
                    yield BackendEvent({"content": "token"})
                    await asyncio.sleep(.01)
            finally:
                self.closed.set()

    settings.cache.enabled = False
    backend = LongBackend()
    backends = {tier.name: backend for tier in settings.tiers}
    app = create_app(settings, backends=backends)
    async with serve_app(app) as url, httpx.AsyncClient() as client:
        async with client.stream("POST", url + "/v1/chat/completions", headers=AUTH, json={**BODY, "stream": True}) as response:
            async for line in response.aiter_lines():
                if '"content":"token"' in line:
                    break
        await asyncio.wait_for(backend.closed.wait(), 3)
        assert not app.state.store.entries
    assert app.state.active == 0


async def test_spans_share_request_trace_and_exclude_prompts(settings):
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    app = create_app(settings)
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        app.state.telemetry.provider.add_span_processor(SimpleSpanProcessor(exporter))
        response = await client.post("/v1/chat/completions", json=BODY, headers=AUTH)
        assert response.status_code == 200
        spans = exporter.get_finished_spans()
        assert {span.name for span in spans} == {"chat.completions", "auth", "rate_limit", "capabilities", "cache", "router", "backend"}
        assert len({span.context.trace_id for span in spans}) == 1
        assert "Hello" not in str([span.attributes for span in spans])
