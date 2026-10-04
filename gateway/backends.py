from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from gateway.config import Tier
from gateway.models import ChatRequest


class BackendError(Exception):
    def __init__(self, kind: str, status: int = 502, retryable: bool = True):
        super().__init__(kind)
        self.kind, self.status, self.retryable = kind, status, retryable


@dataclass
class BackendEvent:
    delta: dict = field(default_factory=dict)
    finish_reason: str | None = None
    usage: dict | None = None


class Backend(Protocol):
    def complete(self, request: ChatRequest) -> AsyncIterator[BackendEvent]: ...


def validate_omissions(tier: Tier, request: ChatRequest):
    for parameter in tier.omit_parameters:
        if parameter in request.model_fields_set:
            raise BackendError(f"unsupported_parameter_{parameter}", 400, False)


def with_extra_body(tier: Tier, payload: dict) -> dict:
    # Extensions cannot overwrite protocol fields or translated client settings.
    protected = set(ChatRequest.model_fields) | {"contents", "generationConfig", "systemInstruction", "system"}
    if set(tier.extra_body) & (set(payload) | protected):
        raise BackendError("provider_configuration_conflict", 503, False)
    return {**payload, **tier.extra_body}


def provider_key(tier: Tier, default_env: str) -> str:
    key = os.getenv(tier.api_key_env or default_env)
    if not key:
        raise BackendError("provider_credentials_missing", 503, False)
    return key


async def sse_data(response: httpx.Response) -> AsyncIterator[str]:
    lines: list[str] = []
    async for line in response.aiter_lines():
        if line.startswith("data:"):
            lines.append(line[5:].lstrip())
        elif not line and lines:
            yield "\n".join(lines)
            lines.clear()


def check_status(response: httpx.Response):
    if response.is_error:
        status = response.status_code
        # Provider response bodies can contain secrets or prompt text; never relay them.
        raise BackendError(
            f"upstream_http_{status}", 503 if status == 429 else 502,
            retryable=status == 429 or status >= 500,
        )


class OpenAIBackend:
    def __init__(self, tier: Tier, client: httpx.AsyncClient):
        self.tier, self.client = tier, client

    async def complete(self, request: ChatRequest) -> AsyncIterator[BackendEvent]:
        validate_omissions(self.tier, request)
        payload = request.upstream_payload()
        payload.pop("max_tokens", None)
        payload.pop("max_completion_tokens", None)
        payload[self.tier.token_limit_parameter] = request.max_tokens or request.max_completion_tokens
        for parameter in self.tier.omit_parameters:
            payload.pop(parameter, None)
        for parameter in ("frequency_penalty", "presence_penalty"):
            if not getattr(self.tier.capabilities, parameter) and parameter not in request.model_fields_set:
                payload.pop(parameter, None)
        for message in payload["messages"]:
            message.pop("provider_metadata", None)
        payload.update(model=self.tier.model, stream=True)
        if self.tier.send_stream_usage:
            payload["stream_options"] = {"include_usage": True}
        payload = with_extra_body(self.tier, payload)
        headers = {}
        if self.tier.api_key_env:
            key = os.getenv(self.tier.api_key_env)
            if not key:
                raise BackendError("provider_credentials_missing", 503, False)
            headers["Authorization"] = f"Bearer {key}"
        finished = False
        async with self.client.stream(
            "POST", f"{self.tier.base_url.rstrip('/')}/chat/completions",
            json=payload, headers=headers, timeout=None,
        ) as response:
            check_status(response)
            async for raw in sse_data(response):
                if raw == "[DONE]":
                    if not finished:
                        raise BackendError("upstream_missing_finish")
                    return
                try:
                    item = json.loads(raw)
                except ValueError as exc:
                    raise BackendError("invalid_upstream_event") from exc
                if "error" in item:
                    code = item["error"].get("code")
                    retryable = code in (429, "429", "rate_limit_exceeded") or isinstance(code, int) and code >= 500
                    raise BackendError("upstream_stream_error", 502, retryable)
                choices = item.get("choices", [])
                if choices:
                    choice = choices[0]
                    delta = {k: v for k, v in choice.get("delta", {}).items() if k != "role" and v not in (None, "", [])}
                    reason = choice.get("finish_reason")
                    if reason:
                        finished = True
                    if delta or reason:
                        yield BackendEvent(delta, reason, item.get("usage"))
                elif item.get("usage") and finished:
                    yield BackendEvent(usage=item["usage"])
            # A truncated stream is not a successful completion, even if HTTP was 200.
            raise BackendError("upstream_connection_dropped")


class OllamaBackend:
    def __init__(self, tier: Tier, client: httpx.AsyncClient):
        self.tier, self.client = tier, client

    async def complete(self, request: ChatRequest) -> AsyncIterator[BackendEvent]:
        validate_omissions(self.tier, request)
        if request.tools or request.tool_choice or request.parallel_tool_calls is not None or any(
            message.role == "tool" or message.tool_calls for message in request.messages
        ):
            raise BackendError("ollama_tools_not_supported", 400, False)
        if request.frequency_penalty or request.presence_penalty:
            raise BackendError("ollama_penalty_not_supported", 400, False)
        options = {
            "temperature": request.temperature, "top_p": request.top_p,
            "num_predict": request.max_tokens or request.max_completion_tokens,
        }
        if request.stop is not None:
            options["stop"] = [request.stop] if isinstance(request.stop, str) else request.stop
        if request.seed is not None:
            options["seed"] = request.seed
        for parameter in self.tier.omit_parameters:
            options.pop(parameter, None)
        payload = {
            "model": self.tier.model, "stream": True, "options": options,
            "messages": [
                {"role": "system" if m.role == "developer" else m.role, "content": m.content}
                for m in request.messages
            ],
        }
        if request.response_format:
            if request.response_format.get("type") == "text":
                pass
            elif request.response_format.get("type") == "json_object":
                payload["format"] = "json"
            elif request.response_format.get("type") == "json_schema":
                payload["format"] = request.response_format["json_schema"]["schema"]
            else:
                raise BackendError("unsupported_response_format", 400, False)
        async with self.client.stream(
            "POST", f"{self.tier.base_url.rstrip('/')}/api/chat", json=with_extra_body(self.tier, payload), timeout=None
        ) as response:
            check_status(response)
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except ValueError as exc:
                    raise BackendError("invalid_upstream_event") from exc
                if "error" in item:
                    raise BackendError("ollama_generation_error")
                message = item.get("message", {})
                delta = {}
                if message.get("content"):
                    delta["content"] = message["content"]
                if message.get("thinking"):
                    delta["reasoning_content"] = message["thinking"]
                done = item.get("done", False)
                usage = None
                if done:
                    prompt = item.get("prompt_eval_count", 0)
                    output = item.get("eval_count", 0)
                    usage = {"prompt_tokens": prompt, "completion_tokens": output, "total_tokens": prompt + output}
                if delta or done:
                    yield BackendEvent(delta, (item.get("done_reason") or "stop") if done else None, usage)
                if done:
                    return
            raise BackendError("upstream_connection_dropped")


class MockBackend:
    def __init__(self, tier: Tier):
        self.tier = tier
        self.calls = 0

    async def complete(self, request: ChatRequest) -> AsyncIterator[BackendEvent]:
        self.calls += 1
        text = f"Demo response from {self.tier.name}: {request.messages[-1].content or ''}"
        words = text.split()
        for index, word in enumerate(words):
            await asyncio.sleep(self.tier.mock_delay)
            yield BackendEvent({"content": (" " if index else "") + word})
        prompt = max(1, len(request.prompt_text()) // 4)
        yield BackendEvent(finish_reason="stop", usage={
            "prompt_tokens": prompt, "completion_tokens": len(words), "total_tokens": prompt + len(words)
        })


def make_backend(tier: Tier, client: httpx.AsyncClient) -> Backend:
    from gateway.providers import AnthropicBackend, GeminiBackend

    if tier.provider == "mock":
        return MockBackend(tier)
    adapters = {
        "openai": OpenAIBackend, "openai_compatible": OpenAIBackend,
        "ollama": OllamaBackend, "anthropic": AnthropicBackend, "gemini": GeminiBackend,
    }
    return adapters[tier.provider](tier, client)
