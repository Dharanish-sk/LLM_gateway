import json

import httpx
import pytest

from gateway.backends import BackendError, OllamaBackend, OpenAIBackend
from gateway.config import Tier
from gateway.models import ChatRequest
from gateway.protocol import Accumulator

REQUEST = ChatRequest(messages=[{"role": "user", "content": "hello"}])


async def test_openai_sse_tool_calls_usage_and_payload():
    def transport(request):
        payload = json.loads(request.content)
        assert payload["stream_options"] == {"include_usage": True}
        assert payload["model"] == "hosted-model"
        events = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "weather", "arguments": '{"city":'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        ]
        return httpx.Response(200, text=": ping\n\n" + "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n")

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        backend = OpenAIBackend(Tier(name="large", provider="openai", model="hosted-model", base_url="https://example.test/v1"), client)
        result = Accumulator()
        async for event in backend.complete(REQUEST):
            result.add(event)
        assert result.message()["tool_calls"][0]["function"] == {"name": "weather", "arguments": '{"city":"Paris"}'}
        assert result.usage["total_tokens"] == 15
        assert result.reason == "tool_calls"


@pytest.mark.parametrize("status,retryable", [(429, True), (500, True), (401, False), (400, False)])
async def test_upstream_status_classification(status, retryable):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(status))) as client:
        backend = OpenAIBackend(Tier(name="large", provider="openai", model="test"), client)
        with pytest.raises(BackendError) as caught:
            await anext(backend.complete(REQUEST))
        assert caught.value.retryable == retryable


async def test_ollama_mapping_and_truncated_stream():
    def transport(request):
        payload = json.loads(request.content)
        assert payload["options"]["num_predict"] == 256
        return httpx.Response(200, text=json.dumps({"message": {"content": "Hello"}, "done": False}) + "\n")

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        backend = OllamaBackend(Tier(name="local", provider="ollama", model="test"), client)
        stream = backend.complete(REQUEST)
        assert (await anext(stream)).delta == {"content": "Hello"}
        with pytest.raises(BackendError, match="dropped"):
            await anext(stream)
