import json

import httpx
import pytest

from gateway.backends import BackendError, OpenAIBackend, make_backend
from gateway.config import Tier, load_settings
from gateway.models import ChatRequest
from gateway.protocol import Accumulator
from gateway.providers import AnthropicBackend, GeminiBackend

REQUEST = ChatRequest(messages=[{"role": "user", "content": "hello"}])
TOOLS = [{"type": "function", "function": {"name": "weather", "description": "Read weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def wire(events):
    return ": heartbeat\n\n" + "".join(f"event: message\ndata: {json.dumps(event)}\n\n" for event in events)


async def accumulate(backend, request=REQUEST):
    result = Accumulator()
    async for event in backend.complete(request):
        result.add(event)
    return result


async def test_anthropic_text_system_usage_and_auth(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

    def transport(request):
        assert str(request.url) == "https://api.anthropic.com/v1/messages"
        assert request.headers["x-api-key"] == "test-anthropic-key"
        assert request.headers["anthropic-version"] == "2023-06-01"
        assert "authorization" not in request.headers
        payload = json.loads(request.content)
        assert payload["system"] == "Be concise\n\nBe accurate"
        assert payload["max_tokens"] == 100
        assert payload["temperature"] == 0 and "top_p" not in payload
        assert payload["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        return httpx.Response(200, text=wire([
            {"type": "message_start", "message": {"usage": {"input_tokens": 10, "cache_read_input_tokens": 4, "output_tokens": 1}}},
            {"type": "ping"},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
            {"type": "message_stop"},
        ]))

    request = ChatRequest(messages=[{"role": "system", "content": "Be concise"}, {"role": "developer", "content": "Be accurate"}, {"role": "user", "content": "hello"}], max_completion_tokens=100)
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        result = await accumulate(make_backend(Tier(name="claude", provider="anthropic", model="claude-test"), client), request)
    assert result.content == "Hello" and result.reason == "stop"
    assert result.usage == {"prompt_tokens": 14, "completion_tokens": 2, "total_tokens": 16}


async def test_anthropic_tool_fragments_thinking_signature_and_continuation(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 20, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Check weather."}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "signed"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "weather", "input": {}}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"city":'}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '"Paris"}'}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 8}},
        {"type": "message_stop"},
    ]
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, text=wire(events)))) as client:
        backend = AnthropicBackend(Tier(name="claude", provider="anthropic", model="test"), client)
        request = ChatRequest(messages=REQUEST.messages, tools=TOOLS, tool_choice="required", parallel_tool_calls=False)
        payload = backend.payload(request)
        assert payload["tool_choice"] == {"type": "any", "disable_parallel_tool_use": True}
        assert payload["tools"][0]["input_schema"] == TOOLS[0]["function"]["parameters"]
        result = await accumulate(backend, request)
        assert result.reason == "tool_calls"
        message = result.message()
        assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"city": "Paris"}
        continuation = ChatRequest(messages=[*REQUEST.messages, message, {"role": "tool", "tool_call_id": "toolu_1", "content": "sunny"}], tools=TOOLS)
        translated = backend.payload(continuation)
        assert translated["messages"][1]["content"][0] == {"type": "thinking", "thinking": "Check weather.", "signature": "signed"}
        assert translated["messages"][-1]["content"] == [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"}]


async def test_anthropic_zero_argument_tool_and_json_schema(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    events = [
        {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "now", "input": {}}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
        {"type": "message_stop"},
    ]
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, text=wire(events)))) as client:
        backend = AnthropicBackend(Tier(name="claude", provider="anthropic", model="test"), client)
        result = await accumulate(backend)
        assert result.message()["tool_calls"][0]["function"]["arguments"] == "{}"
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}
        request = ChatRequest(messages=REQUEST.messages, response_format={"type": "json_schema", "json_schema": {"name": "result", "schema": schema}})
        assert backend.payload(request)["output_config"]["format"]["schema"] == schema


async def test_gemini_text_usage_reasoning_and_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-google-key")

    def transport(request):
        assert request.url.path == "/v1beta/models/gemini-test:streamGenerateContent"
        assert dict(request.url.params) == {"alt": "sse"}
        assert request.headers["x-goog-api-key"] == "test-google-key"
        assert "test-google-key" not in str(request.url)
        payload = json.loads(request.content)
        assert payload["systemInstruction"] == {"parts": [{"text": "Be brief"}]}
        assert payload["generationConfig"]["maxOutputTokens"] == 256
        assert payload["generationConfig"]["stopSequences"] == ["END"]
        assert payload["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
        return httpx.Response(200, text=wire([
            {"candidates": [{"content": {"parts": [{"text": "A thought summary", "thought": True}]}}]},
            {"candidates": [{"content": {"parts": [{"text": "Hello"}]}, "finishReason": "STOP"}]},
            {"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 2, "thoughtsTokenCount": 3, "totalTokenCount": 15}},
        ]))

    request = ChatRequest(messages=[{"role": "system", "content": "Be brief"}, *REQUEST.messages], stop="END")
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        result = await accumulate(make_backend(Tier(name="gemini", provider="gemini", model="models/gemini-test"), client), request)
    assert result.content == "Hello" and result.reasoning == "A thought summary"
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


async def test_gemini_tool_signature_and_result_round_trip(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    part = {"functionCall": {"name": "weather", "args": {"city": "Paris"}, "id": "native-call"}, "thoughtSignature": "opaque-signature"}
    events = [{"candidates": [{"content": {"parts": [part]}, "finishReason": "STOP"}]}]
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, text=wire(events)))) as client:
        backend = GeminiBackend(Tier(name="gemini", provider="gemini", model="test"), client)
        request = ChatRequest(messages=REQUEST.messages, tools=TOOLS, tool_choice={"type": "function", "function": {"name": "weather"}})
        assert backend.payload(request)["toolConfig"]["functionCallingConfig"] == {"mode": "ANY", "allowedFunctionNames": ["weather"]}
        result = await accumulate(backend, request)
        assert result.reason == "tool_calls"
        message = result.message()
        assert message["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "opaque-signature"
        continuation = ChatRequest(messages=[*REQUEST.messages, message, {"role": "tool", "tool_call_id": message["tool_calls"][0]["id"], "content": '{"weather":"sunny"}'}], tools=TOOLS)
        translated = backend.payload(continuation)
        assert translated["contents"][1]["parts"] == [part]
        assert translated["contents"][2]["parts"] == [{"functionResponse": {"name": "weather", "id": "native-call", "response": {"weather": "sunny"}}}]


@pytest.mark.parametrize("events,reason", [
    ([{"promptFeedback": {"blockReason": "SAFETY"}}], "content_filter"),
    ([{"candidates": [{"finishReason": "MAX_TOKENS"}]}], "length"),
])
async def test_gemini_finish_normalization(monkeypatch, events, reason):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, text=wire(events)))) as client:
        result = await accumulate(GeminiBackend(Tier(name="gemini", provider="gemini", model="test"), client))
    assert result.reason == reason and result.finished


@pytest.mark.parametrize("provider", ["anthropic", "gemini"])
@pytest.mark.parametrize("status,retryable", [(429, True), (500, True), (401, False)])
async def test_native_provider_http_errors(monkeypatch, provider, status, retryable):
    monkeypatch.setenv("TEST_KEY", "test-key")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(status))) as client:
        backend = make_backend(Tier(name="test", provider=provider, model="test", api_key_env="TEST_KEY"), client)
        with pytest.raises(BackendError) as caught:
            await accumulate(backend)
        assert caught.value.retryable == retryable


@pytest.mark.parametrize("provider", ["anthropic", "gemini"])
async def test_native_truncation_and_stream_errors(monkeypatch, provider):
    monkeypatch.setenv("TEST_KEY", "test-key")
    tier = Tier(name="test", provider=provider, model="test", api_key_env="TEST_KEY")
    partial = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "partial"}} if provider == "anthropic" else {"candidates": [{"content": {"parts": [{"text": "partial"}]}}]}
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, text=wire([partial])))) as client:
        stream = make_backend(tier, client).complete(REQUEST)
        assert (await anext(stream)).delta["content"] == "partial"
        with pytest.raises(BackendError):
            await anext(stream)
    failure = {"type": "error", "error": {"type": "overloaded_error"}} if provider == "anthropic" else {"error": {"code": 429}}
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, text=wire([failure])))) as client:
        with pytest.raises(BackendError) as caught:
            await accumulate(make_backend(tier, client))
        assert caught.value.retryable


@pytest.mark.parametrize("provider,updates", [
    ("anthropic", {"temperature": 1.5}),
    ("anthropic", {"seed": 42}),
    ("anthropic", {"response_format": {"type": "json_object"}}),
    ("gemini", {"parallel_tool_calls": False, "tools": TOOLS}),
])
async def test_unsupported_native_options_are_client_errors(provider, updates):
    async with httpx.AsyncClient() as client:
        backend = make_backend(Tier(name="test", provider=provider, model="test"), client)
        with pytest.raises(BackendError) as caught:
            await accumulate(backend, ChatRequest(messages=REQUEST.messages, **updates))
        assert caught.value.status == 400 and not caught.value.retryable


@pytest.mark.parametrize("name,url,model", [
    ("deepseek", "https://api.deepseek.com/v1", "deepseek-chat"),
    ("qwen", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
])
async def test_openai_compatible_custom_models_and_keys(monkeypatch, name, url, model):
    monkeypatch.setenv("CUSTOM_KEY", "test-key")

    def transport(request):
        assert str(request.url) == url + "/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload["model"] == model
        assert payload["enable_thinking"] is False
        assert "stream_options" not in payload
        assert "temperature" not in payload
        assert payload["max_completion_tokens"] == 256 and "max_tokens" not in payload
        assert payload["messages"][0]["reasoning_content"] == "prior reasoning"
        assert "provider_metadata" not in payload["messages"][0]
        return httpx.Response(200, text=wire([
            {"choices": [{"delta": {"reasoning_content": "reasoning"}}]},
            {"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]},
        ]) + "data: [DONE]\n\n")

    tier = Tier(name=name, provider="openai_compatible", model=model, base_url=url, api_key_env="CUSTOM_KEY", send_stream_usage=False, token_limit_parameter="max_completion_tokens", omit_parameters=["temperature"], extra_body={"enable_thinking": False})
    request = ChatRequest(messages=[{"role": "assistant", "content": "prior answer", "reasoning_content": "prior reasoning", "provider_metadata": {"anthropic": {}}}, *REQUEST.messages])
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        backend = make_backend(tier, client)
        assert isinstance(backend, OpenAIBackend)
        result = await accumulate(backend, request)
        assert result.content == "answer" and result.reasoning == "reasoning"
        with pytest.raises(BackendError, match="unsupported_parameter_temperature"):
            await accumulate(backend, ChatRequest(messages=REQUEST.messages, temperature=0))


def test_semantic_cache_rejects_omitted_temperature():
    config = load_settings("config/demo.yaml")
    data = config.model_dump()
    data["tiers"][0]["omit_parameters"] = ["temperature"]
    data["cache"]["semantic_enabled"] = True
    with pytest.raises(ValueError, match="temperature"):
        type(config).model_validate(data)


@pytest.mark.parametrize("partial", [False, True])
async def test_cross_provider_fallback_respects_first_output(settings, monkeypatch, partial):
    from gateway.app import create_app

    monkeypatch.setenv("TEST_KEY", "test-key")
    calls = []

    def transport(request):
        calls.append(request.url.host)
        if request.url.host == "api.anthropic.com":
            events = [{"type": "message_start", "message": {"usage": {"input_tokens": 1}}}]
            if partial:
                events.append({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "partial answer"}})
            events.append({"type": "error", "error": {"type": "overloaded_error"}})
            return httpx.Response(200, text=wire(events))
        return httpx.Response(200, text=wire([{"candidates": [{"content": {"parts": [{"text": "Gemini fallback"}]}, "finishReason": "STOP"}]}]))

    settings.cache.enabled = False
    settings.tiers[0] = Tier(name="local", provider="anthropic", model="claude-test", api_key_env="TEST_KEY")
    settings.tiers[1] = Tier(name="small", provider="gemini", model="gemini-test", api_key_env="TEST_KEY")
    settings.reliability.fallback_order = ["small"]
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as upstream:
        app = create_app(settings, backends={tier.name: make_backend(tier, upstream) for tier in settings.tiers})
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/v1/chat/completions", headers={"Authorization": "Bearer demo-key-change-me"}, json={"model": "local", "messages": [{"role": "user", "content": "Hello"}], "stream": True})
            assert response.status_code == 200
            if partial:
                assert "partial answer" in response.text and '"error"' in response.text
                assert calls == ["api.anthropic.com"]
            else:
                assert "Gemini fallback" in response.text and '"error"' not in response.text
                assert calls == ["api.anthropic.com", "generativelanguage.googleapis.com"]


def test_gemini_schema_and_sampling_translation():
    backend = GeminiBackend(Tier(name="gemini", provider="gemini", model="test"), None)
    schema = {"type": "object", "properties": {"value": {"type": "integer"}}}
    request = ChatRequest(messages=REQUEST.messages, seed=7, frequency_penalty=.2, response_format={"type": "json_schema", "json_schema": {"name": "value", "schema": schema}})
    generation = backend.payload(request)["generationConfig"]
    assert generation["responseJsonSchema"] == schema
    assert generation["responseMimeType"] == "application/json"
    assert generation["seed"] == 7 and generation["frequencyPenalty"] == .2


@pytest.mark.parametrize("provider", ["anthropic", "gemini"])
async def test_native_system_message_order_is_not_silently_changed(provider):
    request = ChatRequest(messages=[*REQUEST.messages, {"role": "system", "content": "late system instruction"}])
    async with httpx.AsyncClient() as client:
        with pytest.raises(BackendError, match="leading_system") as caught:
            await accumulate(make_backend(Tier(name="test", provider=provider, model="test"), client), request)
        assert caught.value.status == 400


async def test_gemini_opaque_metadata_does_not_bypass_text_contract():
    request = ChatRequest(messages=[{"role": "assistant", "content": "text", "provider_metadata": {"gemini": {"parts": [{"inlineData": {"mimeType": "image/png", "data": "test"}}]}}}, *REQUEST.messages])
    backend = GeminiBackend(Tier(name="gemini", provider="gemini", model="test"), None)
    with pytest.raises(BackendError, match="invalid_gemini_part_metadata"):
        backend.payload(request)
