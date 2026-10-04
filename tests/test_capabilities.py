import json
from contextlib import asynccontextmanager

import httpx
import pytest

from gateway.app import create_app
from gateway.backends import BackendError, BackendEvent, MockBackend, OpenAIBackend, make_backend
from gateway.cache import ResponseCache
from gateway.capabilities import (
    CapabilityRegistry,
    SelectionError,
    effective_capabilities,
    estimate_input_tokens,
)
from gateway.config import (
    ModelCapabilities,
    ReliabilitySettings,
    RouterSettings,
    Tier,
    load_settings,
)
from gateway.embeddings import DemoEmbedder
from gateway.models import ChatRequest
from gateway.reliability import BackendPool
from gateway.router import Router
from gateway.state import MemoryState
from gateway.telemetry import Telemetry

AUTH = {"Authorization": "Bearer demo-key-change-me"}
MESSAGES = [{"role": "user", "content": "Hello"}]
TOOL = {"type": "function", "function": {"name": "weather", "parameters": {"type": "object", "properties": {}}}}


def registry(tiers, *, order=None, fallback=True, retries=2):
    return CapabilityRegistry(tiers, ReliabilitySettings(fallback_order=order or [t.name for t in tiers], fallback_enabled=fallback, max_retries=retries))


@asynccontextmanager
async def api(settings, backends=None, embedder=None, state=None):
    app = create_app(settings, backends=backends, embedder=embedder, state=state)
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield app, client


class FailedBackend:
    def __init__(self, partial=False):
        self.calls = 0
        self.partial = partial

    async def complete(self, request):
        self.calls += 1
        if self.partial:
            yield BackendEvent({"content": "partial"})
        raise BackendError("injected_500")


@pytest.mark.parametrize("provider,changes,reason", [
    ("anthropic", {"response_format": {"type": "json_object"}}, "unsupported_json_object"),
    ("anthropic", {"seed": 3}, "unsupported_seed"),
    ("anthropic", {"temperature": 1.5}, "temperature_out_of_range"),
    ("ollama", {"tools": [TOOL]}, "unsupported_tools"),
    ("gemini", {"tools": [{"type": "function", "function": {"name": "weather", "strict": True}}]}, "unsupported_strict_tools"),
    ("gemini", {"tools": [TOOL], "parallel_tool_calls": False}, "unsupported_disable_parallel_tool_calls"),
])
def test_native_adapter_limits_cannot_be_overridden(provider, changes, reason):
    tier = Tier(name="chosen", provider=provider, model="test")
    request = ChatRequest(model="chosen", messages=MESSAGES, routing={"mode": "strict"}, **changes)
    with pytest.raises(SelectionError) as caught:
        registry([tier]).preflight(request)
    assert reason in caught.value.details["incompatible_tiers"]["chosen"]


def test_per_model_declared_limits_and_estimated_context():
    tier = Tier(name="model", provider="openai", model="test", capabilities=ModelCapabilities(json_schema=False, tools=False, max_output_tokens=32))
    request = ChatRequest(model="model", messages=MESSAGES, max_tokens=33)
    with pytest.raises(SelectionError) as caught:
        registry([tier]).preflight(request)
    assert caught.value.details["incompatible_tiers"]["model"] == ["output_token_limit_exceeded"]
    request = ChatRequest(model="model", messages=MESSAGES, max_tokens=32)
    count = estimate_input_tokens(request)
    tier.capabilities.context_window_tokens = count + 32
    assert registry([tier]).preflight(request).eligible == ("model",)
    tier.capabilities.context_window_tokens -= 1
    with pytest.raises(SelectionError, match="No configured tier"):
        registry([tier]).preflight(request)
    unicode_request = ChatRequest(model="model", messages=[{"role": "user", "content": "你好" * 100}], max_tokens=32)
    assert estimate_input_tokens(unicode_request) > 600
    with_tools = ChatRequest(model="model", messages=MESSAGES, tools=[TOOL], max_tokens=32)
    assert estimate_input_tokens(with_tools) > count


def test_omitted_defaults_and_explicit_controls_remain_distinct():
    tier = Tier(name="model", provider="openai", model="test", omit_parameters=["temperature", "top_p"])
    implicit = ChatRequest(model="model", messages=MESSAGES)
    assert registry([tier]).preflight(implicit).eligible == ("model",)
    explicit = ChatRequest(model="model", messages=MESSAGES, temperature=0)
    with pytest.raises(SelectionError) as caught:
        registry([tier]).preflight(explicit)
    assert "unsupported_temperature" in caught.value.details["incompatible_tiers"]["model"]


async def test_auto_json_object_routes_from_claude_to_gemini_before_provider_io(settings, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "unused-test-key")
    settings.tiers = [
        Tier(name="local", provider="ollama", model="local"),
        Tier(name="small", provider="gemini", model="gemini", api_key_env="TEST_KEY"),
        Tier(name="large", provider="anthropic", model="claude", api_key_env="TEST_KEY"),
    ]
    settings.reliability.fallback_order = ["large", "small", "local"]
    calls = []

    def transport(request):
        calls.append(request.url.host)
        assert request.url.host == "generativelanguage.googleapis.com"
        assert json.loads(request.content)["generationConfig"]["responseMimeType"] == "application/json"
        event = {"candidates": [{"content": {"parts": [{"text": '{"answer":"hello"}'}]}, "finishReason": "STOP"}]}
        return httpx.Response(200, text=f"data: {json.dumps(event)}\n\n")

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as upstream:
        backends = {tier.name: make_backend(tier, upstream) for tier in settings.tiers}
        async with api(settings, backends) as (_, client):
            result = await client.post("/v1/chat/completions", headers=AUTH, json={"messages": MESSAGES, "response_format": {"type": "json_object"}})
            assert result.status_code == 200
            assert result.json()["model"] == "small"
            assert calls == ["generativelanguage.googleapis.com"]
            assert 'gateway_capability_reroutes_total{from_tier="large",to_tier="small"} 1.0' in (await client.get("/metrics")).text


async def test_no_compatible_tier_rejected_before_cache_embeddings_and_backends(settings):
    class NoEmbed:
        model = "demo-hash-64"

        async def embed(self, text):
            pytest.fail("Embedding called for an impossible request")

    class NoCacheReads(MemoryState):
        async def get(self, *args):
            pytest.fail("Cache read for an impossible request")

    for tier in settings.tiers:
        tier.capabilities.json_object = False
    async with api(settings, embedder=NoEmbed(), state=NoCacheReads()) as (app, client):
        result = await client.post("/v1/chat/completions", headers=AUTH, json={"messages": MESSAGES, "response_format": {"type": "json_object"}})
        assert result.status_code == 400
        assert result.json()["error"]["code"] == "no_compatible_tier"
        assert set(result.json()["error"]["details"]["incompatible_tiers"]) == {"local", "small", "large"}
        assert sum(b.calls for b in app.state.pool.backends.values()) == 0
        assert app.state.active == 0


@pytest.mark.parametrize("stream", [False, True])
async def test_strict_pin_never_falls_back_even_on_retryable_failure(settings, stream):
    bad = FailedBackend()
    backends = {tier.name: MockBackend(tier) for tier in settings.tiers}
    backends["local"] = bad
    async with api(settings, backends) as (_, client):
        result = await client.post("/v1/chat/completions", headers=AUTH, json={"model": "local", "messages": MESSAGES, "stream": stream, "routing": {"mode": "strict"}})
        assert result.status_code == 502
        assert bad.calls == 1
        assert backends["small"].calls == backends["large"].calls == 0


async def test_strict_pin_rejects_incompatible_primary_without_trying_compatible_backup(settings):
    settings.tiers[0].capabilities.tools = False
    async with api(settings) as (app, client):
        result = await client.post("/v1/chat/completions", headers=AUTH, json={"model": "local", "messages": MESSAGES, "tools": [TOOL], "routing": {"mode": "strict"}})
        assert result.status_code == 400
        assert sum(b.calls for b in app.state.pool.backends.values()) == 0


async def test_allowlist_limits_substitution_and_failure_fallback(settings):
    bad = FailedBackend()
    backends = {tier.name: MockBackend(tier) for tier in settings.tiers}
    backends["local"] = bad
    async with api(settings, backends) as (_, client):
        result = await client.post("/v1/chat/completions", headers=AUTH, json={"model": "local", "messages": MESSAGES, "routing": {"fallbacks": ["small"]}})
        assert result.status_code == 200 and result.json()["model"] == "small"
        assert backends["large"].calls == 0


@pytest.mark.parametrize("blocked_by", ["global_disable", "retry_zero", "client_empty"])
async def test_explicit_incompatible_model_cannot_escape_disabled_fallback(settings, blocked_by):
    settings.tiers[0].capabilities.json_schema = False
    routing = {}
    if blocked_by == "global_disable":
        settings.reliability.fallback_enabled = False
    elif blocked_by == "retry_zero":
        settings.reliability.max_retries = 0
    else:
        routing = {"fallbacks": []}
    body = {"model": "local", "messages": MESSAGES, "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}, "routing": routing}
    async with api(settings) as (app, client):
        response = await client.post("/v1/chat/completions", headers=AUTH, json=body)
        assert response.status_code == 400
        assert sum(b.calls for b in app.state.pool.backends.values()) == 0


async def test_auto_allowlist_bounds_primary_pool_with_retries_disabled(settings):
    settings.reliability.fallback_enabled = False
    async with api(settings) as (app, client):
        response = await client.post("/v1/chat/completions", headers=AUTH, json={"messages": MESSAGES, "routing": {"fallbacks": ["small"]}})
        assert response.status_code == 200 and response.json()["model"] == "small"
        assert app.state.pool.backends["local"].calls == 0


async def test_skipping_incompatible_fallback_does_not_consume_retry_budget(settings):
    settings.reliability.max_retries = 1
    settings.tiers[2].capabilities.strict_tools = False
    bad = FailedBackend()
    backends = {tier.name: MockBackend(tier) for tier in settings.tiers}
    backends["local"] = bad
    strict_tool = {"type": "function", "function": {"name": "weather", "strict": True}}
    async with api(settings, backends) as (_, client):
        result = await client.post("/v1/chat/completions", headers=AUTH, json={"model": "local", "messages": MESSAGES, "tools": [strict_tool]})
        assert result.status_code == 200 and result.json()["model"] == "small"
        assert bad.calls == 1 and backends["large"].calls == 0 and backends["small"].calls == 1


async def test_partial_stream_never_switches_to_eligible_fallback(settings):
    bad = FailedBackend(partial=True)
    backends = {tier.name: MockBackend(tier) for tier in settings.tiers}
    backends["local"] = bad
    async with api(settings, backends) as (_, client):
        result = await client.post("/v1/chat/completions", headers=AUTH, json={"model": "local", "messages": MESSAGES, "stream": True, "routing": {"fallbacks": ["small"]}})
        assert "partial" in result.text and '"error"' in result.text
        assert backends["small"].calls == backends["large"].calls == 0


async def test_cache_isolates_strict_and_fallback_policies_and_rechecks_actual_model(settings):
    bad = FailedBackend()
    backends = {tier.name: MockBackend(tier) for tier in settings.tiers}
    backends["local"] = bad
    async with api(settings, backends) as (app, client):
        body = {"model": "local", "messages": MESSAGES}
        first = await client.post("/v1/chat/completions", headers=AUTH, json=body)
        assert first.json()["model"] == "large"
        assert (await client.post("/v1/chat/completions", headers=AUTH, json=body)).headers["x-cache"] == "exact"
        pinned = {**body, "routing": {"mode": "strict"}}
        assert (await client.post("/v1/chat/completions", headers=AUTH, json=pinned)).status_code == 502
        # Even a stale/misplaced response in the pinned namespace is not accepted.
        context = app.state.cache.context(ChatRequest.model_validate(pinned), "demo")
        await app.state.cache.store(context, first.json())
        assert (await client.post("/v1/chat/completions", headers=AUTH, json=pinned)).status_code == 502
        allowed = {**body, "routing": {"fallbacks": ["small"]}}
        result = await client.post("/v1/chat/completions", headers=AUTH, json=allowed)
        assert result.json()["model"] == "small" and result.headers["x-cache"] == "miss"


async def test_semantic_cache_skips_ineligible_actual_tier(settings):
    settings.cache.semantic_enabled = True
    cache = ResponseCache(MemoryState(), DemoEmbedder(), settings.cache, "test")
    request = ChatRequest(messages=[{"role": "user", "content": "hello world"}])
    _, _, context = await cache.lookup(request, "tenant")
    await cache.store(context, {"model": "large", "answer": "cached"})
    paraphrase = ChatRequest(messages=[{"role": "user", "content": "world hello"}])
    assert (await cache.lookup(paraphrase, "tenant"))[1] == "semantic"
    assert (await cache.lookup(paraphrase, "tenant", {"small"}))[0] is None


@pytest.mark.parametrize("routing", [
    {"mode": "strict", "fallbacks": ["small"]},
    {"fallbacks": ["small", "small"]},
    {"fallbacks": ["unknown"]},
])
async def test_invalid_routing_policies_are_client_errors(settings, routing):
    async with api(settings) as (app, client):
        result = await client.post("/v1/chat/completions", headers=AUTH, json={"model": "local", "messages": MESSAGES, "routing": routing})
        assert result.status_code == 400
        assert sum(b.calls for b in app.state.pool.backends.values()) == 0


def test_strict_auto_is_invalid():
    with pytest.raises(ValueError, match="explicit model"):
        ChatRequest(messages=MESSAGES, routing={"mode": "strict"})


async def test_classifier_failure_cannot_bypass_capabilities():
    class BrokenEmbedding:
        model = "test"

        async def embed(self, text):
            raise httpx.ConnectError("offline")

    tiers = [Tier(name=name, provider="mock", model=name) for name in ("local", "small", "large")]
    tiers[2].capabilities.max_output_tokens = 1
    rules = RouterSettings()
    registry_ = CapabilityRegistry(tiers, ReliabilitySettings(fallback_order=["large", "small", "local"]), rules)
    router = Router(rules, BrokenEmbedding(), registry_)
    router.artifact = {"schema_version": 1}
    decision = await router.choose(ChatRequest(messages=MESSAGES))
    assert decision.tier == "small" and decision.preferred_tier == "large"
    assert decision.reason == "capability_reroute"


async def test_classifier_high_probability_cannot_select_an_incompatible_small_tier():
    class Embedding:
        model = "test"

        async def embed(self, text):
            return [1, 0]

    tiers = [Tier(name=name, provider="mock", model=name) for name in ("local", "small", "large")]
    tiers[0].capabilities.max_output_tokens = 1
    rules = RouterSettings()
    registry_ = CapabilityRegistry(tiers, ReliabilitySettings(fallback_order=["large", "small", "local"]), rules)
    router = Router(rules, Embedding(), registry_)
    router.artifact = {"mean": [0, 0, 0], "scale": [1, 1, 1], "coef": [0, 0, 0], "intercept": 100}
    decision = await router.choose(ChatRequest(messages=MESSAGES))
    assert decision.preferred_tier == "local" and decision.tier == "large"
    assert decision.probability > .99


def test_unknown_compatible_endpoints_need_explicit_advanced_capabilities():
    tier = Tier(name="custom", provider="openai_compatible", model="custom", base_url="http://example.test/v1")
    assert registry([tier]).preflight(ChatRequest(model="custom", messages=MESSAGES)).eligible == ("custom",)
    with pytest.raises(SelectionError):
        registry([tier]).preflight(ChatRequest(model="custom", messages=MESSAGES, response_format={"type": "json_object"}))
    tier.capabilities.json_object = True
    assert registry([tier]).preflight(ChatRequest(model="custom", messages=MESSAGES, response_format={"type": "json_object"})).eligible == ("custom",)


async def test_pool_independently_enforces_pin_and_capabilities():
    tiers = [Tier(name="one", provider="mock", model="one"), Tier(name="two", provider="mock", model="two")]
    backends = {tier.name: MockBackend(tier) for tier in tiers}
    telemetry = Telemetry()
    pool = BackendPool(tiers, backends, ReliabilitySettings(fallback_order=["one", "two"]), telemetry)
    request = ChatRequest(model="one", messages=MESSAGES, routing={"mode": "strict"})
    try:
        with pytest.raises(SelectionError):
            await anext(pool.complete(request, "two"))
        assert backends["two"].calls == 0
    finally:
        telemetry.close()


async def test_pool_cannot_substitute_explicit_model_when_server_disables_fallback():
    tiers = [Tier(name="one", provider="mock", model="one"), Tier(name="two", provider="mock", model="two")]
    backends = {tier.name: MockBackend(tier) for tier in tiers}
    telemetry = Telemetry()
    pool = BackendPool(tiers, backends, ReliabilitySettings(fallback_enabled=False, fallback_order=["two"]), telemetry)
    try:
        with pytest.raises(SelectionError):
            await anext(pool.complete(ChatRequest(model="one", messages=MESSAGES), "two"))
        assert backends["two"].calls == 0
    finally:
        telemetry.close()


async def test_client_allowlist_cannot_add_server_disallowed_fallback(settings):
    settings.reliability.fallback_order = ["large"]
    bad = FailedBackend()
    backends = {tier.name: MockBackend(tier) for tier in settings.tiers}
    backends["local"] = bad
    async with api(settings, backends) as (_, client):
        result = await client.post("/v1/chat/completions", headers=AUTH, json={"model": "local", "messages": MESSAGES, "routing": {"fallbacks": ["small"]}})
        assert result.status_code == 502
        assert backends["large"].calls == backends["small"].calls == 0


async def test_gateway_policy_is_not_sent_to_hosted_provider():
    def transport(request):
        assert "routing" not in json.loads(request.content)
        return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        backend = OpenAIBackend(Tier(name="chosen", provider="openai", model="test"), client)
        request = ChatRequest(model="chosen", messages=MESSAGES, routing={"mode": "strict"})
        assert [event async for event in backend.complete(request)]


async def test_models_endpoint_exposes_effective_constraints(settings):
    settings.tiers[0] = Tier(name="local", provider="ollama", model="test", capabilities=ModelCapabilities(tools=True, max_output_tokens=100))
    async with api(settings) as (_, client):
        response = await client.get("/v1/models", headers=AUTH)
        local = next(row for row in response.json()["data"] if row["id"] == "local")
        assert local["capabilities"]["tools"] is False
        assert local["capabilities"]["max_output_tokens"] == 100
        assert "api_key_env" not in response.text


def test_provider_defaults_document_effective_hard_limits():
    settings = load_settings("config/providers.yaml")
    descriptions = {tier.name: effective_capabilities(tier) for tier in settings.tiers}
    assert descriptions["claude"]["json_object"] is False
    assert descriptions["gemini"]["disable_parallel_tool_calls"] is False
