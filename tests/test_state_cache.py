import asyncio

import fakeredis.aioredis
import pytest

from gateway.cache import ResponseCache
from gateway.config import CacheSettings
from gateway.embeddings import DemoEmbedder
from gateway.models import ChatRequest
from gateway.state import MemoryState, RedisState


@pytest.mark.parametrize("redis", [False, True])
async def test_exact_token_bucket_arithmetic(redis):
    now = [100.0]
    store = RedisState(fakeredis.aioredis.FakeRedis(), lambda: now[0]) if redis else MemoryState(lambda: now[0])
    results = [await store.consume("key", 20, 2) for _ in range(20)]
    assert all(result.allowed for result in results)
    assert [r.remaining for r in results] == list(range(19, -1, -1))
    rejected = await store.consume("key", 20, 2)
    assert (rejected.allowed, rejected.remaining, rejected.retry_after) == (False, 0, 0.5)
    now[0] += 5
    refilled = await store.consume("key", 20, 2)
    assert refilled.allowed
    assert refilled.remaining + 1 == 10  # Exactly ten before consuming the new request.
    assert refilled.remaining == 9
    now[0] += 100
    assert (await store.consume("key", 20, 2)).remaining == 19
    await store.close()


async def test_redis_atomic_concurrency_and_tenant_bucket_isolation():
    store = RedisState(fakeredis.aioredis.FakeRedis(), lambda: 100)
    results = await asyncio.gather(*(store.consume("same", 20, 2) for _ in range(100)))
    assert sum(result.allowed for result in results) == 20
    assert (await store.consume("other", 20, 2)).remaining == 19
    await store.close()


@pytest.mark.parametrize("redis", [False, True])
async def test_cache_isolation_parameters_and_bounded_storage(redis):
    state = RedisState(fakeredis.aioredis.FakeRedis()) if redis else MemoryState()
    cache = ResponseCache(state, DemoEmbedder(), CacheSettings(semantic_enabled=True, max_entries_per_namespace=2), "v1")
    request = ChatRequest(messages=[{"role": "system", "content": "Be brief"}, {"role": "user", "content": "capital of France"}])
    found, status, context = await cache.lookup(request, "a")
    assert found is None and status == "miss"
    await cache.store(context, {"answer": "Paris"})
    assert (await cache.lookup(request, "a"))[1] == "exact"
    assert (await cache.lookup(request, "b"))[0] is None
    for change in ({"temperature": 1}, {"max_tokens": 100}, {"seed": 42}, {"tools": [{"type": "function"}]}):
        assert (await cache.lookup(request.model_copy(update=change), "a"))[0] is None
    other_system = request.model_copy(deep=True)
    other_system.messages[0].content = "Be verbose"
    assert (await cache.lookup(other_system, "a"))[0] is None
    paraphrase = request.model_copy(deep=True)
    paraphrase.messages[1].content = "France capital of"
    assert (await cache.lookup(paraphrase, "a"))[1] == "semantic"
    for index in range(5):
        await state.put("scope", str(index), {"index": index}, 60, 2)
    assert len(await state.candidates("scope", 100)) == 2
    assert await state.get("scope", "0") is None
    await state.close()


async def test_cache_ttl_and_multiturn_eligibility():
    now = [0]
    state = MemoryState(lambda: now[0])
    await state.put("scope", "key", {"answer": 1}, 10, 20)
    now[0] = 10
    assert await state.get("scope", "key") is None
    assert not await state.candidates("scope", 20)
    request = ChatRequest(messages=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}, {"role": "user", "content": "Why?"}])
    assert request.semantic_text() is None


async def test_cache_preserves_explicit_sampling_for_provider_capability_checks():
    cache = ResponseCache(MemoryState(), DemoEmbedder(), CacheSettings(), "policy")
    implicit = ChatRequest(messages=[{"role": "user", "content": "Hello"}])
    _, _, context = await cache.lookup(implicit, "tenant")
    await cache.store(context, {"answer": "cached"})
    explicit = ChatRequest(messages=implicit.messages, temperature=0)
    # An omitted model-specific default must not satisfy an explicitly requested
    # parameter that this tier would reject on an uncached request.
    assert (await cache.lookup(explicit, "tenant"))[0] is None
