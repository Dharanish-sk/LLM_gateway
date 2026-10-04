import asyncio
import os
import uuid

import pytest
from redis.asyncio import Redis

from gateway.state import RedisState


@pytest.mark.skipif(not os.getenv("TEST_REDIS_URL"), reason="Set TEST_REDIS_URL to test a real Redis server")
async def test_real_redis_lua_concurrency_and_cache():
    client = Redis.from_url(os.environ["TEST_REDIS_URL"], decode_responses=True)
    state = RedisState(client)
    identity = "integration-" + uuid.uuid4().hex
    try:
        results = await asyncio.gather(*(state.consume(identity, 20, .001) for _ in range(100)))
        assert sum(row.allowed for row in results) == 20
        await state.put(identity, "a", {"answer": "test"}, 30, 1)
        assert await state.get(identity, "a") == {"answer": "test"}
        await state.put(identity, "b", {"answer": "new"}, 30, 1)
        assert await state.get(identity, "a") is None
        assert len(await state.candidates(identity, 10)) == 1
    finally:
        index, entry_a = state.keys(identity, "a")
        _, entry_b = state.keys(identity, "b")
        await client.delete(f"gateway:bucket:{identity}", index, entry_a, entry_b)
        await state.close()
