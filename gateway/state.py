from __future__ import annotations

import asyncio
import json
import math
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from redis.asyncio import Redis


@dataclass
class BucketResult:
    allowed: bool
    remaining: float
    retry_after: float


TOKEN_BUCKET_LUA = """
local now
if ARGV[3] ~= '' then now = tonumber(ARGV[3])
else
  local t = redis.call('TIME')
  now = tonumber(t[1]) + tonumber(t[2]) / 1000000
end
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local state = redis.call('HMGET', KEYS[1], 'tokens', 'updated')
local tokens = tonumber(state[1]) or capacity
local updated = tonumber(state[2]) or now
now = math.max(now, updated)
tokens = math.min(capacity, tokens + (now - updated) * rate)
local allowed = 0
local wait = 0
if tokens >= 1 then tokens = tokens - 1; allowed = 1
else wait = (1 - tokens) / rate end
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'updated', tostring(now))
redis.call('EXPIRE', KEYS[1], math.ceil(capacity / rate) + 1)
return {allowed, tostring(tokens), tostring(wait)}
"""

CACHE_PUT_LUA = """
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
redis.call('SET', KEYS[2], ARGV[1], 'EX', ARGV[2])
redis.call('ZADD', KEYS[1], now + tonumber(ARGV[2]), KEYS[2])
local excess = redis.call('ZCARD', KEYS[1]) - tonumber(ARGV[3])
if excess > 0 then
  local old = redis.call('ZRANGE', KEYS[1], 0, excess - 1)
  for _, k in ipairs(old) do redis.call('DEL', k); redis.call('ZREM', KEYS[1], k) end
end
redis.call('EXPIRE', KEYS[1], ARGV[2])
return 1
"""


class MemoryState:
    """Single-process demo/test state; deliberately prohibited in production config."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self.buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self.entries: OrderedDict[str, tuple[str, float, dict]] = OrderedDict()
        self.lock = asyncio.Lock()

    async def consume(self, key: str, capacity: int, rate: float) -> BucketResult:
        async with self.lock:
            now = self.clock()
            tokens, previous = self.buckets.get(key, (float(capacity), now))
            now = max(now, previous)
            tokens = min(float(capacity), tokens + (now - previous) * rate)
            allowed = tokens >= 1
            if allowed:
                tokens -= 1
            self.buckets[key] = (tokens, now)
            self.buckets.move_to_end(key)
            if len(self.buckets) > 10000:
                self.buckets.popitem(last=False)
            return BucketResult(allowed, tokens, 0 if allowed else (1 - tokens) / rate)

    async def get(self, namespace: str, key: str) -> dict | None:
        entry = self.entries.get(f"{namespace}:{key}")
        if entry and entry[1] > self.clock():
            return entry[2]
        return None

    async def candidates(self, namespace: str, limit: int) -> list[dict]:
        now = self.clock()
        return [e[2] for e in self.entries.values() if e[0] == namespace and e[1] > now][-limit:]

    async def put(self, namespace: str, key: str, value: dict, ttl: int, limit: int):
        async with self.lock:
            full = f"{namespace}:{key}"
            self.entries[full] = (namespace, self.clock() + ttl, value)
            self.entries.move_to_end(full)
            own = [k for k, v in self.entries.items() if v[0] == namespace]
            for old in own[:-limit]:
                del self.entries[old]
            while len(self.entries) > 10000:
                self.entries.popitem(last=False)

    async def ping(self):
        return True

    async def close(self):
        pass


class RedisState:
    def __init__(self, client: Redis, clock: Callable[[], float] | None = None):
        self.client = client
        self.clock = clock  # Only tests inject a clock; production uses Redis TIME inside Lua.

    async def consume(self, key: str, capacity: int, rate: float) -> BucketResult:
        result = await self.client.eval(
            TOKEN_BUCKET_LUA, 1, f"gateway:bucket:{key}", capacity, rate,
            "" if self.clock is None else self.clock(),
        )
        return BucketResult(bool(int(result[0])), float(result[1]), float(result[2]))

    @staticmethod
    def keys(namespace: str, key: str = "") -> tuple[str, str]:
        # Hash tags keep Lua keys in one Redis Cluster slot.
        prefix = f"gateway:cache:{{{namespace}}}"
        return f"{prefix}:index", f"{prefix}:entry:{key}"

    async def get(self, namespace: str, key: str) -> dict | None:
        raw = await self.client.get(self.keys(namespace, key)[1])
        return json.loads(raw) if raw else None

    async def candidates(self, namespace: str, limit: int) -> list[dict]:
        keys = await self.client.zrange(self.keys(namespace)[0], -limit, -1)
        if not keys:
            return []
        values = await self.client.mget(keys)
        return [json.loads(value) for value in values if value]

    async def put(self, namespace: str, key: str, value: dict, ttl: int, limit: int):
        index, entry = self.keys(namespace, key)
        await self.client.eval(CACHE_PUT_LUA, 2, index, entry, json.dumps(value), ttl, limit)

    async def ping(self):
        return await self.client.ping()

    async def close(self):
        await self.client.aclose()


def retry_seconds(result: BucketResult) -> str:
    return str(max(1, math.ceil(result.retry_after)))
