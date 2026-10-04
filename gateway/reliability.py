from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from contextlib import aclosing
from dataclasses import dataclass

import httpx

from gateway.backends import BackendError
from gateway.capabilities import CapabilityRegistry
from gateway.config import ReliabilitySettings, Tier


@dataclass
class Circuit:
    failures: int = 0
    opened_at: float | None = None
    probe_active: bool = False


class CircuitBreaker:
    def __init__(self, threshold: int, cooldown: float, clock: Callable = time.monotonic):
        self.threshold, self.cooldown, self.clock = threshold, cooldown, clock
        self.circuits: dict[str, Circuit] = {}
        self.lock = asyncio.Lock()

    async def acquire(self, tier: str) -> bool:
        async with self.lock:
            circuit = self.circuits.setdefault(tier, Circuit())
            if circuit.opened_at is None:
                return True
            if self.clock() - circuit.opened_at < self.cooldown or circuit.probe_active:
                return False
            circuit.probe_active = True
            return True

    async def success(self, tier: str):
        async with self.lock:
            self.circuits[tier] = Circuit()

    async def failure(self, tier: str):
        async with self.lock:
            circuit = self.circuits.setdefault(tier, Circuit())
            circuit.failures += 1
            if circuit.failures >= self.threshold or circuit.probe_active:
                circuit.opened_at = self.clock()
            circuit.probe_active = False

    async def release(self, tier: str):
        async with self.lock:
            self.circuits.setdefault(tier, Circuit()).probe_active = False


class BackendPool:
    def __init__(self, tiers: list[Tier], backends: dict, config: ReliabilitySettings, telemetry, capabilities=None):
        self.tiers = {tier.name: tier for tier in tiers}
        self.backends, self.config, self.telemetry = backends, config, telemetry
        self.breaker = CircuitBreaker(config.failure_threshold, config.cooldown_seconds)
        self.capabilities = capabilities or CapabilityRegistry(tiers, config)

    async def complete(self, request, preferred: str, parent=None):
        self.capabilities.validate_aliases(request)
        order = self.capabilities.retry_order(request, preferred) if preferred in self.capabilities.initial_order(request) else []
        report = self.capabilities.evaluate(request, order)
        self.capabilities.require_eligible(report)
        order = report.eligible
        attempts = 0
        last_error = BackendError("all_backends_unavailable", 503)
        request_deadline = time.monotonic() + self.config.request_timeout
        for name in order:
            if attempts >= self.config.max_retries + 1:
                break
            if not await self.breaker.acquire(name):
                self.telemetry.circuit_skips.labels(name).inc()
                continue
            attempts += 1
            if attempts > 1:
                self.telemetry.fallbacks.labels(preferred, name).inc()
            tier = self.tiers[name]
            start = time.monotonic()
            deadline = min(request_deadline, start + tier.total_timeout)
            started = False
            failed = False
            with self.telemetry.span("backend", parent, tier=name, attempt=attempts) as span:
                try:
                    async with aclosing(self.backends[name].complete(request)) as iterator:
                        while True:
                            timeout = min(tier.idle_timeout if started else tier.ttft_timeout, deadline - time.monotonic())
                            if timeout <= 0:
                                raise TimeoutError
                            try:
                                async with asyncio.timeout(timeout):
                                    event = await anext(iterator)
                            except StopAsyncIteration:
                                if not started:
                                    raise BackendError("empty_upstream_stream") from None
                                break
                            if not started:
                                self.telemetry.ttft.labels(name).observe(time.monotonic() - start)
                            started = True
                            yield name, event
                    await self.breaker.success(name)
                    self.telemetry.backend_calls.labels(name, "success").inc()
                    return
                except (TimeoutError, httpx.TimeoutException) as exc:
                    last_error = BackendError("upstream_timeout", 504)
                    last_error.__cause__ = exc
                    failed = True
                except (httpx.TransportError, BackendError) as exc:
                    last_error = exc if isinstance(exc, BackendError) else BackendError("upstream_connection_error")
                    failed = True
                except (ValueError, KeyError, TypeError) as exc:
                    last_error = BackendError("invalid_upstream_payload")
                    last_error.__cause__ = exc
                    failed = True
                finally:
                    if not failed:
                        await self.breaker.release(name)
                if failed:
                    if last_error.retryable:
                        await self.breaker.failure(name)
                    else:
                        await self.breaker.release(name)
                    span.set_attribute("error.type", last_error.kind)
                    self.telemetry.backend_calls.labels(name, last_error.kind).inc()
                    if started or not last_error.retryable:
                        raise last_error
        raise last_error
