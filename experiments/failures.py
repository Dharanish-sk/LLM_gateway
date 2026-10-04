import asyncio
import os
import random

import httpx

from experiments.common import append_jsonl, provenance, write_table
from experiments.load import level, serve_app
from gateway.app import create_app
from gateway.backends import BackendError, BackendEvent, MockBackend
from gateway.config import load_settings


class InjectedBackend:
    def __init__(self, backend, schedule: dict[int, str], delay=.05):
        self.backend, self.schedule, self.delay = backend, schedule, delay

    async def complete(self, request):
        index = int((request.user or "experiment--1").removeprefix("experiment-"))
        mode = self.schedule.get(index)
        if mode == "slow":
            await asyncio.sleep(self.delay)
        elif mode == "500":
            raise BackendError("injected_500")
        elif mode == "429":
            raise BackendError("injected_429", 503)
        elif mode == "drop":
            raise httpx.ReadError("injected drop before output")
        elif mode == "partial":
            yield BackendEvent({"content": "partial output"})
            raise httpx.ReadError("injected drop after output")
        async for event in self.backend.complete(request):
            yield event


async def run(args):
    indices = list(range(args.requests))
    random.Random(args.seed).shuffle(indices)
    bad = indices[:round(args.requests * args.failure_rate)]
    modes = ["500", "429", "slow", "drop", "partial"] if args.fault == "mixed" else [args.fault]
    schedule = {index: modes[i % len(modes)] for i, index in enumerate(bad)}
    summaries = []
    for fallback in (False, True):
        config = load_settings("config/demo.yaml")
        config.cache.enabled = False
        config.bucket_capacity, config.bucket_refill_rate = args.requests + 100, 10000
        config.reliability.fallback_enabled = fallback
        config.reliability.failure_threshold = args.breaker_threshold
        config.reliability.cooldown_seconds = .1
        config.tiers[-1].ttft_timeout = .02
        backends = {tier.name: MockBackend(tier) for tier in config.tiers}
        backends["large"] = InjectedBackend(backends["large"], schedule)
        async with serve_app(create_app(config, backends=backends)) as url:
            summary, rows = await level(url, os.getenv("GATEWAY_DEMO_KEY", "demo-key-change-me"), args.users, args.requests, model="large")
        summaries.append({"fallback": fallback, "fault": args.fault, "scheduled_faults": len(bad), **summary})
        for row in rows:
            append_jsonl(args.output + ".jsonl", {**provenance(demo=True, seed=args.seed), "fallback": fallback, "scheduled_fault": schedule.get(row["index"]), **row})
    write_table(args.output, summaries, "DEMO ONLY — injected backend faults over real loopback HTTP; circuit breaker enabled")
