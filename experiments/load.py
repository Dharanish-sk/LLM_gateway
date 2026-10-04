import asyncio
import json
import os
import socket
import time
from contextlib import asynccontextmanager

import httpx
import uvicorn

from experiments.common import append_jsonl, percentile, provenance, write_table


@asynccontextmanager
async def serve_app(app):
    """Real loopback TCP, so streaming timings include network/ASGI behavior."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    port = sock.getsockname()[1]
    config = uvicorn.Config(app, log_level="error", access_log=False, lifespan="on")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(10):
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("Embedded gateway failed to start")
                await asyncio.sleep(.01)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await task
        sock.close()


async def one_request(client, url, key, index, model="auto", prompt="What is a gateway?"):
    start, first, status, failure = time.perf_counter(), None, None, None
    done, finished = False, False
    try:
        async with client.stream("POST", url.rstrip("/") + "/v1/chat/completions", headers={"Authorization": f"Bearer {key}"}, json={
            "model": model, "messages": [{"role": "user", "content": f"{prompt} Request {index}."}],
            "user": f"experiment-{index}", "stream": True, "max_tokens": 200,
        }) as response:
            status = response.status_code
            if response.is_error:
                failure = f"http_{status}"
                await response.aread()
            else:
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if raw == "[DONE]":
                        done = True
                        continue
                    event = json.loads(raw)
                    if "error" in event:
                        failure = event["error"].get("code", "stream_error")
                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {})
                        if first is None and any(delta.get(k) for k in ("content", "tool_calls", "reasoning_content")):
                            first = time.perf_counter() - start
                        if choice.get("finish_reason"):
                            finished = True
                if not done or not finished:
                    failure = failure or "incomplete_stream"
    except (httpx.HTTPError, ValueError) as exc:
        failure = type(exc).__name__
    return {"index": index, "status": status, "error": failure, "latency_s": time.perf_counter() - start, "ttft_s": first}


async def level(url, key, users, count, *, model="auto", offset=0):
    semaphore = asyncio.Semaphore(users)
    async with httpx.AsyncClient(timeout=300, limits=httpx.Limits(max_connections=users, max_keepalive_connections=users), trust_env=False) as client:
        async def task(index):
            async with semaphore:
                return await one_request(client, url, key, index + offset, model)

        start = time.perf_counter()
        rows = await asyncio.gather(*(task(i) for i in range(count)))
        elapsed = time.perf_counter() - start
    successes = [r for r in rows if not r["error"]]
    return {
        "concurrency": users, "n": len(rows), "success_n": len(successes),
        "throughput_rps": len(rows) / elapsed, "success_rps": len(successes) / elapsed,
        "p50_s": percentile([r["latency_s"] for r in rows], 50),
        "p95_s": percentile([r["latency_s"] for r in rows], 95),
        "success_p95_s": percentile([r["latency_s"] for r in successes], 95),
        "error_rate": 1 - len(successes) / len(rows),
    }, rows


async def run(args):
    async def measure(url, key):
        summaries = []
        run = provenance(demo=args.demo, url=url, cache_busted=True, rate_limit_changed=args.demo)
        for index, users in enumerate(args.users):
            summary, rows = await level(url, key, users, args.requests, model=args.model, offset=index * args.requests)
            summaries.append(summary)
            for row in rows:
                append_jsonl(args.output + ".jsonl", {**run, "concurrency": users, **row})
        write_table(args.output, summaries, "DEMO ONLY — loopback mock streaming load" if args.demo else "Gateway streaming load (all-request latency includes errors)")

    if args.demo:
        from gateway.app import create_app
        from gateway.config import load_settings
        config = load_settings("config/demo.yaml")
        config.cache.enabled = False
        config.bucket_capacity = args.requests * len(args.users) + 100
        config.bucket_refill_rate = 10000
        async with serve_app(create_app(config)) as url:
            await measure(url, os.getenv("GATEWAY_DEMO_KEY", "demo-key-change-me"))
    else:
        key = os.environ[args.key_env]
        await measure(args.url, key)
