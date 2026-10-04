import asyncio
import random
import time
from pathlib import Path

import httpx
import psutil
import yaml

from experiments.common import (
    append_jsonl,
    fingerprint,
    percentile,
    provenance,
    read_jsonl,
    write_table,
)
from experiments.data import DATA_VERSION, prompts
from gateway.backends import make_backend
from gateway.config import Tier
from gateway.models import ChatRequest
from gateway.protocol import Accumulator


async def sample_rss(pid: int | None, stop: asyncio.Event, samples: list[int]):
    if pid is None:
        return
    process = psutil.Process(pid)
    while not stop.is_set():
        try:
            processes = [process, *process.children(recursive=True)]
            samples.append(sum(item.memory_info().rss for item in processes if item.is_running()))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.05)
        except TimeoutError:
            pass


async def generate(backend, request, pid=None, deadline_seconds=300):
    stop, memory = asyncio.Event(), []
    monitor = asyncio.create_task(sample_rss(pid, stop, memory))
    start, first = time.perf_counter(), None
    accumulator = Accumulator()
    failure = None
    try:
        async with asyncio.timeout(deadline_seconds):
            async for event in backend.complete(request):
                if first is None and (event.delta or event.finish_reason):
                    first = time.perf_counter()
                accumulator.add(event)
        if not accumulator.finished:
            failure = "missing_finish"
    except Exception as exc:
        failure = type(exc).__name__ + ":" + getattr(exc, "kind", "generation_failed")
    finally:
        elapsed = time.perf_counter() - start
        stop.set()
        await monitor
    usage, source = accumulator.effective_usage(request.prompt_text())
    ttft = first - start if first else None
    return {
        "ttft_s": ttft, "elapsed_s": elapsed,
        "tokens_per_second": max(0, usage["completion_tokens"] - 1) / max(elapsed - ttft, 1e-6) if ttft is not None else None,
        "peak_rss_mib": max(memory) / 1024**2 if memory else None,
        "rss_samples": len(memory), "usage": usage, "usage_source": source,
        "output": accumulator.content, "error": failure,
    }


async def run(args):
    config = yaml.safe_load(await asyncio.to_thread(Path(args.config).read_text, encoding="utf-8"))
    suite = prompts()
    if args.limit:
        suite = suite[:args.limit]
    suite_hash = fingerprint(suite)
    done = {row["run_id"] for row in read_jsonl(args.output) if not row.get("error")}
    async with httpx.AsyncClient(trust_env=False) as client:
        for item in config["configurations"]:
            tier = Tier.model_validate({**item["tier"], **({"provider": "mock"} if args.demo else {})})
            metadata = {"demo": args.demo, "model": tier.model, "quantization": item["quantization"], "data_version": DATA_VERSION, "suite_hash": suite_hash}
            if not args.demo:
                show = await client.post(tier.base_url.rstrip("/") + "/api/show", json={"model": tier.model}, timeout=30)
                show.raise_for_status()
                metadata["ollama_details"] = show.json().get("details", {})
                actual = metadata["ollama_details"].get("quantization_level", "")
                if actual.lower() != item["quantization"].lower():
                    raise ValueError(f"{tier.model}: expected {item['quantization']}, got {actual}")
            backend = make_backend(tier, client)
            # Warm-up is recorded separately, never mixed into warm prompt percentiles.
            warm = await generate(backend, ChatRequest(messages=[{"role": "user", "content": "Say hello."}], max_tokens=8), args.ollama_pid, args.timeout)
            append_jsonl(str(args.output) + ".warmup.jsonl", {**provenance(**metadata), "configuration": item["id"], **warm})
            if warm["error"]:
                raise RuntimeError(f"Warm-up failed for {item['id']}: {warm['error']}")
            ordered = list(suite)
            random.Random(args.seed).shuffle(ordered)
            for prompt in ordered:
                run_id = fingerprint([metadata, prompt["id"], args.seed, args.max_tokens])
                if run_id in done:
                    continue
                request = ChatRequest(messages=[{"role": "user", "content": prompt["prompt"]}], max_tokens=args.max_tokens, seed=args.seed)
                result = await generate(backend, request, args.ollama_pid, args.timeout)
                append_jsonl(args.output, {**provenance(**metadata), "run_id": run_id, "configuration": item["id"], "prompt_id": prompt["id"], "prompt_type": prompt["type"], **result})
                print(f"{item['id']} {prompt['id']}: {'error' if result['error'] else 'ok'}", flush=True)
            if not args.demo:
                # Sequential isolated model residency makes the RSS comparison interpretable.
                unload = await client.post(tier.base_url.rstrip("/") + "/api/generate", json={"model": tier.model, "keep_alive": 0}, timeout=30)
                unload.raise_for_status()


def report(args):
    rows = read_jsonl(args.input)
    # Retain latest attempt per logical run, with failed attempts visible in raw JSONL.
    unique = {row["run_id"]: row for row in rows}
    grades = {row["run_id"]: row for row in read_jsonl(args.grades)} if args.grades else {}
    summaries = []
    for configuration in sorted({row["configuration"] for row in unique.values()}):
        group = [row for row in unique.values() if row["configuration"] == configuration]
        good = [row for row in group if not row["error"]]
        scored = [grades[row["run_id"]] for row in group if row["run_id"] in grades and grades[row["run_id"]].get("score") is not None]
        scores = [grade["score"] for grade in scored]
        json_grades = [grade for grade in scored if "schema_valid" in grade]
        summaries.append({
            "configuration": configuration, "n": len(group), "errors": len(group) - len(good),
            "median_TTFT_s": percentile([r["ttft_s"] for r in good], 50),
            "p95_TTFT_s": percentile([r["ttft_s"] for r in good], 95),
            "median_tokens_s": percentile([r["tokens_per_second"] for r in good], 50),
            "peak_RSS_MiB": max((r["peak_rss_mib"] for r in good if r["peak_rss_mib"] is not None), default=None),
            "quality_1_to_5": sum(scores) / len(scores) if scores else None, "graded_n": len(scores),
            "json_schema_valid_rate": sum(g["schema_valid"] for g in json_grades) / len(json_grades) if json_grades else None,
            "json_exact_match_rate": sum(g["exact_match"] for g in json_grades) / len(json_grades) if json_grades else None,
        })
    title = "DEMO ONLY — mock profile" if any(row.get("demo") for row in rows) else "Local model profile"
    write_table(args.output, summaries, title)
    if args.timeout_config:
        from experiments.common import write_json
        write_json(args.timeout_config, {row["configuration"]: {"suggested_ttft_timeout": 2 * row["p95_TTFT_s"], "n": row["n"]} for row in summaries if row["p95_TTFT_s"] is not None})
