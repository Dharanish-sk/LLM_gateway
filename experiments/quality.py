import asyncio
import json
from pathlib import Path

import httpx
from jsonschema import ValidationError, validate

from experiments.common import append_jsonl, fingerprint, provenance, read_jsonl
from experiments.data import prompts
from experiments.profile import generate
from gateway.backends import make_backend
from gateway.config import load_settings
from gateway.models import ChatRequest


def load_suite(path=None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else prompts()


def json_score(output: str, prompt: dict):
    try:
        parsed = json.loads(output)
        validate(parsed, prompt["schema"])
    except (ValueError, ValidationError):
        return {"schema_valid": False, "exact_match": False, "score": 1}
    exact = parsed == prompt["reference"]
    return {"schema_valid": True, "exact_match": exact, "score": 5 if exact else 3}


async def collect(args):
    settings = load_settings(args.config)
    selected_tiers = args.tiers or [settings.router.small_tier, settings.router.large_tier]
    if not set(selected_tiers) <= {tier.name for tier in settings.tiers}:
        raise ValueError("unknown collection tier")
    suite = load_suite(args.suite)
    done = {row["run_id"] for row in read_jsonl(args.output) if not row.get("error")}
    async with httpx.AsyncClient(trust_env=False) as client:
        for tier in settings.tiers:
            if tier.name not in selected_tiers:
                continue
            backend = make_backend(tier, client)
            for prompt in suite:
                run_id = fingerprint([tier.model_dump(), prompt, args.max_tokens])
                if run_id in done:
                    continue
                result = await generate(backend, ChatRequest(messages=[{"role": "user", "content": prompt["prompt"]}], max_tokens=args.max_tokens), deadline_seconds=tier.total_timeout)
                append_jsonl(args.output, {**provenance(), "run_id": run_id, "configuration": tier.name, "model": tier.model, "prompt_id": prompt["id"], "demo": settings.mode == "demo", **result})


async def grade(args):
    settings = load_settings(args.config)
    judge_name = args.judge_tier or settings.router.large_tier
    judge = next((tier for tier in settings.tiers if tier.name == judge_name), None)
    if judge is None:
        raise ValueError("unknown judge tier")
    if judge.provider == "mock":
        raise ValueError("quality grading requires a real judge")
    rubric = await asyncio.to_thread(Path(args.rubric).read_text, encoding="utf-8")
    rubric_hash = fingerprint(rubric)
    suite = {row["id"]: row for row in load_suite(args.suite)}
    done = {(row["run_id"], row.get("rubric_hash"), row.get("judge")) for row in read_jsonl(args.output) if row.get("score") is not None}
    async with httpx.AsyncClient(trust_env=False) as client:
        backend = make_backend(judge, client)
        latest = {row["run_id"]: row for row in read_jsonl(args.input)}
        for row in latest.values():
            if row.get("demo"):
                raise ValueError("Mock generations are not valid quality measurements")
            if (row["run_id"], rubric_hash, judge.model) in done:
                continue
            prompt = suite[row["prompt_id"]]
            result = {"run_id": row["run_id"], "configuration": row["configuration"], "prompt_id": row["prompt_id"], "rubric_hash": rubric_hash, "judge": judge.model}
            if row.get("error"):
                result.update(score=1, reason="Generation failed")
                if prompt["type"] == "json":
                    result.update(schema_valid=False, exact_match=False)
            elif prompt["type"] == "json":
                result.update(json_score(row["output"], prompt))
            else:
                response_format = {"type": "json_object"}
                if judge.provider == "anthropic":
                    response_format = {
                        "type": "json_schema",
                        "json_schema": {"name": "answer_grade", "strict": True, "schema": {
                            "type": "object", "properties": {
                                "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                                "reason": {"type": "string"},
                            }, "required": ["score", "reason"], "additionalProperties": False,
                        }},
                    }
                request = ChatRequest(messages=[
                    {"role": "system", "content": rubric},
                    {"role": "user", "content": json.dumps({"task": prompt["prompt"], "reference": prompt["reference"], "candidate": row["output"]})},
                ], response_format=response_format, max_tokens=256)
                response = await generate(backend, request, deadline_seconds=judge.total_timeout)
                try:
                    if response["error"]:
                        raise ValueError("judge failed")
                    evaluation = json.loads(response["output"])
                    validate(evaluation, {"type": "object", "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5}, "reason": {"type": "string"}}, "required": ["score", "reason"], "additionalProperties": False})
                    result.update(evaluation)
                    result["judge_usage"] = response["usage"]
                except (ValueError, ValidationError):
                    result.update(score=None, error="invalid_judge_result")
            append_jsonl(args.output, {**provenance(), **result})


def pair(args):
    suite = {row["id"]: row for row in load_suite(args.suite)}
    grades = {row["run_id"]: row for row in read_jsonl(args.grades) if row.get("score") is not None}
    generations = {(row["prompt_id"], row["configuration"]): row for row in read_jsonl(args.input) if not row.get("error")}
    existing = {row["id"] for row in read_jsonl(args.output)}
    count = 0
    for prompt_id, prompt in suite.items():
        small, large = generations.get((prompt_id, args.small)), generations.get((prompt_id, args.large))
        if not small or not large or small["run_id"] not in grades or large["run_id"] not in grades:
            continue
        if small.get("demo") or large.get("demo"):
            raise ValueError("Mock outputs cannot train a production router")
        small_grade, large_grade = grades[small["run_id"]], grades[large["run_id"]]
        if (small_grade["judge"], small_grade["rubric_hash"]) != (large_grade["judge"], large_grade["rubric_hash"]):
            raise ValueError("Paired answers must use the same judge and rubric")
        identity = fingerprint([small["run_id"], large["run_id"], small_grade, large_grade])
        if identity in existing:
            continue
        append_jsonl(args.output, {
            "id": identity, "prompt_id": prompt_id, "family": prompt["family"], "prompt": prompt["prompt"],
            "small_score": small_grade["score"], "large_score": large_grade["score"],
            "small_usage": small["usage"], "large_usage": large["usage"],
            "small_model": small["model"], "large_model": large["model"],
            "rubric_hash": small_grade["rubric_hash"], "judge": small_grade["judge"],
        })
        count += 1
    print(f"Added {count} paired examples")
