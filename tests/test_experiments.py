import argparse
import json
from collections import Counter

import numpy as np
import pytest

from experiments.cache_sweep import sweep
from experiments.common import append_jsonl, wilson
from experiments.data import cache_pairs, prompts
from experiments.quality import json_score
from experiments.router_train import curve_point, fit_artifact, run
from gateway.config import Price
from gateway.router import predict


def test_suite_and_pair_design():
    assert Counter(row["type"] for row in prompts()) == {"factual": 10, "summarization": 10, "json": 10, "reasoning": 10}
    pairs = cache_pairs()
    assert len(pairs) == 200
    assert sum(row["paraphrase"] for row in pairs) == 100
    assert len({row["group"] for row in pairs}) == 100
    assert all(row["anchor"] != row["candidate"] for row in pairs)


def test_json_scoring_requires_schema_and_exact_match():
    row = next(row for row in prompts() if row["type"] == "json")
    assert json_score(json.dumps(row["reference"]), row) == {"schema_valid": True, "exact_match": True, "score": 5}
    assert json_score(json.dumps({**row["reference"], "age": 999}), row)["score"] == 3
    assert json_score(json.dumps({**row["reference"], "extra": 1}), row)["score"] == 1
    assert json_score("```json\n{}\n```", row)["schema_valid"] is False


def test_threshold_selects_false_hit_constraint_and_can_decline():
    rows = [{"paraphrase": True, "similarity": .97} for _ in range(100)] + [{"paraphrase": False, "similarity": .92} for _ in range(100)]
    table, threshold = sweep(rows)
    assert len(table) == 20 and table[0]["tau"] == .8 and table[-1]["tau"] == .99
    assert threshold == .97
    selected = next(row for row in table if row["tau"] == threshold)
    assert selected["paraphrase_hit_rate"] == 1 and selected["false_hit_rate"] == 0
    assert sweep([{**row, "similarity": 1} for row in rows])[1] is None
    assert wilson(0, 50)[1] > .02  # Zero observed misses does not certify a 2% population bound.


def test_cost_arithmetic_uses_unrounded_values():
    rows = [{"small_score": 5, "large_score": 5, "small_usage": {"prompt_tokens": 700, "completion_tokens": 200}, "large_usage": {"prompt_tokens": 700, "completion_tokens": 200}} for _ in range(100)]
    point = curve_point(rows, [True] * 65 + [False] * 35, Price(input_per_million=.15, output_per_million=.15), Price(input_per_million=3, output_per_million=3), "v1", .8)
    assert point["cost_per_1000_usd"] == pytest.approx(1.03275)
    assert point["always_large_usd"] == pytest.approx(2.7)
    assert point["saving_fraction"] == pytest.approx(.6175)
    # 61.9% in the original plan used the already rounded $1.03; exact arithmetic is 61.75%.


def test_safe_json_classifier_reproduces_direction():
    features = np.array([[0, 0], [.1, .1], [1, 1], [.9, .9]])
    artifact = fit_artifact(features, np.array([0, 0, 1, 1]), "test")
    assert predict(artifact, [0], 0) < .5
    assert predict(artifact, [1], 2) > .5


async def test_router_pipeline_group_holdout_and_curve(tmp_path):
    source = tmp_path / "paired.jsonl"
    for index in range(100):
        append_jsonl(source, {"id": str(index), "prompt_id": str(index), "family": f"family-{index // 2}", "prompt": ("simple fact" if index % 2 else "derive complex algorithm") + str(index), "small_score": 5 if index % 2 else 2, "large_score": 5, "small_usage": {"prompt_tokens": 50, "completion_tokens": 20}, "large_usage": {"prompt_tokens": 50, "completion_tokens": 30}})
    args = argparse.Namespace(config="config/demo.yaml", input=str(source), output=str(tmp_path / "curve"), artifact=str(tmp_path / "router.json"), seed=42, demo=True)
    await run(args)
    artifact = json.loads((tmp_path / "router.json").read_text())
    assert not set(artifact["train_families"]) & set(artifact["test_families"])
    assert artifact["train_n"] + artifact["test_n"] == 100
    assert (tmp_path / "curve.svg").exists()
    assert "v0" in (tmp_path / "curve.csv").read_text()
    assert "v1" in (tmp_path / "curve.csv").read_text()
    # The exported JSON is actually loadable by the runtime and uses the same feature contract.
    from gateway.app import create_app
    from gateway.config import load_settings
    from gateway.models import ChatRequest
    settings = load_settings("config/demo.yaml")
    settings.router.version = "v1"
    settings.router.artifact = str(tmp_path / "router.json")
    settings.router.probability_threshold = 0
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        decision = await app.state.router.choose(ChatRequest(messages=[{"role": "user", "content": "simple fact"}]))
        assert decision.tier == "local"
        assert decision.probability is not None
        assert 0 <= decision.probability <= 1


async def test_quality_pipeline_uses_configured_claude_judge_schema(tmp_path, monkeypatch):
    from experiments import quality
    from experiments.common import read_jsonl
    from gateway.backends import BackendEvent

    source, output = tmp_path / "candidates.jsonl", tmp_path / "grades.jsonl"
    append_jsonl(source, {"run_id": "test-run", "configuration": "local", "prompt_id": "factual-00", "output": "Canberra", "error": None, "demo": False})

    class Judge:
        async def complete(self, request):
            assert request.response_format["type"] == "json_schema"
            assert request.response_format["json_schema"]["schema"]["properties"]["score"]["enum"] == [1, 2, 3, 4, 5]
            yield BackendEvent({"content": '{"score":5,"reason":"Correct"}'}, "stop")

    monkeypatch.setattr(quality, "make_backend", lambda tier, client: Judge())
    args = argparse.Namespace(config="config/providers.yaml", input=str(source), output=str(output), rubric="data/rubric.md", suite=None, judge_tier=None)
    await quality.grade(args)
    result = read_jsonl(output)[0]
    assert result["score"] == 5 and result["judge"].startswith("claude-")
