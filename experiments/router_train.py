import html
from pathlib import Path

import httpx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from experiments.common import fingerprint, provenance, read_jsonl, wilson, write_json, write_table
from gateway.config import RouterSettings, load_settings
from gateway.embeddings import DemoEmbedder, OllamaEmbedder
from gateway.router import baseline, predict


def fit_artifact(features, labels, embedding_model):
    scaler = StandardScaler().fit(features)
    classifier = LogisticRegression(max_iter=2000, random_state=42).fit(scaler.transform(features), labels)
    return {"schema_version": 1, "embedding_model": embedding_model, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "coef": classifier.coef_[0].tolist(), "intercept": float(classifier.intercept_[0])}


def request_cost(usage, price):
    return (usage["prompt_tokens"] * price.input_per_million + usage["completion_tokens"] * price.output_per_million) / 1e6


def curve_point(rows, small_choices, small_price, large_price, version, threshold):
    costs, agreements = [], []
    for row, use_small in zip(rows, small_choices, strict=True):
        score = row["small_score"] if use_small else row["large_score"]
        agreements.append(score >= row["large_score"] - .5)
        costs.append(request_cost(row["small_usage"], small_price) if use_small else request_cost(row["large_usage"], large_price))
    always_large = sum(request_cost(r["large_usage"], large_price) for r in rows) / len(rows) * 1000
    cost = sum(costs) / len(costs) * 1000
    interval = wilson(sum(agreements), len(rows))
    return {"router": version, "threshold": threshold, "n": len(rows), "small_fraction": sum(small_choices) / len(rows), "agreement": sum(agreements) / len(rows), "agreement_low95": interval[0], "agreement_high95": interval[1], "cost_per_1000_usd": cost, "always_large_usd": always_large, "saving_fraction": (always_large - cost) / always_large if always_large else 0}


def plot_curve(path, rows, title):
    width, height, pad = 800, 450, 65
    max_cost = max(row["cost_per_1000_usd"] for row in rows) or 1
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/><text x="65" y="25" font-family="sans-serif" font-size="15">{html.escape(title)}</text>', f'<path d="M{pad},50 V{height-pad} H{width-pad}" fill="none" stroke="#555"/>']
    for version, color in (("v0", "#d97706"), ("v1", "#2563eb")):
        points = []
        for row in sorted((r for r in rows if r["router"] == version), key=lambda r: r["cost_per_1000_usd"]):
            x = pad + row["cost_per_1000_usd"] / max_cost * (width - 2 * pad)
            y = height - pad - row["agreement"] * (height - pad - 50)
            points.append(f"{x:.1f},{y:.1f}")
        body.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
    body += [f'<g font-family="sans-serif" font-size="13"><text x="{pad}" y="{height-30}">Cost / 1,000 requests: $0 to ${max_cost:.3f}</text><text x="8" y="50">1.0</text><text x="8" y="{height-pad}">0.0</text><text x="80" y="45">Agreement</text><text x="570" y="45" fill="#d97706">v0</text><text x="620" y="45" fill="#2563eb">v1</text></g></svg>']
    Path(path).write_text("\n".join(body), encoding="utf-8")


async def run(args):
    settings = load_settings(args.config)
    data = read_jsonl(args.input)
    if len(data) < 20:
        raise ValueError("Need at least 20 paired examples; collect hundreds for meaningful results")
    if len({row["prompt_id"] for row in data}) != len(data):
        raise ValueError("Duplicate prompt IDs: curate one paired measurement per prompt before training")
    tiers = {tier.name: tier for tier in settings.tiers}
    small, large = tiers[settings.router.small_tier], tiers[settings.router.large_tier]
    if not args.demo and any(row["small_model"] != small.model or row["large_model"] != large.model for row in data):
        raise ValueError("Paired model IDs must match configured routing tiers")
    async with httpx.AsyncClient(trust_env=False) as client:
        embedder = DemoEmbedder() if args.demo else OllamaEmbedder(client, settings.cache.embedding_url, settings.cache.embedding_model)
        vectors = [await embedder.embed(row["prompt"]) for row in data]
    features = np.asarray([[*vector, np.log1p(len(row["prompt"]))] for row, vector in zip(data, vectors, strict=True)])
    labels = np.asarray([row["small_score"] >= row["large_score"] - .5 for row in data], dtype=int)
    splitter = GroupShuffleSplit(n_splits=1, test_size=.3, random_state=args.seed)
    train, test = next(splitter.split(features, labels, [row["family"] for row in data]))
    if len(set(labels[train])) < 2:
        raise ValueError("Training split needs both good-enough and not-good-enough labels")
    heldout_artifact = fit_artifact(features[train], labels[train], embedder.model)
    heldout = [data[i] for i in test]
    probabilities = [predict(heldout_artifact, vectors[i], len(data[i]["prompt"])) for i in test]
    curve = []
    for step in range(0, 21):
        p = step / 20
        curve.append(curve_point(heldout, [prob >= p for prob in probabilities], small.price, large.price, "v1", p))
    # Controlled two-tier v0 ablation: medium maps to large so both curves use paired answers.
    for chars in (1, 80, 160, 300, 600, 1200, 2400, 4800, 9600):
        baseline_config = RouterSettings(small_tier=small.name, medium_tier=large.name, large_tier=large.name, short_chars=chars, long_chars=chars)
        choices = [baseline(row["prompt"], baseline_config).tier == small.name for row in heldout]
        curve.append(curve_point(heldout, choices, small.price, large.price, "v0", chars))
    title = ("DEMO FEATURES — " if args.demo else "") + "Held-out cost vs rubric agreement (v0 two-tier ablation)"
    write_table(args.output, curve, title)
    plot_curve(args.output + ".svg", curve, title)
    # Refit on all labels only AFTER producing held-out predictions.
    deployment = fit_artifact(features, labels, embedder.model)
    deployment.update(provenance(demo=args.demo, dataset_hash=fingerprint(data)))
    deployment.update(small_model=small.model, large_model=large.model, train_n=len(train), test_n=len(test), evaluation_artifact=heldout_artifact, train_families=sorted({data[i]["family"] for i in train}), test_families=sorted({data[i]["family"] for i in test}), prices_verified=bool(small.price.verified_on and large.price.verified_on), note="Threshold must be selected for your quality/cost objective. The curve is not an unbiased final evaluation if its threshold is tuned on these held-out results; use an additional final set.")
    write_json(args.artifact, deployment)
