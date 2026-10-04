import random

import httpx

from experiments.common import fingerprint, provenance, read_jsonl, wilson, write_json, write_table
from experiments.data import cache_pairs
from gateway.embeddings import DemoEmbedder, OllamaEmbedder, cosine


def sweep(rows: list[dict]) -> tuple[list[dict], float | None]:
    if not any(row["paraphrase"] for row in rows) or not any(not row["paraphrase"] for row in rows):
        raise ValueError("each evaluation split needs positive and negative examples")
    results = []
    for step in range(80, 100):
        threshold = step / 100
        positives = [r for r in rows if r["paraphrase"]]
        negatives = [r for r in rows if not r["paraphrase"]]
        true_hits = sum(r["similarity"] >= threshold for r in positives)
        false_hits = sum(r["similarity"] >= threshold for r in negatives)
        results.append({
            "tau": threshold, "positive_n": len(positives), "negative_n": len(negatives),
            "true_positives": true_hits, "false_positives": false_hits,
            "paraphrase_hit_rate": true_hits / len(positives), "false_hit_rate": false_hits / len(negatives),
        })
    eligible = [r for r in results if r["false_hit_rate"] <= 0.02]
    best = max(eligible, key=lambda r: (r["paraphrase_hit_rate"], r["tau"])) if eligible else None
    return results, best["tau"] if best else None


async def run(args):
    pairs = read_jsonl(args.pairs) if args.pairs else cache_pairs()
    if not pairs or not any(r["paraphrase"] for r in pairs) or not any(not r["paraphrase"] for r in pairs):
        raise ValueError("cache evaluation requires both positive and negative pairs")
    async with httpx.AsyncClient(trust_env=False) as client:
        embedder = DemoEmbedder() if args.demo else OllamaEmbedder(client, args.url, args.model)
        vectors = {}
        measured = []
        for row in pairs:
            for text in (row["anchor"], row["candidate"]):
                if text not in vectors:
                    vectors[text] = await embedder.embed(text)
            measured.append({**row, "similarity": cosine(vectors[row["anchor"]], vectors[row["candidate"]])})
    groups = sorted({row["group"] for row in measured})
    random.Random(args.seed).shuffle(groups)
    calibration_groups = set(groups[:len(groups) // 2])
    calibration = [row for row in measured if row["group"] in calibration_groups]
    heldout = [row for row in measured if row["group"] not in calibration_groups]
    calibration_table, threshold = sweep(calibration)
    full_table, _ = sweep(measured)
    heldout_table, _ = sweep(heldout)
    prefix = "DEMO HASH EMBEDDINGS — " if args.demo else ""
    write_table(args.output + "-all", full_table, prefix + "Full-pair diagnostic sweep (not used for selection)")
    write_table(args.output + "-calibration", calibration_table, prefix + "Calibration sweep")
    write_table(args.output + "-heldout", heldout_table, prefix + "Held-out sweep (selection fixed on calibration)")
    selected = next((row for row in heldout_table if row["tau"] == threshold), None)
    if selected:
        selected["false_hit_rate_95ci"] = wilson(selected["false_positives"], selected["negative_n"])
        selected["hit_rate_95ci"] = wilson(selected["true_positives"], selected["positive_n"])
    write_json(args.output + "-selection.json", {
        **provenance(demo=args.demo, embedding_model=embedder.model, pairs_hash=fingerprint(pairs)),
        "threshold": threshold, "heldout": selected,
        "recommend_semantic_enabled": not args.demo and selected is not None and selected["false_hit_rate"] <= 0.02,
        "note": "Threshold is embedding- and dataset-specific. Observed <=2% is not a population guarantee. Expand independently authored prompt families.",
        "measurements": measured,
    })
