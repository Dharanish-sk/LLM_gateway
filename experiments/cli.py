import argparse
import asyncio


def main():
    parser = argparse.ArgumentParser(description="Reproducible LLM gateway experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("datasets", help="Export the versioned 40 prompts and 200 pairs")
    data.add_argument("--output-dir", default="results/datasets")
    profile = sub.add_parser("profile")
    profile.add_argument("--config", default="config/profile.yaml")
    profile.add_argument("--output", default="results/profile.jsonl")
    profile.add_argument("--seed", type=int, default=42)
    profile.add_argument("--max-tokens", type=int, default=200)
    profile.add_argument("--timeout", type=float, default=300)
    profile.add_argument("--ollama-pid", type=int)
    profile.add_argument("--limit", type=int)
    profile.add_argument("--demo", action="store_true")
    report = sub.add_parser("profile-report")
    report.add_argument("--input", default="results/profile.jsonl")
    report.add_argument("--grades")
    report.add_argument("--output", default="results/profile-summary")
    report.add_argument("--timeout-config")
    collect = sub.add_parser("collect", help="Generate paired-tier candidates (real providers may charge)")
    collect.add_argument("--config", default="config/gateway.yaml")
    collect.add_argument("--tiers", nargs="+", help="Defaults to the configured router small/large tiers")
    collect.add_argument("--suite")
    collect.add_argument("--output", default="results/candidates.jsonl")
    collect.add_argument("--max-tokens", type=int, default=200)
    grade = sub.add_parser("grade", help="Machine-score JSON; use configured judge for others (may charge)")
    grade.add_argument("--config", default="config/gateway.yaml")
    grade.add_argument("--input", default="results/candidates.jsonl")
    grade.add_argument("--output", default="results/grades.jsonl")
    grade.add_argument("--judge-tier", help="Defaults to the configured router large tier")
    grade.add_argument("--rubric", default="data/rubric.md")
    grade.add_argument("--suite")
    pair = sub.add_parser("pair")
    pair.add_argument("--input", default="results/candidates.jsonl")
    pair.add_argument("--grades", default="results/grades.jsonl")
    pair.add_argument("--small", default="local")
    pair.add_argument("--large", default="large")
    pair.add_argument("--suite")
    pair.add_argument("--output", default="results/paired.jsonl")
    cache = sub.add_parser("cache-sweep")
    cache.add_argument("--url", default="http://localhost:11434")
    cache.add_argument("--model", default="all-minilm")
    cache.add_argument("--pairs")
    cache.add_argument("--seed", type=int, default=42)
    cache.add_argument("--output", default="results/cache")
    cache.add_argument("--demo", action="store_true")
    router = sub.add_parser("train-router")
    router.add_argument("--config", default="config/gateway.yaml")
    router.add_argument("--input", default="results/paired.jsonl")
    router.add_argument("--output", default="results/router-curve")
    router.add_argument("--artifact", default="artifacts/router.json")
    router.add_argument("--seed", type=int, default=42)
    router.add_argument("--demo", action="store_true", help="Use hash features for pipeline tests only")
    load = sub.add_parser("load")
    load.add_argument("--url", default="http://localhost:8000")
    load.add_argument("--key-env", default="GATEWAY_API_KEY")
    load.add_argument("--users", nargs="+", type=int, default=[5, 10, 20, 40])
    load.add_argument("--requests", type=int, default=400, help="Requests PER concurrency level")
    load.add_argument("--model", default="auto")
    load.add_argument("--output", default="results/load")
    load.add_argument("--demo", action="store_true")
    faults = sub.add_parser("failures")
    faults.add_argument("--requests", type=int, default=400)
    faults.add_argument("--users", type=int, default=10)
    faults.add_argument("--seed", type=int, default=42)
    faults.add_argument("--failure-rate", type=float, default=.2)
    faults.add_argument("--fault", choices=["500", "429", "slow", "drop", "partial", "mixed"], default="mixed")
    faults.add_argument("--breaker-threshold", type=int, default=3)
    faults.add_argument("--output", default="results/failures")
    args = parser.parse_args()
    if hasattr(args, "requests") and args.requests < 1:
        parser.error("requests must be positive")
    if hasattr(args, "users") and any(n < 1 for n in (args.users if isinstance(args.users, list) else [args.users])):
        parser.error("users must be positive")
    if args.command == "datasets":
        from experiments.common import write_json
        from experiments.data import cache_pairs, prompts
        write_json(args.output_dir + "/prompts.json", prompts())
        # A JSON array is the canonical export; the pair CLI accepts JSONL for custom pairs.
        write_json(args.output_dir + "/cache-pairs.json", cache_pairs())
        print("Exported 40 prompts and 200 prompt pairs")
    elif args.command in {"profile", "profile-report"}:
        from experiments import profile as module
        if args.command == "profile":
            asyncio.run(module.run(args))
        else:
            module.report(args)
    elif args.command in {"collect", "grade", "pair"}:
        from experiments import quality
        function = getattr(quality, args.command)
        if args.command == "pair":
            function(args)
        else:
            asyncio.run(function(args))
    else:
        from importlib import import_module
        name = {"cache-sweep": "cache_sweep", "train-router": "router_train"}.get(args.command, args.command)
        asyncio.run(import_module("experiments." + name).run(args))


if __name__ == "__main__":
    main()
