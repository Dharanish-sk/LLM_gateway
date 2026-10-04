# Evaluation runbook

Run from the repository root in `.venv`, with the `eval` extras installed. All generated results live under `results/`; expensive generation/grading commands use resumable JSONL. Use a separate output file for each hardware/configuration experiment. Benchmark files contain prompt outputs intentionally; telemetry does not.

## 1. Export and audit data

```bash
gateway-eval datasets --output-dir results/datasets
```

Source of truth: `experiments/data.py`, version `2026-10-v1`.

- 40 prompts: ten each factual, summarization, JSON extraction and reasoning.
- 200 cache pairs: 100 paraphrases, 100 near-misses involving geography, units, arithmetic, percentages, negation, sort order, timezone sign, JSON field selection, comparisons and temperature scales.

The seed data is intentionally compact and templated. It is adequate for profiling and regression tests, not a substitute for domain-specific evaluation. Author additional independent prompt families before choosing production thresholds. Custom router suites are JSON arrays following the exported prompt schema; custom cache pairs use JSONL with the exported pair fields.

## 2. Profile six local configurations

Install Ollama and inspect `config/profile.yaml`. It specifies Llama 3.2 3B, Qwen 2.5 1.5B and Gemma 2 2B, each at Q4_K_M and Q5_K_M. Pull each **exact** tag, for example:

```bash
ollama pull llama3.2:3b-instruct-q4_K_M
ollama pull llama3.2:3b-instruct-q5_K_M
ollama pull qwen2.5:1.5b-instruct-q4_K_M
ollama pull qwen2.5:1.5b-instruct-q5_K_M
ollama pull gemma2:2b-instruct-q4_K_M
ollama pull gemma2:2b-instruct-q5_K_M
ollama pull all-minilm
```

If a tag is unavailable in the registry, create an Ollama model from the corresponding GGUF artifact and update its tag in config. The runner calls `/api/show` and refuses an unexpected quantization. Set `OLLAMA_MAX_LOADED_MODELS=1` on the Ollama server and close unrelated inference clients for a clean CPU profile. Record whether GPU offload is active; CPU and GPU runs should be separate experiments.

Find the local Ollama server PID using your OS process manager, then:

```bash
gateway-eval profile --ollama-pid YOUR_PID --output results/profile.jsonl
```

The default run performs 240 measured generations, with a separate warm-up for each config. Prompt order is deterministically shuffled. Models unload between configurations. Successes are skipped when resuming; failed rows remain in raw data and are retried. Do not edit/merge JSONL midway through a run.

Measurements:

- TTFT: request start to first output event, excluding the separate warm-up.
- Generation tokens/s: `(completion_tokens - 1) / (elapsed - TTFT)`; this is a stream-level approximation because events need not be individual tokens. Provider usage is preferred to estimates.
- End-to-end elapsed time is also retained.
- Peak RSS: 50 ms samples of the specified Ollama process tree, summed RSS; shared pages can be double-counted. This is neither GPU VRAM nor total physical memory, and short spikes may be missed. An absent PID yields **missing**, never made-up RAM. A Docker PID is not the host PID on Windows; profile native Ollama for host RSS or supply a separately instrumented container measurement.
- Warm-up timing goes in `<output>.warmup.jsonl`. It may include a cold load, but is not guaranteed cold unless you started from an unloaded model. Record hardware/offload/other-process context alongside the results.

Budget roughly four hours on a slow CPU for 200-token answers; actual time depends on hardware and early stopping. This command does not call hosted APIs.

## 3. Quality grading

Configure a reachable large-tier judge and set its API key. `grade` makes **paid hosted calls** for non-JSON prompts:

The judge may be Claude, Gemini, Qwen, DeepSeek or another supported chat model. `collect` defaults to the selected config's router small/large aliases; `grade` defaults to its large alias. For the multi-provider catalog use `--config config/providers.yaml` in those commands, and `pair --small local --large claude` before `train-router --config config/providers.yaml`. Claude grading automatically uses a JSON schema rather than schema-less JSON mode. Keep the same judge/rubric for both answers in a pair.

```bash
gateway-eval grade --config config/gateway.yaml --input results/profile.jsonl --output results/profile-grades.jsonl
gateway-eval profile-report --input results/profile.jsonl --grades results/profile-grades.jsonl --output results/profile-summary --timeout-config results/timeouts.json
```

If running outside Compose, use a config with host-reachable Ollama URLs. JSON prompts are graded locally by schema validity and exact parsed-object equality. Other tasks use [the checked-in rubric](../data/rubric.md), an isolated grading system message, a reference answer and a blinded candidate. Responses must match the judge schema. Judge errors are retained and retried, not converted into confident scores. Mock-generated answers are rejected by the grading command.

The report includes TTFT median/p95, throughput, sampled RSS, failure count, graded count, average score, JSON validity and exact-match rates. Failed generations graded as 1 count toward quality. Audit a sample of judge decisions manually.

Choose the local configuration using an explicit RAM constraint and task-quality floor, then compare measured latency among qualifying configs. Update the local tier model and document the tradeoff in README. Do not declare a winner before filling the six-row real-model table.

`results/timeouts.json` proposes 2×p95 TTFT deadlines with N. It is a starting point, not a guarantee; measure hosted TTFT separately using `collect` and `profile-report`, and then run fault/load tests.

## 4. Calibrate semantic cache

```bash
gateway-eval cache-sweep --url http://localhost:11434 --model all-minilm --output results/cache
```

Each anchor's positive and negative pairs stay together in either calibration or held-out data (50 anchors per split). Cosines are swept from 0.80 through 0.99 in steps of 0.01:

- `cache-all.csv/.md`: diagnostic on all 100 positives and 100 negatives.
- `cache-calibration.csv/.md`: choose maximum paraphrase recall with false hits ≤2%; ties favor the higher threshold.
- `cache-heldout.csv/.md`: evaluate the already chosen threshold.
- `cache-selection.json`: selection, held-out Wilson 95% intervals, individual similarities and model/data provenance.

If no threshold satisfies the constraint, semantic caching stays off. Otherwise review the held-out row and set `cache.threshold` and `cache.semantic_enabled` explicitly. A 50-negative held-out set cannot certify a population false-hit probability ≤2%, even with no errors. Expand data substantially before making a strong safety/accuracy claim. The demo hash embedder validates plumbing only and never produces a production enable recommendation.

## 5. Train and compare routers

Use the selected local model and a large reference with the same task suite and token budget:

```bash
gateway-eval collect --tiers local large --suite results/datasets/prompts.json
gateway-eval grade --suite results/datasets/prompts.json
gateway-eval pair --small local --large large --suite results/datasets/prompts.json
gateway-eval train-router --input results/paired.jsonl --output results/router-curve --artifact artifacts/router.json
```

For useful training, replace the 40-row seed with hundreds of independently authored prompts and stable family labels. Keep reformulations of one task in the same family. At least two label classes must occur in the training partition. The pairer checks judge/rubric identity; the trainer checks actual model identity against config.

The label is `small_score >= large_score - 0.5`. With integer single-judge scores, this reduces to tie-or-better. Repeated judging/mean scores are an extension for finer resolution. Agreement with a poor large answer is still a positive label, so audit absolute quality too.

The trainer uses group-separated 70/30 train/held-out data, standardizes embeddings plus log length, fits logistic regression, and emits:

- CSV/Markdown/SVG curves: v1 probability sweep 0–1 in 0.05 steps, and v0 character-threshold sweep.
- Both cost per 1,000 requests and rubric agreement, small-tier fraction, sample size and Wilson intervals.
- An always-large baseline using actual per-answer token counts.
- Safe JSON deployment coefficients, refit on all paired data **after** held-out predictions, plus the evaluation coefficients/split identities.

For an apples-to-apples paired-answer curve, v0's medium tier maps to large. Three-tier comparisons require a third answer and its grades. A threshold chosen on the curve must be checked on an additional final test set to avoid selection bias; refitted probabilities can also shift, so validate the final deployment model.

Enable v1 in a config copy:

```yaml
router:
  version: v1
  artifact: artifacts/router.json
  small_tier: local
  medium_tier: small
  large_tier: large
  probability_threshold: 0.8 # Replace with your chosen, independently validated operating point.
```

Restart the gateway. The embedding model and small/large model identities must match the artifact. To route hosted small instead of local, set `small_tier: small`, collect `small large`, and retrain.

### Cost arithmetic

Prices in `config/gateway.yaml` are standard uncached text-token rates, verified from the linked provider source. Preserve the verification date when updating them. Each curve point uses:

`sum(input_tokens * input_price + output_tokens * output_price) / 1e6 / N * 1000`

At 700 input and 200 output tokens, hosted small costs $0.225 per 1,000 requests; large costs $3.75. Routing 65% small gives `0.65*0.225 + 0.35*3.75 = $1.45875`, saving 61.1%. This is a worked example, not a measured routing operating point.

The original uniform-price example gives $1.03275, with exact saving 61.75%; reporting 61.9% from a rounded $1.03 overstates precision. Always round the final display only.

## 6. Faults and load

```bash
gateway-eval failures --fault 500 --requests 400 --users 10 --output results/fault-500
gateway-eval failures --fault slow --requests 400 --users 10 --output results/fault-slow
gateway-eval failures --fault partial --requests 400 --users 10 --output results/fault-partial
gateway-eval failures --fault mixed --requests 400 --users 10 --output results/fault-mixed
gateway-eval load --demo --users 5 10 20 40 --requests 400 --output results/demo-load
```

The fault harness injects errors at the adapter boundary, served by a real loopback Uvicorn HTTP server. Identical seeded request IDs carry the same scheduled faults in both fallback modes. It does not impersonate measured hosted-provider failures. Breakers stay enabled; use `--breaker-threshold 10000` for a controlled no-opening ablation. Error rate includes HTTP errors, SSE errors and missing stream termination.

For a real deployment, export `GATEWAY_API_KEY` and run:

```bash
gateway-eval load --url http://localhost:8000 --users 5 10 20 40 --requests 400 --output results/real-load
```

With the default 20/2 per-key rate limit, a high-rate one-key test will mostly measure quota rejection. For a throughput experiment, use a dedicated config with sufficient test-key capacity/refill; the production defaults remain 20/2. The harness makes unique requests to bypass cache and reports both all-request latency and successful-request p95. N is shown beside every percentile. `requests` means **per level**. Requests within each level are concurrent; levels run sequentially. This is a closed-loop concurrency sweep, not an open-loop arrival-rate/soak benchmark.

## 7. Dashboard evidence and completion

Start Compose, make sustained requests, and open Grafana's **LLM Gateway / Intelligent LLM Gateway** dashboard. Check the five-minute sample-count panel before interpreting histograms. Capture a real screenshot and update README with a link. Set Langfuse environment variables to verify trace ingestion on your instance.

Finish the empirical gates only after replacing pending entries with actual real-model tables, cache selection, router curves, hosted failure/load measurements and dashboard evidence. Optional vLLM requires a separately budgeted GPU and can be benchmarked through the hosted-compatible adapter.
