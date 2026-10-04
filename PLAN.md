# Intelligent LLM Gateway — revised delivery plan

## Scope and acceptance rules

Build a Python 3.11+ FastAPI gateway in a project-local `.venv`. The production path uses Redis, Ollama and an OpenAI-compatible hosted provider. A separate, explicitly labelled demo configuration uses in-memory state and deterministic adapters so development does not require credentials or paid requests.

**Updated requirements:** develop and evaluate natively now; prepare container assets but defer Docker execution until the user introduces it. Support native Anthropic Claude and Google Gemini protocols as well as configurable OpenAI-compatible endpoints for Qwen, DeepSeek and other text models. API keys/providers can change between experiments; model changes require fresh measurements and routing artifacts. `config/providers.yaml` is the native multi-provider catalog.

Keep measurements honest: no estimated hardware results in benchmark tables. Store raw, resumable JSONL measurements, configuration, sample sizes and provenance; render tables only from those measurements. Model downloads, paid grading and overnight runs are explicit operator commands.

## Improvements to the original plan

1. **Protocol and security first:** define supported OpenAI request fields, SSE termination, disconnect behavior, key hashing, tenant isolation and cache eligibility. Do not log prompt text or credentials. Fail closed if Redis is unavailable.
2. **Concurrency correctness:** implement an atomic Redis Lua token bucket using Redis server time. Test the exact 20/21/5-second arithmetic with an injected clock, and independently test concurrent Redis requests. Use monotonic clocks for deadlines/circuit breakers.
3. **Meaningful caching:** exact matching first; semantic matching only for single-user, deterministic, tool-free requests. Namespace by credential, conversation/system context, all model parameters and deployment policy revision. Cache only successful complete answers. Bound TTL and scan size. Do not enable semantic hits until an embedding-specific threshold has been calibrated.
4. **Evaluation without leakage:** separate cache threshold calibration from held-out evaluation, and group router train/test splits by prompt family. The 40-prompt suite is a profiling seed, not enough data to claim a generally reliable classifier. Collect paired small/large answers with the same rubric; report classifier limitations and confidence intervals.
5. **Comparable costs:** use separate input/output prices per tier, source URL and verification date. Keep example prices explicitly unverified until the operator checks their provider. Include local infrastructure costs separately; a local model is not inherently free. Plot a threshold sweep rather than one favorable point.
6. **Measured reliability:** separate TTFT, inter-token idle and total deadlines. Retry at most twice across distinct tiers on timeout, 429, connection errors or 5xx. Do not retry other 4xx. Permit fallback only before the first backend output; after streaming begins emit an error and terminate. Circuit breakers have a single half-open probe.
7. **Reproducible experiments:** warm models, record cold-start separately, distinguish generation throughput from end-to-end throughput, record Ollama process-tree peak RSS as sampled RSS (not system RAM or GPU VRAM), and retain failures in reports. Randomize/order runs reproducibly. Quantization tags must resolve to the intended artifact.
8. **Observable by default:** low-cardinality Prometheus metrics and OTLP spans compatible with Langfuse. Include cache/router/backend decisions without raw message content. Run streaming-aware load tests with hundreds of samples per level; never infer production capacity from the mock tier.
9. **Capability-aware selection (implemented):** preflight adapter/model features and deployment budgets before cache/embedding calls; filter both routing and retries. Support strict model pinning and fallback allowlists, include policy in cache identity, and reject incompatible cached actual tiers. Context limits use an explicitly approximate byte-based admission estimate; verify declarations with live provider versions when credentials arrive.

## Week 1 — executable foundation and profiling

- FastAPI `/v1/chat/completions`, `/health/live`, `/health/ready`, `/metrics`, `/docs`.
- Shared async backend interface; Ollama, OpenAI-compatible, native Anthropic and native Gemini adapters, plus demo/failure-injection adapters.
- Hashed file-based API keys, atomic Redis rate limiting, isolated virtual environment.
- Versioned 40-prompt suite: ten factual, summarization, JSON extraction and reasoning tasks each. JSON schema + exact-match checks; checked-in 1–5 rubric for hosted grading.
- Six configurable model/quantization combinations, 240 resumable runs, warm-up, TTFT/throughput/RSS/quality reports. Select a local tier **only after real measurements**.

## Week 2 — calibrated cache and routers

- Exact cache plus bounded Redis semantic index with local Ollama embeddings; replace bounded scan with Redis Search/HNSW at larger scale.
- 100 paraphrase and 100 meaning-changing pairs, with deterministic grouped calibration/held-out split. Sweep 0.80–0.99, maximize recall subject to calibration false hits ≤2%, report held-out false hits and binomial intervals. Small samples cannot certify a universal 2% bound.
- v0 length/task rules; v1 standardized logistic regression on embeddings and prompt length, serialized as safe JSON coefficients (no pickle). Paired rubric labels use `small_score >= large_score - 0.5`.
- Held-out v0 and v1 cost/agreement sweeps, sample sizes and always-large baseline. Refit deployment model only after reporting held-out performance.

## Week 3 — fault tolerance

- Tier-specific timeouts, retry budget, single-probe circuit breaker, cancellation cleanup and explicit post-first-output SSE error policy.
- Deterministic slow/500/429/drop injection with the same seeded request schedule for fallback on/off; report error rate and p95 with sample counts.
- Derive candidate TTFT deadlines from measured p95 using a script; configuration changes remain explicit.

## Week 4 — telemetry and load

- Prometheus request/outcome/tier, TTFT/latency, tokens, costs, cache, fallback and rejection metrics.
- Checked-in Grafana dashboard and provisioning; optional Langfuse OTLP endpoint.
- Streaming-aware load generator at 5/10/20/40 concurrency, at least 400 requests per level by default, p50/p95/error rates and sample counts.
- README links to generated result tables; capture a real Grafana screenshot once Docker is available.

## Week 5 — operational handoff

- Prepare Docker Compose for gateway, Redis, Ollama, Prometheus and Grafana, including provider-secret environment wiring. Build/start verification is deferred by user request.
- Configuration-driven tiers, prices, routing and cache policy; environment-based secrets.
- Architecture decisions and rejected alternatives, Windows/Linux setup, test commands, benchmark runbook and limitations.
- Optional vLLM benchmark is deferred until hardware budget and deployment are provided.

## Definition of done

Implementation is done when protocol, arithmetic, isolation, routing and injected-failure tests pass in `.venv`, and demo evaluation/load scripts run end to end. Empirical acceptance gates remain visibly pending until real models, verified pricing and infrastructure are available. A generated demo table is never presented as a local-model benchmark.
