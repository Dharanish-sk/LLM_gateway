# Intelligent LLM Gateway: how it works and what to improve next

This guide explains the implemented gateway in plain language, from an incoming request to the returned answer. It also records proposed improvements, their purpose and how to verify them.

**Current position:** the application and provider adapters are implemented and locally tested. Demo mode runs without paid API access. Live provider validation, real-model benchmarks, semantic-cache calibration and a production-trained router remain pending. Development uses the project-local `.venv`; Docker execution is deferred, with container files prepared for later.

## Contents

- [Overview](#overview)
- [Models, providers and tiers](#models-providers-and-tiers)
- [The request lifecycle](#the-request-lifecycle)
- [How caching works](#how-caching-works)
- [How routing works](#how-routing-works)
- [How provider adapters work](#how-provider-adapters-work)
- [Streaming and ordinary responses](#streaming-and-ordinary-responses)
- [Reliability and failure handling](#reliability-and-failure-handling)
- [Metrics, traces and cost](#metrics-traces-and-cost)
- [Offline evaluation and learning](#offline-evaluation-and-learning)
- [Running the project now](#running-the-project-now)
- [What has actually been verified](#what-has-actually-been-verified)
- [Recommended improvements](#recommended-improvements)
- [Suggested implementation order](#suggested-implementation-order)
- [Where each part lives](#where-each-part-lives)

## Overview

The gateway is one API between your application and several language models. Your application makes a consistent request; the gateway manages authentication, quotas, caching, model selection, provider communication and failures.

```text
Your application
       |
       v
Validate request
       |
       v
Authenticate gateway client key
       |
       v
Redis request quota + active-request limit
       |
       v
Capability + routing-policy preflight
       |
       v
Cache lookup ---------------- hit ------------------+
       |                                           |
      miss                                         |
       |                                           |
       v                                           |
Choose preferred model                             |
       |                                           |
       v                                           |
Backend adapter + deadlines + circuit breaker       |
       |                                           |
       +--- retryable failure before output         |
       |        -> another configured backend       |
       |                                           |
       v                                           |
Stream/collect response                            |
       |                                           |
       v                                           |
Cache complete successful response                  |
       |                                           |
       +-------------------------------------------+
       |
       v
Return answer; finish metrics and trace
```

The gateway does not train or host a large model by itself. Hosted providers run their models on their infrastructure. For local inference, the gateway sends requests to a separate Ollama service, which runs an installed model on your machine.

## Models, providers and tiers

These terms refer to different things:

| Term | Meaning | Example |
| --- | --- | --- |
| Model | The actual AI model being called | `deepseek-chat` |
| Provider/adapter | The service and protocol used to access it | DeepSeek's OpenAI-compatible endpoint |
| Tier/alias | The name exposed by your gateway | `deepseek`, `small`, or `large` |

The configuration connects them. For example:

```yaml
- name: deepseek
  provider: openai_compatible
  model: deepseek-chat
  base_url: https://api.deepseek.com/v1
  api_key_env: DEEPSEEK_API_KEY
```

A client sends `model: "deepseek"`. The gateway looks up this entry, reads the provider key from the named environment variable, and calls the actual model.

You can rotate an API key without changing model behavior. You can also change the model/provider behind an alias. When the model changes, update prices, remeasure latency/quality, and retrain any router that depends on that model. Keep versions and the grading judge fixed within a comparison experiment so the results remain comparable.

**OpenAI-compatible describes the client-facing request/response format.** It does not require GPT as the backend. The catalog includes Claude, Gemini, Qwen, DeepSeek, GPT and a local tier.

## The request lifecycle

Consider this incoming request:

```http
POST /v1/chat/completions
Authorization: Bearer YOUR_GATEWAY_KEY
Content-Type: application/json
```

```json
{
  "model": "auto",
  "messages": [
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "stream": true
}
```

### 1. Validate the request

The API checks the message structure and supported parameters. The default maximum request body is 1 MiB; the current API accepts one completion per request (`n=1`). Text messages and supported tool-call data are accepted. Images/audio and unknown request fields are outside the current contract.

Invalid requests receive a structured error. Unknown model aliases receive 404. The default generation budget is 256 tokens when the client supplies no token limit.

### 2. Authenticate the client

There are two separate kinds of credentials:

```text
Application -- gateway client key --> Gateway
Gateway     -- provider API key   --> Hosted model
```

- A **gateway key** identifies a client allowed to use your service.
- A **provider key** lets the gateway call Claude, Gemini, DeepSeek or another hosted provider.

Gateway client key hashes are stored in `config/keys.json`. The server hashes an incoming bearer key and compares it with those records. Provider credentials are read from environment variables and sent using the relevant provider's authentication headers. They are not returned to your application.

Each gateway client identity gets its own request bucket and cache namespace. This prevents one client's cached answer from becoming another client's cache hit.

### 3. Enforce the Redis token bucket

A **bucket token means permission to make one request**. It is different from an LLM's text tokens.

Default settings:

```text
Capacity C = 20 requests
Refill r   = 2 requests per second
```

Before deciding whether to allow a request:

```text
available = min(C, previous_available + elapsed_seconds * r)
```

If `available >= 1`, subtract one and allow the request. Otherwise return HTTP 429 with `Retry-After`.

For an instantaneous burst with a full bucket:

| Event | Tokens available |
| --- | ---: |
| Start | 20 |
| After 20 accepted requests | 0 |
| Immediate request 21 | Rejected; still 0 |
| After 5 idle seconds | 10 |
| After the next accepted request | 9 |

Redis runs the refill/consume calculation atomically using Lua and Redis server time. Two simultaneous requests cannot independently spend the same final token. The fake-clock tests verify the exact arithmetic; real requests naturally allow refill as time passes.

The quota check runs before cache lookup, so cached responses still consume request quota.

### 4. Limit active work

The default active-request limit is 100 per gateway process. A request above the limit receives 503 instead of entering an unlimited backend queue.

These controls serve different purposes:

- The token bucket limits request arrival rate for each client key.
- The active-request limit bounds work being processed at the same time.

Redis stores shared quota/cache state. Active-request counts and circuit breakers are currently process-local; the deployment pattern is one gateway process per container.

### 5. Preflight capabilities and routing policy

Before reading cache or requesting embeddings, the gateway checks the request against each permitted deployment's declared features and the adapter's implementation limits. It checks tool/strict-tool support, JSON modes, sampling controls, message constraints, requested output limits and an estimated input/context budget.

If no permitted tier can serve the request, it returns HTTP 400 `no_compatible_tier` with reason codes before a generation call. `GET /v1/models` exposes effective capabilities. Declarations still need verification against live model versions; this preflight does not check credentials or provider availability.

The context estimate uses UTF-8 bytes of serialized messages/tools/response format plus overhead. It is a conservative admission heuristic, not a model tokenizer count or a guarantee of context fit. Details and configuration fields are in [capability-aware routing](capability-routing.md).

## How caching works

The cache tries to reuse a completed answer before paying for another generation.

### Exact cache

An exact cache key includes the canonical request and its model-affecting settings:

- Messages and system/developer instructions
- Model alias
- Temperature, token limits, stop sequences and other supported settings
- Tool definitions and response format
- Client identity
- Deployment/router/cache configuration, including the classifier artifact when present
- Per-request strict/fallback policy and deployment capability declarations

Explicit versus implicit sampling fields are also distinguished. This prevents a cached response generated with an omitted default from bypassing a provider's rejection of an explicitly requested unsupported parameter.

Example:

```text
First matching request:
  no cache entry -> generate "Paris" -> save successful completion

Later matching request:
  cache hit -> return saved completion -> no new backend generation
```

The default TTL is one hour. Only complete successful generations are cached; a normal finish reason is required, and a partial/error response is not stored. Successful completion does not itself guarantee that the answer is factually correct.

Streaming and non-streaming delivery can reuse the same completion. A streaming cache hit replays the saved answer without artificial token delays, using a fresh response ID. It can report the original completion's usage for API compatibility while recording no new backend generation cost.

The stored response's actual serving tier must also remain eligible for the current request. This applies to exact and semantic hits, so cached fallback answers cannot bypass strict pinning, an allowlist or the current context budget.

Exact caching can also reuse sampled answers at nonzero temperature. If a workload requires a fresh sample every time, its cache policy needs to account for that behavior.

### Semantic cache

Semantic caching tries to recognize different wording with the same meaning:

```text
"What is the capital of France?"
"Name France's capital city."
```

A local embedding model converts the prompt to a numeric vector. The gateway normalizes the vectors and compares cosine similarity. A hit is possible when the best candidate reaches the configured threshold `tau`.

High similarity can still produce a wrong answer:

```text
"Capital of Australia?"
"Capital of Austria?"
```

For that reason, semantic caching is **disabled by default**. The current numeric threshold is a configuration placeholder until real embedding measurements justify it.

Eligibility is deliberately narrow: one user message, temperature zero, no tools/tool choice, no structured-response format, and no assistant/tool conversation history. Leading system/developer messages and settings remain part of the namespace. Multi-turn requests can still use exact caching.

The current semantic index is a bounded Redis candidate scan, defaulting to 256 entries per namespace. Embeddings run through local Ollama, avoiding a hosted embedding API charge while still consuming local compute.

If embedding inference fails, the request proceeds without a semantic hit. A Redis read/quota failure returns 503. If a cache write fails after a successful generation, the completed answer can still be returned.

## How routing works

### Explicit selection

Sending `model: "claude"`, `"gemini"`, `"qwen"` or `"deepseek"` selects that configured alias as the preferred backend.

By default an explicit alias is a preference: an allowed compatible alternative may be selected if the preferred tier is incompatible or fails before output. The returned model alias identifies the backend that actually served the answer.

Set `"routing":{"mode":"strict"}` to require that exact alias, including on cache hits. Set `"routing":{"fallbacks":["gemini","deepseek"]}` to narrow alternatives in server-defined order. With `auto`, this list bounds the whole primary/retry pool; with an explicit alias, the alias itself is also permitted. Empty lists disable explicit-model alternatives or leave `auto` with no candidate. These gateway-only controls are never forwarded to providers.

### v0: the current default

For `model: "auto"`, v0 uses the combined message-text length and a small keyword classifier. With the current multi-provider catalog:

| Condition, in decision order | Preferred backend |
| --- | --- |
| Tools/tool history or a non-text response format | Configured large tier, when compatible |
| More than 2,400 characters, or `prove`, `reason`, `derive`, `debug`, `algorithm` appears | Claude |
| At most 600 characters, with no earlier match | Local Ollama tier |
| Otherwise | Gemini |

The keywords are simple substring rules. They do not reliably measure the true difficulty of every task. Qwen and DeepSeek remain available through explicit aliases and eligible fallback paths.

These rules produce a preference, which is checked against the eligible pool. If it is incompatible or outside the client's allowed pool, the router selects an eligible alternative in server fallback order, then large/medium/small preference order. For example, schema-less `json_object` excludes Claude's native adapter and can select Gemini directly without calling Claude first. Automatic initial selection still works when retrying a failed generation is disabled.

### v1: learned small-versus-large selection

The learned router estimates:

> How likely is the small model's answer to be good enough for this prompt?

Offline training follows this process:

1. Ask the configured small and large models the same questions.
2. Grade both answers with the same judge and rubric.
3. Label the pair using `small_score >= large_score - 0.5`.
4. Build features from local embeddings and log prompt length.
5. Train standardized logistic regression with prompt-family-separated data.
6. Sweep the routing threshold and compare cost against rubric agreement.

At runtime:

```text
probability >= chosen threshold -> small tier
otherwise                       -> large tier
```

The router predicts a label learned from prior examples. It does not know the correctness of the current answer in advance, and the normal request path does not call a judge model to score each generated answer.

The model is stored as JSON coefficients, with embedding/model identity checks. A production v1 artifact has not yet been trained from real model outputs; v0 remains enabled. A classifier preference or inference-error quality-tier preference goes through the same capability/policy filter. Even a high small-model probability cannot select an incompatible tier.

## How provider adapters work

Each adapter implements a shared interface:

```text
complete(request) -> asynchronous iterator of response events
```

Events can contain text, reasoning summaries, tool-call fragments, metadata, usage or a finish reason. A chunk is not necessarily a single token.

| Backend family | Provider communication |
| --- | --- |
| Claude | Native Anthropic Messages API and `x-api-key` authentication |
| Gemini | Native Google GenerateContent streaming and `x-goog-api-key` authentication |
| Hosted Qwen | Configured DashScope-compatible endpoint |
| Hosted DeepSeek | Configured DeepSeek-compatible endpoint |
| GPT and other compatible services | Configured Chat Completions endpoint with bearer authentication |
| Local models | Ollama chat API |

Adapters translate the gateway request into the provider's schema, then normalize response events into the gateway's OpenAI-style format. They handle provider-specific tool schemas, supported output formats, usage and completion reasons. Model-specific settings can select a token-limit parameter, omit unsupported implicit sampling defaults, disable upstream stream-usage options, or add supported provider body extensions.

The adapters do not make every feature interchangeable. See [the compatibility guide](providers.md#feature-compatibility) for exact restrictions. For example, the current Ollama adapter handles text/JSON generation but does not translate tool calls.

### Tool calls and signed context

When a model requests a function:

```text
Model emits tool call
       -> gateway returns call to your application
       -> your application executes the function
       -> your application sends a tool-result message
       -> gateway asks the model to continue
```

The gateway does not execute tools itself. Claude/Gemini can require signed context during continuation. Clients should preserve the returned assistant message, including `provider_metadata` and `tool_calls[].extra_content`. Streaming clients need to accumulate those fields alongside ordinary text and tool arguments.

## Streaming and ordinary responses

### Streaming: `stream: true`

The gateway forwards chunks through server-sent events (SSE) as the model generates them. It first fetches one backend output event before sending HTTP success headers, so an initial backend failure can still produce an appropriate HTTP error.

Simplified SSE illustration, omitting normal ID/model fields:

```text
data: {"choices":[{"delta":{"content":"The capital "}}]}

data: {"choices":[{"delta":{"content":"is Paris."}}]}

data: {"choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

If requested, a usage-only chunk is included before `[DONE]`. The client must also recognize error frames; HTTP 200 or `[DONE]` alone is insufficient to establish success.

### Non-streaming: `stream: false`

The gateway still consumes the backend stream internally, accumulates its events, and returns a completed JSON response. This reuses the same adapters and reliability behavior for both delivery modes.

## Reliability and failure handling

### Deadlines

The implementation distinguishes:

- **TTFT timeout:** maximum wait for the first backend output event.
- **Idle timeout:** maximum wait between backend events.
- **Tier total timeout:** total time available to that backend attempt.
- **Cross-attempt timeout:** total backend-phase budget across the fallback chain.

Current defaults are provisional. Real measurements should guide them, for example using roughly twice measured p95 TTFT as a starting point.

An important implementation detail: `request_timeout` currently starts when backend execution begins. It does not encompass all earlier authentication, Redis, embedding or routing work. Redis/embedding calls have their own timeouts. A true shared end-to-end deadline and slow-client write policy are recommended improvements below.

### Retry and fallback

Before output begins, eligible failures include timeouts, transport failures, HTTP 429 and backend 5xx errors. Invalid credentials and other non-retryable errors do not trigger repeated attempts.

At most two retries are allowed, visiting distinct configured tiers. Normally that means at most three backend attempts total. Backends excluded by capability checks or skipped by an open circuit do not consume an attempt. The backend pool independently enforces capabilities and client pinning/allowlists, even after the router has selected a preferred tier.

Example using the current catalog, with the required keys configured:

```text
Preferred local model fails before output
       -> try Claude
       -> if another retryable failure occurs, try Gemini
```

The chain is bounded; listing six providers does not mean all six are called for every request.

### Why fallback stops after output

Once a model has emitted part of an answer or tool call, switching models could combine incompatible outputs. The gateway therefore falls back only before its first backend output event. This conservative rule also applies to non-streaming requests even though their output is buffered.

After output begins, a failure terminates the response. Streaming clients receive an error record followed by `[DONE]`; no successful finish is issued and no partial response is cached.

### Circuit breaker

Default behavior:

```text
3 consecutive retryable failures
       -> open circuit
       -> skip backend for 30 seconds
       -> admit one recovery probe
       -> success closes circuit; failure starts another cool-down
```

This prevents a known unhealthy backend from adding a full timeout to every request. Circuit state is process-local.

When a streaming client disconnects, the upstream iterator is closed and resources are released. A disconnect while waiting for the first backend event can currently wait until output or the backend deadline; earlier cancellation detection is a proposed improvement.

## Metrics, traces and cost

### Prometheus and Grafana

Prometheus exposes measurements for requests by tier/outcome, end-to-end latency, backend TTFT, tokens, configured costs, cache outcomes, fallback attempts, circuit skips and rate-limit rejections.

The checked-in Grafana dashboard visualizes those metrics and includes sample counts beside percentile charts. Docker startup and a real dashboard screenshot remain deferred/pending.

### Langfuse-compatible tracing

For chat requests entering the handler, a trace records applicable steps:

```text
Chat request
  |-- Authentication
  |-- Rate limit
  |-- Capability/policy preflight
  |-- Cache lookup
  |-- Router decision, including v1 probability when used
  `-- Backend attempts
```

An optional OTLP exporter sends the spans to a configured Langfuse endpoint. Raw prompts, answers and API keys are omitted from these traces. Benchmark output files intentionally contain evaluated outputs; they are a separate data path.

### Token costs

Basic token-cost estimation:

```text
cost = (input_tokens * input_price_per_million
      + output_tokens * output_price_per_million) / 1,000,000
```

Provider usage is preferred. When unavailable, a character-based estimate is labelled as estimated. Gemini thought tokens are included in normalized completion usage; Anthropic cumulative usage is not repeatedly added as though each event were a separate charge.

These values are not a billing ledger. Invisible failed-attempt charges, provider cache discounts, tool fees and local infrastructure costs are not fully represented. New provider entries currently leave prices unconfigured: zero-valued default metrics do not mean their API is free. Explicit unknown-price handling is proposed below.

## Offline evaluation and learning

Evaluation scripts run separately from ordinary gateway traffic:

| Experiment | What it establishes |
| --- | --- |
| Local profiling | Compare three local models at two quantizations across 40 prompts: 240 measured jobs |
| Quality grading | JSON schema/exact-match checks locally; rubric grading by a selected judge for other tasks |
| Cache sweep | Compare 100 paraphrases and 100 meaning-changing near-misses; select and evaluate a similarity threshold |
| Router training | Learn small-versus-large labels and report cost/agreement sweeps on separated data |
| Failure injection | Measure errors/latency under the same scheduled faults with fallback on and off |
| Load testing | Record throughput, latency and error rates at increasing concurrency, with sample sizes |

The prompt suite has ten factual, ten summarization, ten JSON-extraction and ten reasoning tasks. Six model/quantization configurations produce the 240-job profile.

The cache sweep evaluates thresholds from 0.80 through 0.99, selecting on calibration data and reporting held-out behavior. The router uses group-separated evaluation rather than mixing related prompt families across training and testing. A small seed dataset still cannot establish broad production reliability.

The normal gateway does not use a judge on every request. That would add another model call, latency and cost. Evaluation instead supplies evidence for choosing cache thresholds, models and routing policies ahead of deployment.

Detailed commands and measurement definitions: [evaluation runbook](evaluation.md).

## Running the project now

### Demo mode

From the repository root:

```powershell
& ".venv\Scripts\python.exe" -m gateway.cli serve --config config/demo.yaml
```

Demo mode uses deterministic mock backends and in-memory state. It requires no hosted API credentials. Open `http://127.0.0.1:8000/docs` after starting it. The documented demo client key is `demo-key-change-me`.

### Real-provider mode

Use `config/providers.yaml` and arrange:

1. A generated gateway client key and its hashed record in `config/keys.json`.
2. A reachable Redis service, whether native, WSL or managed remotely.
3. API credentials for the hosted tiers you select.
4. Ollama when using local generation or local embeddings.

For example, once those prerequisites are configured:

```powershell
$env:GEMINI_API_KEY = "your-gemini-key"
$env:REDIS_URL = "redis://localhost:6379/0"
& ".venv\Scripts\python.exe" -m gateway.cli serve --config config/providers.yaml
```

Send `model: "gemini"` with your gateway client key. Explicit hosted requests can run without Ollama while semantic caching and v1 are disabled. Automatic routing requires the dependencies/credentials for the routes you enable.

Native commands use the shell environment. Compose reads its `.env` file automatically; the native CLI does not.

### Docker later

`Dockerfile`, `.dockerignore`, Compose services, provider-secret variables and monitoring configuration are prepared. The same provider catalog supports an `OLLAMA_BASE_URL` override so Compose can use service networking while native development uses localhost.

Containerization changes packaging and networking, not the routing/cache algorithms. Image build/start verification will happen when you introduce Docker; it is not required for current native development.

## What has actually been verified

The latest recorded local suite passed **96 test cases**, with one real-Redis integration test skipped because a Redis server was unavailable.

Provider protocol tests used `httpx.MockTransport` and dummy credentials such as `test-key`. They intercepted requests locally, checked URLs/headers/payloads, and returned synthetic SSE responses. No real hosted API keys were obtained or used for those tests.

Coverage includes token-bucket arithmetic, Lua concurrency through fakeredis, capability filtering, context/output admission, strict pinning/allowlists, policy-isolated cache hits, v1 eligibility checks, native and compatible provider translations, tool/signature round-trips, truncation/error handling, cross-provider fallback, disconnect cleanup, scoring, cost arithmetic and dataset separation.

The demo also completed 1,600 loopback streaming load requests and seeded failure experiments. Those measurements validate the local harness/gateway behavior, not model quality or production provider capacity.

Still pending: live hosted-provider checks, real Redis verification in this environment, actual local-model profiling/selection, calibrated semantic caching, production v1 training, and external monitoring/deployment validation.

## Recommended improvements

The first improvement below is now implemented and locally tested. The remaining items are proposals. Priorities focus first on correct model selection and predictable operation, then on efficiency and scale.

### Completed: capability-aware routing and fallback

**Problem addressed:** a larger model does not necessarily support the requested API feature. Previously, all structured requests preferred the large tier without filtering first, including schema-less JSON requests that Claude's native adapter cannot translate.

**Implemented:** per-tier capability records and a shared eligibility registry filter both routing and fallback. Supported checks include tools/strictness, JSON modes, sampling controls/combinations, message constraints, requested output limits and estimated input/context budgets. Incompatible candidates are skipped without consuming generation attempts. Empty eligible pools return a structured client error before cache embeddings or generation.

Strict pinning and fallback allowlists are included in cache identity, and cached actual tiers are checked again. Classifier predictions and inference failures cannot bypass these restrictions. Gateway-only policy fields are stripped from upstream requests.

**Verification:** local tests cover Claude-to-Gemini JSON rerouting, native adapter restrictions, strict-tool filtering, no-eligible early errors, disabled/restricted fallback, retry counts, context/output budgets, cached actual-tier checks, and v1 paths. See [usage and semantics](capability-routing.md).

**Remaining external check:** verify capability declarations against live provider/model versions when API keys are available. Context admission uses an estimate rather than exact model tokenization.

### P0: configuration diagnostics and explicit tier activation

**Current gap:** a listed alias may lack credentials, an accessible endpoint or a downloaded local model. `/health/ready` currently checks the state store and deliberately does not probe every provider.

**Improvement:** add a native `gateway doctor` command and explicit tier activation. Report selected routing paths, missing environment-variable names, invalid model/endpoint combinations and unconfigured prices without printing secrets. Keep configuration validation separate from an explicitly invoked live inference check. Validate the active fallback graph at startup.

**Done when:** an operator can configure a hosted-only setup without accidentally routing ordinary requests to an unavailable local tier, and can distinguish a valid configuration from a provider verified by a live call.

### P0: one shared deadline and earlier cancellation

**Current gap:** the cross-attempt deadline begins in the backend pool; earlier Redis/embedding/routing work has independent limits. A client can also disconnect while the gateway is still prefetching its first backend output.

**Improvement:** start one deadline at admission and propagate remaining time into each stage. Monitor disconnects during prefetch and cancel upstream work promptly. Specify bounded handling for stalled downstream writes separately from backend generation timeouts.

**Done when:** a slow embedding stage reduces the remaining backend budget; cancelled clients release work before TTFT expires; combined stage delays cannot silently exceed the configured processing deadline except for a documented cleanup margin.

### P0: unknown-cost handling and a stronger usage ledger

**Current gap:** missing prices default to zero, and final-response accounting cannot fully reconcile charges for failed or abandoned attempts.

**Improvement:** represent unknown pricing explicitly, require verified input/output prices for published cost curves, and record per-attempt usage with a provenance label such as provider-reported, estimated or unavailable. Keep cached-input, reasoning-token and infrastructure assumptions visible. Add configurable spend budgets after this accounting foundation is reliable.

**Done when:** an unpriced Claude/Gemini/Qwen/DeepSeek deployment is never presented as free; a fallback request's report shows both attempts and any unmeasured charge; cost comparisons use matching pricing/version metadata.

### P1: deduplicate simultaneous identical cache misses

**Current gap:** if many identical requests arrive before the first answer is cached, each can trigger a separate generation.

**Improvement:** add single-flight coordination for eligible requests, scoped to the existing client/request/configuration cache key. One leader generates; followers reuse the completed result. Define wait limits, ownership, failure propagation and cancellation behavior. Start with completion reuse rather than broadcasting a partially generated stream to many clients.

**Done when:** 100 simultaneously admitted identical eligible requests cause one successful backend generation; unrelated clients/settings remain isolated; leader failure/cancellation cannot leave followers waiting indefinitely. This does not bypass the existing client quota.

### P1: token-aware provider limits and per-tier admission

**Current gap:** one short request and one long generation each spend one request-bucket token. The global active-request limit also does not reflect a provider's RPM/TPM quotas or a local model's throughput.

**Improvement:** retain the existing 20/2 client policy and add separate token-budget reservations and concurrency limits per provider/tier. Estimate input plus bounded output before admission, reconcile known usage afterward, and handle provider `Retry-After` information within the deadline. Use bounded queues or immediate rejection with explicit metrics.

**Done when:** a burst cannot exceed configured provider concurrency; large requests consume larger token reservations; time spent waiting is included in the request budget; reservations are released or reconciled after every outcome.

### P1: stronger evaluation, calibration and version tracking

**Current gap:** the 40-prompt seed and templated cache pairs are small. Logistic-regression scores are not automatically calibrated probabilities. Agreement with a weak large-model answer can still be a misleading positive label.

**Improvement:** collect independently authored, realistic prompt families, including multi-turn, JSON, tools, long-context and meaning-changing details. Use distinct training, threshold-calibration and final-test groups. Audit judge decisions, track an absolute quality floor, measure probability calibration and define conservative behavior on unfamiliar inputs. Record model/provider revisions, generation settings, embedding identity and judge/rubric versions in artifacts.

**Done when:** the deployed model/threshold is evaluated on a final untouched set; curves report sample sizes, uncertainty and absolute quality alongside agreement; model revisions trigger a deliberate re-evaluation. Small false-hit samples are not described as certifying a universal 2% bound.

### P1: live provider contracts and client examples

**Current gap:** simulated responses cannot establish that a particular live model supports a parameter today. Application authors also need examples that handle SSE errors and signed tool metadata correctly.

**Improvement:** once keys arrive, add an explicitly invoked live smoke suite for each selected deployment: simple text, streaming termination, supported JSON mode, tool round-trip and usage. Store sanitized protocol fixtures/version metadata for repeatable regression checks. Add a small client example that accumulates metadata, detects partial failures and can request strict model selection when that option is introduced.

**Done when:** a dated provider/model support matrix is backed by live checks, routine CI remains credential-independent, and example clients detect an error frame even when the HTTP status is 200.

### P2: stronger cache lifecycle and storage isolation

**Current gap:** one TTL fits every answer; caches share Redis capacity with quota state; bounded vector scans eventually limit recall/scale. Redis logical databases do not independently reserve memory.

**Improvement:** add workload-appropriate TTL/bypass/invalidation policies and measure semantic-search latency and index coverage. At sufficient scale, separate cache storage from quota storage using independently bounded resources, then introduce a vector index if measurements justify it. Continue exact-only handling for requests whose meaning-changing details are not safely captured by the embedding.

**Done when:** cache growth cannot consume the quota store's memory allocation; stale-data policies are testable; any index migration preserves client/settings isolation and is benchmarked for hit quality as well as speed.

### P2: fuller lifecycle telemetry and realistic load evidence

**Current gap:** the request handler's trace begins after schema validation, leaving some early rejection paths outside its metrics. The demo load generator shares a machine/process environment with the mock gateway, and its closed-loop results do not establish production capacity.

**Improvement:** measure all relevant HTTP outcomes with low-cardinality labels, separate backend TTFT from client-visible TTFT, and expose phase latency plus successful/error sample counts. Run real load from a separate process or host, include sustained and arrival-rate-based workloads, and define alerts against measured availability/latency goals.

**Done when:** malformed/oversized requests are accounted for, fast failures cannot disguise slow successful requests, and every published percentile states its population and sample size.

### Later: container validation and optional GPU serving

Build/start the prepared images when Docker is introduced, verify health/provisioning, and capture the actual dashboard. Benchmark vLLM or GPU serving only when measured local-tier limits and budget justify it. Additional infrastructure is not a prerequisite for the native correctness improvements above.

## Suggested implementation order

1. **While waiting for API keys:** capability-aware routing/fallback is complete; next consider configuration diagnostics, shared deadlines/cancellation, and explicit unknown-price handling.
2. **Once one provider key is available:** validate that deployment live, configure real prices, run an actual streaming/tool/JSON smoke suite and collect a latency baseline. Start with an explicit alias before enabling broad automatic routing.
3. **Once comparison models are available:** finish profiling and paired quality collection, expand/calibrate evaluation data, then choose cache/router operating points.
4. **When duplicate traffic or quotas become limiting:** add single-flight coordination, token-aware quotas and per-tier admission based on measurements.
5. **When Docker is introduced:** validate the prepared deployment, monitoring and persistent state; evaluate scale-specific storage/GPU changes afterward.

**Best next feature to implement:** configuration diagnostics and explicit tier activation, to make native setup straightforward when the first provider key arrives.

## Where each part lives

| Path | Responsibility |
| --- | --- |
| `gateway/app.py` | API lifecycle, validation handling, caching/routing orchestration and response delivery |
| `gateway/auth.py` | Gateway client-key authentication |
| `gateway/state.py` | Redis Lua quota/cache state and demo in-memory state |
| `gateway/cache.py`, `gateway/embeddings.py` | Cache identity, semantic eligibility/lookup and local embeddings |
| `gateway/router.py` | v0 rules and v1 classifier inference |
| `gateway/capabilities.py` | Request requirements, deployment eligibility, pinning and allowlist enforcement |
| `gateway/backends.py`, `gateway/providers.py` | Shared backend interface and provider protocol adapters |
| `gateway/reliability.py` | Deadlines, retry budget and circuit breaker |
| `gateway/protocol.py` | SSE serialization and completion/tool/metadata accumulation |
| `gateway/telemetry.py` | Prometheus metrics and optional Langfuse-compatible OTLP tracing |
| `config/providers.yaml` | Native/container-aware multi-provider catalog |
| `experiments/` | Dataset export, profiling, grading, cache/router evaluation and load/fault scripts |
| `tests/` | Automated behavior and protocol checks |
| `results/`, `artifacts/` | Generated measurements and trained router coefficients |
| `.venv/` | Isolated local Python environment |
| `Dockerfile`, `compose.yaml`, `ops/` | Prepared container and monitoring assets |

Related guides: [provider setup](providers.md), [architecture decisions](architecture.md), [evaluation runbook](evaluation.md), and the [delivery plan](../PLAN.md).

In one sentence: **the gateway gives your application one consistent API that manages model selection, response reuse, reliability and measurement around different LLMs.**
