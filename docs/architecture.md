# Architecture and design decisions

## Request lifecycle

1. Bound the incoming body to 1 MiB and validate the supported Chat Completions contract.
2. Authenticate the bearer secret against SHA-256 hashes in a file. Keys are generated from 256 bits of randomness; changing a file takes effect on process restart.
3. Run an atomic Redis Lua token bucket, per key identity. The script reads Redis server time, clamps backwards clock movement, refills and consumes in one transaction. State expires after the refill-to-capacity interval plus one second.
4. Admit up to the configured number of active requests. There is no unbounded backend queue; excess requests get 503.
5. Run pure capability/policy preflight: validate tool/format envelopes and filter deployment candidates against adapter constraints, declared model features and configured size limits. Reject an empty eligible pool before cache or embedding calls.
6. Try exact cache, then eligible semantic cache, filtering each stored response's actual tier. Embedding failures bypass semantic matching; Redis failures during auth/rate/cache reads return 503. Cache-write failure does not invalidate an already successful answer.
7. Use an explicit preference or v0/v1 to select from the compatible allowed pool. Structured/tool requests prefer the large tier when eligible. Classifier inference failures prefer the quality tier through the same capability filter.
8. Attempt the selected backend and independently capability-filtered fallback tiers, respecting pinning/allowlists. Prefetch the first backend event before committing HTTP response headers.
9. Aggregate to OpenAI JSON or stream OpenAI-style SSE deltas. Store complete successful answers only. Record metrics and end the request trace after streaming finishes or is cancelled.

## Protocol contract

The implemented endpoint is a **text Chat Completions subset**, not the whole OpenAI API. Supported fields are defined in `gateway/models.py` and exposed through `/docs`:

- `model`: `auto` or a configured tier name; the gateway returns the actual tier alias used.
- `messages`: system, developer, user, assistant or tool roles, string/null content, optional name/tool-call data. Content arrays, images and audio are not supported.
- `stream`, `stream_options.include_usage`, temperature, top-p, one of max-tokens/max-completion-tokens, stop, seed, frequency/presence penalties.
- Tools, tool choice, parallel tool calls, response format, user and `n=1`.
- Gateway-only `routing.mode` (`prefer` or `strict`) and `routing.fallbacks`; these are removed from provider payloads.

Hosted adapters preserve tool-call argument fragments and usage. `openai_compatible` supports arbitrary configured endpoints/model IDs, including Qwen and DeepSeek. Native `anthropic` and `gemini` adapters translate system instructions, conversation turns, function schemas/results, supported JSON formats and provider-specific SSE into the shared interface. Anthropic uses `x-api-key`; Gemini uses `x-goog-api-key` in a header, not in the URL. Provider-specific limitations are in [the compatibility guide](providers.md#feature-compatibility).

Assistant messages may include `reasoning_content`, opaque `provider_metadata`, and `tool_calls[].extra_content`. Accumulation/cache preserve these extensions so signed thinking/function-call context can be round-tripped. Native metadata is not sent as an unknown field to OpenAI-compatible endpoints. Claude's cumulative usage is not summed twice; Gemini completion usage includes thought tokens.

Ollama maps text/JSON format and sampling options; tool messages/calls and frequency/presence penalties are explicitly rejected. The gateway never executes tools. Other OpenAI fields, logprobs, multiple choices and the Responses API are out of scope.

Unknown fields produce a sanitized HTTP 400 error. Pre-output backend failures preserve error HTTP status. SSE has `data: {chat.completion.chunk}` records, an optional usage-only chunk, and `data: [DONE]`. An HTTP 200 stream is successful only if it has a normal finish reason and no error frame.

## Streaming and deadlines

- The HTTP status is not sent until the first backend output or valid completion is available. Role-only and empty keepalive deltas do not commit a backend selection.
- Only pre-output timeout, connection failure, upstream 429 or 5xx may fall back. Invalid provider credentials and other 4xx are non-retryable.
- A partial stream is never restarted on another model. Emit an error record and terminate it. Non-streaming requests follow the same conservative rule: buffering output does not authorize a retry after generation starts.
- Configure TTFT, inter-event idle, per-tier total and cross-attempt request deadlines separately. TTFT defaults are provisional. Use measured p95 as input to a roughly 2×p95 candidate deadline, then test failures under load.
- Deadline checks bound backend waits, including time already consumed between yielded events. They are not slow-client socket write deadlines; configure those at the ingress/reverse proxy.
- ASGI streaming cancellation closes the upstream async iterator and releases admission/circuit resources. A disconnect during prefetch can wait until the first-event deadline; requests do not run without a bound.
- Circuit state is local to a gateway process: N consecutive failures open it, cool-down expiry admits one probe, success closes it. Run one process per container. Shared Redis rate limits work across containers; cross-replica circuit coordination is a future extension.

## Cache correctness

Exact keys include canonical messages and all supported model-affecting parameters. `stream` and `stream_options` affect delivery only, so the same completion can satisfy streaming and non-streaming calls. API-key identity and a hash of deployment/router/cache/reliability policy also participate; the classifier artifact content is hashed.

Routing policy and deployment capability declarations also participate. Every cache lookup filters on the actual stored tier's current eligibility. Strictly pinned requests therefore cannot inherit another model's response from earlier fallback or semantic reuse.

Explicit versus implicit sampling fields are distinguished: a model may omit an unsupported default, but must reject an explicit client request for it. A cached implicit-default response therefore cannot bypass the capability check. Semantic caching is disallowed when a tier omits temperature control. Environment-resolved backend/embedding URLs are included in the deployment policy hash.

Semantic namespaces remove only the single eligible user's text while preserving its position and every other field. System/developer text, temperature, seed, stop, token limits, tools, user and model alias cannot leak across namespaces. Semantic matching is limited to deterministic (`temperature=0`), single-user, tool-free requests without structured-output formatting. Changing embeddings or deployed models invalidates old namespaces.

Vectors are normalized, finite, nonzero, and compared using cosine similarity. A bounded Redis sorted-set index points to TTL-bound JSON response/vector entries. Lua insertion trims old entries atomically; hash tags keep keys in one Redis Cluster slot. At most 256 candidates per namespace are scanned by default. Redis `noeviction` avoids silently evicting rate-limit state when memory is full; state-store errors fail closed. Monitor size and split state/cache stores when scaling.

Exact caching at nonzero temperature deliberately reuses a previously sampled answer. Disable caching for workloads that require a fresh sample each time. Concurrent identical misses can generate duplicate calls; distributed single-flight was rejected for this first version because lock ownership, cancellation and stalled generators complicate correctness. It can be added after measuring stampede frequency.

## Routing and evaluation

v0 uses prompt length and a small hard-task keyword classifier. v1 uses standardized local embeddings plus log prompt length and a logistic classifier. The label is `small_score >= large_score - 0.5`; it is agreement with rubric quality, not ground-truth correctness.

`gateway/capabilities.py` centralizes request requirements, adapter ceilings, deployment limits and policy filtering. Automatic candidates come from router aliases plus server fallback destinations. Explicit preferences may substitute only through enabled server fallback policy. Rejected candidates do not consume generation retries. Initial auto selection remains possible when generation retries are disabled. See [capability routing](capability-routing.md) for client controls and exact ordering.

Context admission uses serialized UTF-8 bytes plus overhead as a conservative input estimate. It may over-reject and does not guarantee provider context fit. Requested output limits are checked directly. This avoids presenting an approximate cross-provider estimate as an exact tokenizer result.

Training uses prompt-family-separated groups, records held-out predictions, then refits deployment coefficients. Artifacts are plain JSON, with validated dimensions/scales and matching embedding/tier identities. Pickle was rejected because loading it executes arbitrary Python. A changed tier or embedding model requires retraining.

The cost/quality report compares paired small/large answers. Its v0 curve is explicitly a two-tier ablation: medium maps to large. This makes cost accounting honest without inventing missing medium-tier answers. Full three-tier router evaluation needs triples and a larger labelled suite.

## Observability and accounting

All metric labels are fixed low-cardinality tiers/outcomes, never API keys, user IDs or prompts. Tokens use provider counts where available; otherwise a clearly labelled character-based estimate. Cache hits reuse original API usage for compatibility but add no new backend spend. Costs are configured standard token estimates, not billing invoices: failed attempts without usage, provider prompt-cache discounts, tool fees and local hardware are not known to the gateway.

OTLP export uses an optional Langfuse endpoint with Basic authentication from environment variables. Root request IDs correlate traces with response headers. Spans report decisions and timings, with no raw prompt or output capture. Export queues are bounded by the OpenTelemetry SDK.

## Alternatives considered

| Choice | Why | Rejected/deferred alternative |
| --- | --- | --- |
| FastAPI + httpx | Async SSE and testable transport adapters | Synchronous proxy which ties workers to slow streams |
| Hashed key file | Small, inspectable deployment with no extra database | Postgres key lifecycle/admin UI until rotation/tenancy requirements justify it |
| Redis Lua limiter | Atomic arithmetic and one authoritative clock | Client-side read/modify/write with races |
| Bounded Redis vector scan | Works on ordinary Redis without specialized modules | HNSW/Redis Search until candidate scale justifies indexing |
| Local Ollama embeddings | No per-request hosted embedding charges | Hosted embeddings for the cache |
| Logistic classifier | Fast, inspectable, probability threshold sweeps | Opaque routing LLM with its own latency/cost |
| Explicit pre-output fallback | Preserves response coherence | Splicing answers from multiple models |
| Compose, CPU-first | Reproducible low-cost local setup | Kubernetes/vLLM/GPU rental until profiling establishes a need |
