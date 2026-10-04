# Capability-aware selection, pinning and fallback

The gateway checks a request against the configured deployment capabilities **before cache lookup, embedding inference or backend calls**. Both v0 and v1 choose from the eligible pool, and the backend pool independently filters retry destinations. Unsupported tiers consume neither backend attempts nor retry budget.

These checks validate declared capabilities and known adapter restrictions. They do not query provider availability, credentials or actual model feature support. Live verification remains required when API keys are available.

## Request controls

The optional gateway-only `routing` object defaults to `{"mode":"prefer"}`. It is part of cache identity and is never forwarded to a provider.

### Strict model pinning

```json
{
  "model": "claude",
  "messages": [{"role": "user", "content": "Explain Redis briefly."}],
  "routing": {"mode": "strict"},
  "stream": true
}
```

Only `claude` may answer. An incompatible request gets HTTP 400 before a backend call; a provider failure is returned instead of switching models. The cache cannot supply an answer from another tier. Strict mode requires an explicit model alias and rejects nonempty fallback lists.

### Prefer a model, restrict its alternatives

```json
{
  "model": "claude",
  "messages": [{"role": "user", "content": "Return a JSON object with a greeting."}],
  "response_format": {"type": "json_object"},
  "routing": {"fallbacks": ["gemini", "deepseek"]}
}
```

Claude's native adapter cannot serve schema-less JSON mode. If fallback is enabled, selection can move directly to a compatible permitted tier without calling Claude first. If a backend later fails before output, retries remain restricted to compatible permitted destinations.

The list is an **allowlist**, not an ordering instruction. Server `fallback_order` determines order. A client can narrow the server policy but cannot enable retries disabled by the server or add destinations outside its configured fallback order. For an explicit model, `fallbacks: []` disables alternatives. Unknown or duplicate aliases return HTTP 400.

### Restrict automatic selection

```json
{
  "model": "auto",
  "messages": [{"role": "user", "content": "Summarize this paragraph."}],
  "routing": {"fallbacks": ["gemini", "deepseek"]}
}
```

With `auto`, there is no explicit primary alias, so the list bounds the **entire selection pool**, including the first backend. The server's auto pool consists of its router small/medium/large aliases plus its configured fallback destinations. The client's list intersects that pool; `fallbacks: []` permits no backend and produces HTTP 400.

Server `fallback_enabled: false` or `max_retries: 0` disables retrying a failed generation. It still permits automatic initial selection among compatible allowed candidates. For an explicit model, those server settings prevent capability-based substitution as well as failure fallback.

## Deployment capability records

Each tier accepts a `capabilities` section. Example deployment policy:

```yaml
- name: small
  provider: openai_compatible
  model: your-model-id
  base_url: https://your-provider.example/v1
  api_key_env: SMALL_API_KEY
  capabilities:
    tools: true
    strict_tools: false
    tool_choice: true
    json_object: true
    json_schema: false
    tools_with_response_format: false
    seed: false
    max_output_tokens: 4096
    context_window_tokens: 32768
```

Inspect the effective record in authenticated `GET /v1/models`. No API keys or endpoint credentials are returned.

### Supported configuration fields

| Fields | Purpose |
| --- | --- |
| `tools`, `strict_tools`, `tool_choice` | Function definitions/history, strict argument schemas and tool-selection controls |
| `parallel_tool_calls`, `disable_parallel_tool_calls` | Ability to accept explicit parallel enable/disable controls |
| `json_object`, `json_schema` | Supported response formats |
| `tools_with_response_format` | Whether function tools and a structured response format can be combined |
| `seed`, `stop` | Support for these optional generation controls |
| `temperature`, `top_p`, `frequency_penalty`, `presence_penalty` | Whether the deployment can accept these sampling controls |
| `temperature_with_top_p`, `max_temperature` | Sampling-combination and temperature-range restrictions |
| `named_messages`, `interleaved_system` | Message-shape capabilities |
| `max_input_tokens`, `context_window_tokens` | Estimated input and combined input/output admission budgets |
| `max_output_tokens`, `max_stop_sequences` | Requested output-budget and stop-sequence limits |

Unspecified boolean restrictions generally inherit the adapter-level defaults. For a new `openai_compatible` tier, **tools, strict tools, JSON-object mode, JSON-schema mode and seed default to disabled unless explicitly declared**. A compatible wire protocol alone is insufficient evidence of these features. Numeric limits are optional; unset limits do not establish an unlimited model context.

Adapter restrictions always apply, even if a capability is set to true:

- Ollama's current adapter does not translate tools or frequency/presence penalties.
- Claude's native adapter does not accept schema-less JSON mode or seed/penalty controls, and its maximum translated temperature is 1.
- Gemini's translator does not support OpenAI strict tool schemas or disabling parallel calls.
- Native Claude/Gemini require leading system messages and do not translate named non-tool messages.

The supplied catalog records conservative deployment limits and feature declarations. They are not advertised vendor maxima or proof that a model version is available. Update them after checking the exact deployment.

### Sampling defaults

Existing `omit_parameters` still controls which implicit gateway defaults are omitted from provider requests. Explicit requests for an omitted parameter are incompatible with that tier; another permitted tier can be selected, or the gateway returns 400.

If a model cannot accept temperature or top-p at all, configure the corresponding omission explicitly. Merely setting a capability to false must not silently change the meaning of a default the gateway would otherwise send. Semantic caching remains disabled for configurations with an omitted temperature control.

### Context estimates

The input estimate is:

```text
UTF-8 bytes of compact JSON containing messages, tools and response_format
+ 16 per message
+ 64 fixed overhead
```

The gateway checks that estimate against `max_input_tokens`, and checks `estimate + requested_output_tokens` against `context_window_tokens`. It independently checks the requested output budget against `max_output_tokens`.

This is a conservative admission heuristic, **not a provider tokenizer count or a guarantee that a request fits**. It counts schema and signed metadata bytes as well as text, and can reject requests that a true tokenizer would accept. Configure budgets for your deployment; model-specific tokenizers/provider token-count endpoints can replace the estimator later. Provider-reported usage still drives generation metrics when available.

## Selection and failure semantics

1. Authenticate, enforce the existing quota, and admit active work.
2. Validate policy aliases and common tool/response-format envelopes.
3. Compute eligible tiers using model declarations, adapter constraints, request controls and limits.
4. If none are eligible, return 400 `no_compatible_tier` with per-tier reason codes. No cache embedding or generation is attempted.
5. Try cache entries whose stored actual tier remains eligible for this request.
6. Compute the usual v0/v1 preference. If it is ineligible, choose the first eligible tier in server fallback order, then large/medium/small preference order.
7. Attempt generation. The backend pool rechecks its retry destinations independently.

Examples of rejection reasons include `unsupported_json_object`, `unsupported_strict_tools`, `unsupported_temperature`, `output_token_limit_exceeded` and `context_window_estimate_exceeded`.

Capability substitution before a call is recorded separately from retrying a failed backend. Neither model pinning nor capability filtering changes the streaming rule: **after the first backend output event, the gateway never switches models**. It emits a stream error on a subsequent failure.

The registry does not silently convert JSON formats, remove tools, truncate the requested output budget, or increase the server retry limit. A classifier probability—even a high one—cannot override incompatibility, and the classifier-error quality-tier path goes through the same filter.

## Cache and telemetry

- Request `routing` controls are part of exact and semantic namespaces.
- Deployment capabilities participate in the configuration policy hash.
- Cached responses are checked using their actual stored tier before being returned, including semantic matches whose prompt lengths may differ.
- `capabilities` spans record eligible aliases and the estimated input budget.
- `gateway_capability_rejections_total{tier,reason}` records rejected candidates in successful preflight evaluations; wholly incompatible requests appear in request outcomes.
- `gateway_capability_reroutes_total{from_tier,to_tier}` records initial preference changes caused by capabilities/client policy.

The HTTP preflight and backend pool use this admission policy. Offline profiling/grading scripts call adapters directly and do not exercise the HTTP router or its admission limits.

## Verification

Tests cover native adapter restrictions, configurable output/context limits, Unicode/schema input estimates, initial Claude-to-Gemini JSON routing, no-eligible early rejection, strict pinning, allowlists, retry budgets, post-output errors, cache isolation and actual-tier checks, v1 inference/error paths, and removal of gateway-only policy fields from upstream requests.

These are local behavior and simulated protocol tests. Live model compatibility remains to be verified with real credentials. Docker execution remains deferred.
