# Claude, Gemini, Qwen, DeepSeek and other models

The client-facing API stays OpenAI-compatible while adapters translate requests to the selected provider. **[config/providers.yaml](../config/providers.yaml)** is a ready-to-edit multi-provider catalog; GPT is optional.

| Family | Adapter (`provider`) | Example model ID | Credential environment variable |
| --- | --- | --- | --- |
| Claude | `anthropic` — native Messages API | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| Gemini | `gemini` — native GenerateContent API | `gemini-2.5-flash` | `GEMINI_API_KEY` |
| Hosted Qwen | `openai_compatible` — DashScope | `qwen-plus` | `DASHSCOPE_API_KEY` |
| Hosted DeepSeek | `openai_compatible` — DeepSeek API | `deepseek-chat` | `DEEPSEEK_API_KEY` |
| GPT | `openai` | `gpt-4o-mini` | `OPENAI_API_KEY` |
| Local Qwen/DeepSeek/Llama/Gemma/etc. | `ollama` | Any installed chat-model tag | Usually none |
| Other compatible services | `openai_compatible` | Provider's model ID | Any name set in `api_key_env` |

Model IDs are configurable examples, not a guarantee of availability in your account. Use the correct provider/region endpoint. Other compatible services can include OpenRouter, Together, Groq, vLLM and LM Studio, subject to their supported request fields. The native Gemini adapter targets the Gemini Developer API; Vertex AI and Bedrock authentication/protocols require their own integration.

## Does the API have to stay the same?

No. Each tier has its own `provider`, `model`, `base_url` and `api_key_env`. You can rotate keys, change models, or use different compatible providers for the small and large tiers. One provider account/key with access to both models is enough to start.

The `openai` and `openai_compatible` adapters use text Chat Completions with bearer authentication and streaming SSE. Claude uses its own `x-api-key`/Messages protocol; Gemini uses `x-goog-api-key` and `streamGenerateContent`. Check that the selected models support the requested features; an API key alone does not grant access to every model.

| Change | What to do |
| --- | --- |
| Rotate the key for the same provider/model | Set the new environment value and restart the gateway; existing quality benchmarks remain applicable. |
| Change the provider or endpoint | Update `base_url` and `api_key_env`; verify protocol compatibility, prices, quotas and latency. Recheck quality if the deployment/model behavior differs. |
| Change the small or large model | Update `model`, prices and deadlines; collect/grade new paired answers and retrain v1. Use v0 until the replacement artifact is ready. |
| Change the local model or quantization | Rerun the affected profile and quality measurements; retrain v1 if its small tier changed. |
| Change the embedding model | Recalibrate the semantic-cache threshold and retrain v1, since its features change. |

Changing tier configuration changes the cache policy namespace. Router artifacts also check model identity at startup. Use separate result files for different model experiments. For fair comparisons, keep the provider/model versions and grading judge fixed within an experiment; if changing the judge, regrade both sides with the same rubric and judge.

## Use real APIs without Docker

The existing `.venv` is the development environment. `config/providers.yaml` supports the families above; `config/native.yaml` retains the original local + hosted-small/large example. Both use `localhost` for Ollama natively, with an optional `OLLAMA_BASE_URL` override for later container networking.

1. Make Redis available through a native/WSL installation or a managed Redis endpoint. Set `REDIS_URL` accordingly (`rediss://` for TLS where required).
2. Install/run Ollama when using the local tier. Pull `all-minilm` before enabling semantic caching or v1. Explicit hosted requests with those features disabled do not require Ollama.
3. Generate a gateway client key:

   ```powershell
   & ".venv\Scripts\gateway.exe" keygen --id client-1
   ```

   Save the printed hashed record inside an array in `config/keys.json`. The generated gateway key authenticates your client to this gateway; your provider API key authenticates the gateway to the hosted model. They are separate credentials.

4. Set the key for a provider you want to use and the Redis endpoint in the same shell that starts the process. For example, with Gemini:

   ```powershell
   $env:GEMINI_API_KEY = "your-gemini-key"
   $env:REDIS_URL = "redis://localhost:6379/0"
   & ".venv\Scripts\python.exe" -m gateway.cli serve --config config/providers.yaml
   ```

   Native commands use shell environment variables. The `.env` example is consumed automatically by Compose, not by the native CLI.

5. Send `model: "gemini"` using your gateway client key. Use `"claude"`, `"qwen"`, `"deepseek"`, `"gpt"` or `"local"` for the other catalog entries. Only selected tiers need provider credentials. `model: "auto"` uses the configured routing/fallback tiers, so configure credentials for the paths you enable.

Example request body:

```json
{
  "model": "gemini",
  "messages": [{"role": "user", "content": "Explain what an LLM gateway does."}],
  "stream": true
}
```

You can instead keep aliases such as `small` and `large` and replace their provider/model settings. Alias names do not determine the backend family. For hosted-only automatic routing, point `router.small_tier`, `medium_tier` and `large_tier` to the configured hosted aliases; remove unused fallback destinations.

When collecting or grading natively, select the same config explicitly:

```powershell
& ".venv\Scripts\gateway-eval.exe" collect --config config/providers.yaml
& ".venv\Scripts\gateway-eval.exe" grade --config config/providers.yaml
& ".venv\Scripts\gateway-eval.exe" pair --small local --large claude
& ".venv\Scripts\gateway-eval.exe" train-router --config config/providers.yaml
```

Collection defaults to the configured router small/large aliases; grading defaults to its large alias. Override `--tiers` or `--judge-tier` as needed. The grader uses Claude's JSON-schema format automatically. Set each model's **input/output prices and verification date** before reporting costs: new provider entries explicitly leave prices unconfigured, and zero-valued default metrics are not evidence that the APIs are free.

## Feature compatibility

All adapters support ordinary text chat, streamed output and non-streaming aggregation. Native adapters normalize finish reasons, usage and errors into the gateway's response format.

The gateway now performs [capability-aware selection and fallback](capability-routing.md). Configure per-model capability records, inspect effective capabilities in `/v1/models`, and use `routing.mode: strict` to pin a model or `routing.fallbacks` to restrict alternatives. Unsupported tiers are excluded before calls; a request with no compatible allowed tier gets HTTP 400. Generic `openai_compatible` endpoints require explicit declarations for tools, strict tools, JSON modes and seed support.

- **Claude:** system/developer instructions, multi-turn text, function tools/results, streamed argument fragments, JSON-schema output on supporting models, and preserved signed thinking blocks. Native schema-less `json_object`, seed and frequency/presence penalties make this tier ineligible; the gateway can select an allowed compatible alternative, or return 400 when pinned/none is eligible. The grader supplies a schema when using Claude. Some Claude versions reject combined temperature/top-p or certain sampling controls.
- **Gemini:** system instructions, multi-turn text, function calls/results, JSON output/schema on supporting models, finish-reason normalization and thought-token accounting. Signed response parts and function-call signatures are retained for subsequent tool turns. `parallel_tool_calls: false` and OpenAI's `strict: true` tool flag are explicitly unsupported by this translator.
- **Qwen/DeepSeek/other compatible APIs:** model IDs and supported parameters are forwarded through the compatible protocol; `reasoning_content` is retained for reasoning-model responses and subsequent assistant turns. Feature availability depends on the exact endpoint/model.
- **Ollama:** local text and JSON generation plus sampling controls. The current Ollama adapter does not translate tools or frequency/presence penalties.

For native tool/thinking conversations, round-trip the **whole returned assistant message**, including `provider_metadata` and `tool_calls[].extra_content`. Streaming clients must accumulate those extension fields along with text/tool arguments. This preserves signatures rather than reconstructing signed data from text. The gateway returns tool calls but does not execute them.

This is a text-chat gateway. Images, audio, provider-hosted search/computer tools and every provider-specific endpoint are outside the current contract. Native providers require leading system/developer messages; interleaved system instructions and named non-tool messages are rejected rather than reordered or discarded.

### Model-specific configuration

Optional per-tier settings handle differences between compatible models:

```yaml
provider: openai_compatible
base_url: https://your-provider.example/v1
model: your-model-id
api_key_env: YOUR_PROVIDER_KEY
token_limit_parameter: max_completion_tokens # Default: max_tokens
send_stream_usage: false # For endpoints that reject stream_options
omit_parameters: [temperature, top_p] # For models without these sampling controls
extra_body:
  reasoning_effort: medium # Only if your provider/model supports this extension
```

`omit_parameters` removes implicit gateway defaults only. An explicit request for an omitted parameter makes that tier ineligible; the gateway selects an allowed compatible alternative or returns 400. Cache lookup cannot bypass this check. Semantic caching must remain disabled when any tier omits temperature control. `extra_body` adds deployment-specific fields; it cannot overwrite model, messages, streaming, routing policy or translated client fields. Qwen endpoints that expose `enable_thinking` can configure it through `extra_body`.

## Different keys/providers for each tier

In your selected config, set the appropriate tier's `api_key_env` to its credential variable. The multi-provider catalog already uses independent provider key names. For the `config/native.yaml` small/large example, you can use `SMALL_API_KEY` and `LARGE_API_KEY`. Set the model IDs, adapter types, base URLs and prices in those same tier entries. Compatible base URLs end before `/chat/completions`; native Claude URLs end at `/v1` and native Gemini URLs at `/v1beta`.

```powershell
$env:SMALL_API_KEY = "small-provider-key"
$env:LARGE_API_KEY = "large-provider-key"
& ".venv\Scripts\python.exe" -m gateway.cli serve --config config/native.yaml
```

The two values can also be the same key if one account serves both models. Settings are deployment choices; the client still sends `model: auto`, `small`, `large` or `local`.

## Container handoff later

`Dockerfile`, `.dockerignore`, `compose.yaml`, the Compose-oriented config and monitoring provisioning are prepared. Docker execution is deferred by request. Image build/start validation and a real Grafana screenshot will happen when you introduce Docker.

At that point:

- Set `GATEWAY_CONFIG=config/providers.yaml` in your Compose `.env` to keep using the same catalog. Compose sets `OLLAMA_BASE_URL=http://ollama:11434`, updating both generation and embedding endpoints without changing native defaults. `config/native.yaml` also supports this override; the original `config/gateway.yaml` remains available.
- Supply secrets through Compose environment variables; `.env.example` and Compose already list Anthropic, Gemini, DashScope, DeepSeek, OpenAI and optional small/large keys. If adding a custom variable, pass that name through Compose too.
- Mount `config/keys.json` and any trained artifacts using the existing config/artifact volume mounts. The credential file is excluded from the image build context.
- Start the demo stack with `docker compose up --build -d`, or the configured real stack with `docker compose --profile models up --build -d`.

The application runs natively now; container deployment remains a later infrastructure step.
