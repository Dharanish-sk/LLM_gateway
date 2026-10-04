"""Native provider protocols translated to the gateway's OpenAI-style event interface."""

import json
import uuid
from collections.abc import AsyncIterator
from urllib.parse import quote

import httpx

from gateway.backends import (
    BackendError,
    BackendEvent,
    check_status,
    provider_key,
    sse_data,
    validate_omissions,
    with_extra_body,
)
from gateway.config import Tier
from gateway.models import ChatRequest


def invalid(reason: str):
    raise BackendError(reason, 400, False)


def system_text(request: ChatRequest) -> list[str]:
    system, conversation_started = [], False
    for message in request.messages:
        if message.role in {"system", "developer"}:
            if conversation_started:
                invalid("native_provider_requires_leading_system_messages")
            system.append(message.content or "")
        else:
            conversation_started = True
        if message.name and message.role != "tool":
            invalid("native_provider_message_names_not_supported")
    return system


def arguments(value) -> dict:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except ValueError:
        invalid("invalid_tool_arguments")
    if not isinstance(parsed, dict):
        invalid("tool_arguments_must_be_object")
    return parsed


def functions(request: ChatRequest) -> list[dict]:
    result = []
    for tool in request.tools or []:
        if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            invalid("only_function_tools_supported")
        function = tool["function"]
        if not function.get("name"):
            invalid("tool_name_required")
        result.append(function)
    return result


def append_turn(turns: list[dict], role: str, blocks: list[dict], field: str):
    if not blocks:
        return
    if turns and turns[-1]["role"] == role:
        turns[-1][field].extend(blocks)
    else:
        turns.append({"role": role, field: blocks})


def schema_format(request: ChatRequest) -> dict | None:
    value = request.response_format or {"type": "text"}
    kind = value.get("type")
    if kind == "text":
        return None
    if kind == "json_object":
        return {"type": "object"}
    if kind == "json_schema":
        schema = value.get("json_schema", {}).get("schema")
        if isinstance(schema, dict):
            return schema
    invalid("unsupported_response_format")


class AnthropicBackend:
    def __init__(self, tier: Tier, client: httpx.AsyncClient):
        self.tier, self.client = tier, client

    def payload(self, request: ChatRequest) -> dict:
        validate_omissions(self.tier, request)
        if request.seed is not None or request.frequency_penalty or request.presence_penalty:
            invalid("anthropic_seed_or_penalties_not_supported")
        if request.temperature > 1 and "temperature" not in self.tier.omit_parameters:
            invalid("anthropic_temperature_must_be_at_most_one")
        if request.response_format and request.response_format.get("type") == "json_object":
            invalid("anthropic_requires_json_schema_format")
        system = system_text(request)
        turns = []
        for message in request.messages:
            if message.role in {"system", "developer"}:
                continue
            blocks = []
            if message.role == "tool":
                blocks.append({"type": "tool_result", "tool_use_id": message.tool_call_id, "content": message.content or ""})
            else:
                preserved = (message.provider_metadata or {}).get("anthropic", {}).get("thinking_blocks", [])
                if message.role == "assistant":
                    for block in preserved:
                        if block.get("type") not in {"thinking", "redacted_thinking"}:
                            invalid("invalid_anthropic_thinking_metadata")
                        blocks.append(block)
                if message.content:
                    blocks.append({"type": "text", "text": message.content})
                for call in message.tool_calls or []:
                    function = call.get("function", {})
                    if not call.get("id") or not function.get("name"):
                        invalid("invalid_tool_call")
                    blocks.append({"type": "tool_use", "id": call["id"], "name": function["name"], "input": arguments(function.get("arguments", "{}"))})
            append_turn(turns, "assistant" if message.role == "assistant" else "user", blocks, "content")
        if not turns:
            invalid("conversation_required")
        payload = {"model": self.tier.model, "messages": turns, "stream": True, "max_tokens": request.max_tokens or request.max_completion_tokens}
        if system:
            payload["system"] = "\n\n".join(system)
        if "temperature" not in self.tier.omit_parameters:
            payload["temperature"] = request.temperature
        # Omit neutral top_p: some Claude versions reject simultaneous temperature + top_p.
        if request.top_p != 1 and "top_p" not in self.tier.omit_parameters:
            payload["top_p"] = request.top_p
        if request.stop is not None:
            payload["stop_sequences"] = [request.stop] if isinstance(request.stop, str) else request.stop
        specs = functions(request)
        if specs:
            payload["tools"] = []
            for function in specs:
                tool = {"name": function["name"], "input_schema": function.get("parameters", {"type": "object", "properties": {}})}
                for key in ("description", "strict"):
                    if key in function:
                        tool[key] = function[key]
                payload["tools"].append(tool)
            choice = request.tool_choice or "auto"
            if isinstance(choice, str) and choice in {"auto", "none", "required"}:
                payload["tool_choice"] = {"type": {"auto": "auto", "none": "none", "required": "any"}[choice]}
            elif isinstance(choice, dict) and choice.get("type") == "function" and choice.get("function", {}).get("name"):
                payload["tool_choice"] = {"type": "tool", "name": choice["function"]["name"]}
            else:
                invalid("unsupported_tool_choice")
            if request.parallel_tool_calls is not None:
                payload["tool_choice"]["disable_parallel_tool_use"] = not request.parallel_tool_calls
        elif request.tool_choice not in (None, "none") or request.parallel_tool_calls is not None:
            invalid("tools_required")
        schema = schema_format(request)
        if schema is not None:
            payload["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        return with_extra_body(self.tier, payload)

    async def complete(self, request: ChatRequest) -> AsyncIterator[BackendEvent]:
        payload = self.payload(request)
        headers = {"x-api-key": provider_key(self.tier, "ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01"}
        usage, reason, tools, thinking = {}, None, {}, {}
        async with self.client.stream("POST", self.tier.base_url.rstrip("/") + "/messages", json=payload, headers=headers, timeout=None) as response:
            check_status(response)
            async for raw in sse_data(response):
                item = json.loads(raw)
                kind = item.get("type")
                if kind == "error":
                    retryable = item.get("error", {}).get("type") in {"overloaded_error", "rate_limit_error", "api_error"}
                    raise BackendError("anthropic_stream_error", 503 if retryable else 502, retryable)
                if kind == "message_start":
                    usage.update(item.get("message", {}).get("usage", {}))
                elif kind == "content_block_start":
                    index, block = item["index"], item["content_block"]
                    if block["type"] == "tool_use":
                        args = json.dumps(block["input"]) if block.get("input") else ""
                        tools[index] = {"index": len(tools), "has_arguments": bool(args)}
                        yield BackendEvent({"tool_calls": [{"index": tools[index]["index"], "id": block["id"], "type": "function", "function": {"name": block["name"], "arguments": args}}]})
                    elif block["type"] == "text" and block.get("text"):
                        yield BackendEvent({"content": block["text"]})
                    elif block["type"] in {"thinking", "redacted_thinking"}:
                        thinking[index] = dict(block)
                        if block.get("thinking"):
                            yield BackendEvent({"reasoning_content": block["thinking"]})
                elif kind == "content_block_delta":
                    index, delta = item["index"], item["delta"]
                    if delta["type"] == "text_delta" and delta.get("text"):
                        yield BackendEvent({"content": delta["text"]})
                    elif delta["type"] == "input_json_delta" and index in tools and delta.get("partial_json"):
                        tools[index]["has_arguments"] = True
                        yield BackendEvent({"tool_calls": [{"index": tools[index]["index"], "function": {"arguments": delta["partial_json"]}}]})
                    elif delta["type"] == "thinking_delta":
                        thinking[index]["thinking"] = thinking[index].get("thinking", "") + delta.get("thinking", "")
                        if delta.get("thinking"):
                            yield BackendEvent({"reasoning_content": delta["thinking"]})
                    elif delta["type"] == "signature_delta":
                        thinking[index]["signature"] = thinking[index].get("signature", "") + delta.get("signature", "")
                elif kind == "content_block_stop":
                    tool = tools.get(item["index"])
                    if tool and not tool["has_arguments"]:
                        yield BackendEvent({"tool_calls": [{"index": tool["index"], "function": {"arguments": "{}"}}]})
                elif kind == "message_delta":
                    usage.update(item.get("usage", {}))  # Cumulative counts, not additive deltas.
                    stop = item.get("delta", {}).get("stop_reason")
                    if stop:
                        mapping = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "model_context_window_exceeded": "length", "tool_use": "tool_calls", "refusal": "content_filter"}
                        if stop not in mapping:
                            raise BackendError("anthropic_unsupported_stop_reason", 502, False)
                        reason = mapping[stop]
                elif kind == "message_stop":
                    if reason is None:
                        raise BackendError("upstream_missing_finish")
                    incoming = sum(usage.get(k, 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
                    outgoing = usage.get("output_tokens", 0)
                    delta = {"provider_metadata": {"anthropic": {"thinking_blocks": list(thinking.values())}}} if thinking else {}
                    normalized_usage = {"prompt_tokens": incoming, "completion_tokens": outgoing, "total_tokens": incoming + outgoing} if usage else None
                    yield BackendEvent(delta, reason, normalized_usage)
                    return
            raise BackendError("upstream_connection_dropped")


class GeminiBackend:
    def __init__(self, tier: Tier, client: httpx.AsyncClient):
        self.tier, self.client = tier, client

    def payload(self, request: ChatRequest) -> dict:
        validate_omissions(self.tier, request)
        system = system_text(request)
        if request.parallel_tool_calls is False:
            invalid("gemini_cannot_disable_parallel_tool_calls")
        turns, known_calls = [], {}
        for message in request.messages:
            if message.role in {"system", "developer"}:
                continue
            parts = []
            if message.role == "tool":
                call = known_calls.get(message.tool_call_id)
                if not call:
                    invalid("tool_result_requires_matching_call")
                try:
                    result = json.loads(message.content or "null")
                except ValueError:
                    result = message.content
                response = {"name": call["name"], "response": result if isinstance(result, dict) else {"result": result}}
                if call.get("provider_id"):
                    response["id"] = call["provider_id"]
                parts.append({"functionResponse": response})
            else:
                if message.content:
                    parts.append({"text": message.content})
                for call in message.tool_calls or []:
                    function = call.get("function", {})
                    if not call.get("id") or not function.get("name"):
                        invalid("invalid_tool_call")
                    google = call.get("extra_content", {}).get("google", {})
                    known_calls[call["id"]] = {"name": function["name"], "provider_id": google.get("id")}
                    native_call = {"name": function["name"], "args": arguments(function.get("arguments", "{}"))}
                    if google.get("id"):
                        native_call["id"] = google["id"]
                    part = {"functionCall": native_call}
                    if google.get("thought_signature"):
                        part["thoughtSignature"] = google["thought_signature"]
                    parts.append(part)
                preserved = (message.provider_metadata or {}).get("gemini", {}).get("parts")
                if message.role == "assistant" and preserved is not None:
                    allowed = {"text", "thought", "thoughtSignature", "functionCall"}
                    if not isinstance(preserved, list) or any(not isinstance(part, dict) or set(part) - allowed for part in preserved):
                        invalid("invalid_gemini_part_metadata")
                    parts = preserved  # Signed parts must survive unchanged across tool turns.
            append_turn(turns, "model" if message.role == "assistant" else "user", parts, "parts")
        if not turns:
            invalid("conversation_required")
        generation = {"maxOutputTokens": request.max_tokens or request.max_completion_tokens, "candidateCount": 1}
        for field, target in (("temperature", "temperature"), ("top_p", "topP"), ("frequency_penalty", "frequencyPenalty"), ("presence_penalty", "presencePenalty")):
            value = getattr(request, field)
            if field not in self.tier.omit_parameters and (field in {"temperature", "top_p"} or value != 0):
                generation[target] = value
        if request.seed is not None:
            generation["seed"] = request.seed
        if request.stop is not None:
            generation["stopSequences"] = [request.stop] if isinstance(request.stop, str) else request.stop
        schema = schema_format(request)
        if schema is not None:
            generation.update(responseMimeType="application/json", responseJsonSchema=schema)
        payload = {"contents": turns, "generationConfig": generation}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": text} for text in system]}
        specs = functions(request)
        if specs:
            declarations = []
            for function in specs:
                if function.get("strict"):
                    invalid("gemini_openai_strict_tools_not_supported")
                declaration = {"name": function["name"], "parametersJsonSchema": function.get("parameters", {"type": "object", "properties": {}})}
                if "description" in function:
                    declaration["description"] = function["description"]
                declarations.append(declaration)
            payload["tools"] = [{"functionDeclarations": declarations}]
            choice = request.tool_choice or "auto"
            if isinstance(choice, str) and choice in {"auto", "none", "required"}:
                function_config = {"mode": {"auto": "AUTO", "none": "NONE", "required": "ANY"}[choice]}
            elif isinstance(choice, dict) and choice.get("type") == "function" and choice.get("function", {}).get("name"):
                function_config = {"mode": "ANY", "allowedFunctionNames": [choice["function"]["name"]]}
            else:
                invalid("unsupported_tool_choice")
            payload["toolConfig"] = {"functionCallingConfig": function_config}
        elif request.tool_choice not in (None, "none") or request.parallel_tool_calls is not None:
            invalid("tools_required")
        return with_extra_body(self.tier, payload)

    async def complete(self, request: ChatRequest) -> AsyncIterator[BackendEvent]:
        payload = self.payload(request)
        model = quote(self.tier.model.removeprefix("models/"), safe="")
        url = f"{self.tier.base_url.rstrip('/')}/models/{model}:streamGenerateContent"
        headers = {"x-goog-api-key": provider_key(self.tier, "GEMINI_API_KEY")}
        reason, usage, call_count, signed = None, None, 0, False
        parts_seen = []
        async with self.client.stream("POST", url, params={"alt": "sse"}, headers=headers, json=payload, timeout=None) as response:
            check_status(response)
            async for raw in sse_data(response):
                item = json.loads(raw)
                if "error" in item:
                    code = item["error"].get("code", 500)
                    raise BackendError("gemini_stream_error", 503 if code == 429 else 502, code == 429 or code >= 500)
                if item.get("usageMetadata"):
                    counts = item["usageMetadata"]
                    incoming = counts.get("promptTokenCount", 0)
                    outgoing = counts.get("candidatesTokenCount", 0) + counts.get("thoughtsTokenCount", 0)
                    usage = {"prompt_tokens": incoming, "completion_tokens": outgoing, "total_tokens": counts.get("totalTokenCount", incoming + outgoing)}
                if item.get("promptFeedback", {}).get("blockReason"):
                    reason = "content_filter"
                for candidate in item.get("candidates", []):
                    if candidate.get("index", 0) != 0:
                        continue
                    for part in candidate.get("content", {}).get("parts", []):
                        if set(part) - {"text", "thought", "thoughtSignature", "functionCall"}:
                            raise BackendError("gemini_unsupported_content_type", 502, False)
                        parts_seen.append(part)
                        signed = signed or bool(part.get("thoughtSignature"))
                        if part.get("text"):
                            yield BackendEvent({"reasoning_content" if part.get("thought") else "content": part["text"]})
                        if "functionCall" in part:
                            function = part["functionCall"]
                            tool = {"index": call_count, "id": function.get("id") or "call_" + uuid.uuid4().hex, "type": "function", "function": {"name": function["name"], "arguments": json.dumps(function.get("args", {}))}}
                            google = {}
                            if function.get("id"):
                                google["id"] = function["id"]
                            if part.get("thoughtSignature"):
                                google["thought_signature"] = part["thoughtSignature"]
                            if google:
                                tool["extra_content"] = {"google": google}
                            call_count += 1
                            yield BackendEvent({"tool_calls": [tool]})
                    finish = candidate.get("finishReason")
                    if finish:
                        if finish == "STOP":
                            reason = "tool_calls" if call_count else "stop"
                        elif finish == "MAX_TOKENS":
                            reason = "length"
                        elif finish in {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"}:
                            reason = "content_filter"
                        else:
                            raise BackendError("gemini_invalid_completion", 502, False)
            if reason is None:
                raise BackendError("upstream_missing_finish")
            delta = {"provider_metadata": {"gemini": {"parts": parts_seen}}} if signed else {}
            yield BackendEvent(delta, reason, usage)
