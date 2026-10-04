"""Pure request preflight and deployment eligibility; never performs provider I/O."""

import json
from dataclasses import dataclass

from gateway.config import ReliabilitySettings, RouterSettings, Tier
from gateway.models import ChatRequest


class SelectionError(Exception):
    def __init__(self, code: str, message: str, *, details=None, status=400):
        super().__init__(message)
        self.code, self.message, self.details, self.status = code, message, details, status


def invalid_request(message: str):
    raise SelectionError("invalid_request", message)


def estimate_input_tokens(request: ChatRequest) -> int:
    """Conservative UTF-8-byte proxy plus overhead; NOT a provider tokenizer count."""
    payload = {"messages": [m.model_dump(exclude_none=True) for m in request.messages]}
    for field in ("tools", "response_format"):
        if getattr(request, field):
            payload[field] = getattr(request, field)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return len(encoded) + 16 * len(request.messages) + 64


@dataclass(frozen=True)
class Requirements:
    format: str
    tools: bool
    strict_tools: bool
    input_tokens: int
    output_tokens: int


def requirements(request: ChatRequest) -> Requirements:
    format_value = request.response_format
    format_name = format_value.get("type") if format_value is not None else "text"
    if not isinstance(format_name, str) or format_name not in {"text", "json_object", "json_schema"}:
        invalid_request("Unsupported response_format.type")
    if format_name == "json_schema":
        definition = format_value.get("json_schema")
        if not isinstance(definition, dict) or not isinstance(definition.get("schema"), dict):
            invalid_request("response_format.json_schema.schema must be an object")
        if "strict" in definition and not isinstance(definition["strict"], bool):
            invalid_request("response_format.json_schema.strict must be boolean")
    names, strict = set(), False
    for tool in request.tools or []:
        function = tool.get("function")
        if tool.get("type") != "function" or not isinstance(function, dict):
            invalid_request("Only function tool definitions are supported")
        name = function.get("name")
        if not isinstance(name, str) or not name or name in names:
            invalid_request("Function tools require unique, nonempty names")
        names.add(name)
        if "parameters" in function and not isinstance(function["parameters"], dict):
            invalid_request("Tool parameters must be a JSON schema object")
        if "strict" in function and not isinstance(function["strict"], bool):
            invalid_request("Tool strict must be boolean")
        strict = strict or function.get("strict", False)
    choice = request.tool_choice
    if isinstance(choice, dict):
        function = choice.get("function")
        if choice.get("type") != "function" or not isinstance(function, dict) or not isinstance(function.get("name"), str) or function["name"] not in names:
            invalid_request("Named tool_choice must reference a declared function")
    elif choice not in (None, "auto", "none", "required"):
        invalid_request("Unsupported tool_choice")
    if not names and (choice not in (None, "none") or request.parallel_tool_calls is not None):
        invalid_request("Tool selection controls require tool definitions")
    calls = set()
    for message in request.messages:
        if message.tool_calls and message.role != "assistant":
            invalid_request("Only assistant messages may contain tool_calls")
        for call in message.tool_calls or []:
            function = call.get("function")
            if not isinstance(call.get("id"), str) or not call["id"] or not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"]:
                invalid_request("Invalid assistant tool call")
            args = function.get("arguments", "{}")
            try:
                args = json.loads(args) if isinstance(args, str) else args
            except ValueError:
                invalid_request("Tool arguments must be valid JSON")
            if not isinstance(args, dict):
                invalid_request("Tool arguments must be a JSON object")
            calls.add(call["id"])
        if message.role == "tool" and message.tool_call_id not in calls:
            invalid_request("Tool results must reference an earlier assistant tool call")
    if isinstance(request.stop, list) and (not request.stop or any(not value for value in request.stop)):
        invalid_request("Stop sequences must be nonempty strings")
    return Requirements(
        format_name, bool(names or calls or request.tool_choice or request.parallel_tool_calls is not None),
        strict, estimate_input_tokens(request), request.max_tokens or request.max_completion_tokens,
    )


def effective_capabilities(tier: Tier) -> dict:
    result = tier.capabilities.model_dump()
    disabled = set(tier.omit_parameters)
    if tier.provider == "ollama":
        disabled |= {"tools", "strict_tools", "tool_choice", "parallel_tool_calls", "disable_parallel_tool_calls", "frequency_penalty", "presence_penalty", "named_messages"}
    elif tier.provider == "anthropic":
        disabled |= {"json_object", "seed", "frequency_penalty", "presence_penalty", "named_messages", "interleaved_system"}
        result["max_temperature"] = min(result["max_temperature"], 1)
    elif tier.provider == "gemini":
        disabled |= {"strict_tools", "disable_parallel_tool_calls", "named_messages", "interleaved_system"}
    for name in disabled:
        result[name] = False
    if not result["tools"]:
        for name in ("strict_tools", "tool_choice", "parallel_tool_calls", "disable_parallel_tool_calls"):
            result[name] = False
    result["input_token_estimator"] = "utf8_bytes_plus_overhead"
    return result


def incompatibilities(tier: Tier, request: ChatRequest, needs: Requirements) -> list[str]:
    caps = effective_capabilities(tier)
    reasons = []
    required_flags = {
        "tools": needs.tools,
        "strict_tools": needs.strict_tools,
        "tool_choice": request.tool_choice is not None,
        "parallel_tool_calls": request.parallel_tool_calls is True,
        "disable_parallel_tool_calls": request.parallel_tool_calls is False,
        "json_object": needs.format == "json_object",
        "json_schema": needs.format == "json_schema",
        "tools_with_response_format": needs.tools and needs.format != "text",
        "seed": request.seed is not None,
        "stop": request.stop is not None,
    }
    for flag, needed in required_flags.items():
        if needed and not caps[flag]:
            reasons.append("unsupported_" + flag)
    for field, default in (("temperature", 0), ("top_p", 1), ("frequency_penalty", 0), ("presence_penalty", 0)):
        if field in tier.omit_parameters:
            if field in request.model_fields_set:
                reasons.append("unsupported_" + field)
        elif not caps[field] and (field in request.model_fields_set or getattr(request, field) != default or field in {"temperature", "top_p"}):
            # A non-neutral default cannot be silently dropped. Configure omit_parameters
            # explicitly if a deployment must use the provider's own sampling defaults.
            reasons.append("unsupported_" + field)
    if "temperature" not in tier.omit_parameters and request.temperature > caps["max_temperature"]:
        reasons.append("temperature_out_of_range")
    if not caps["temperature_with_top_p"] and "temperature" not in tier.omit_parameters and "top_p" not in tier.omit_parameters:
        if tier.provider != "anthropic" or request.top_p != 1:
            reasons.append("unsupported_temperature_top_p_combination")
    conversation_started = False
    for message in request.messages:
        if message.role in {"system", "developer"}:
            if conversation_started and not caps["interleaved_system"]:
                reasons.append("unsupported_interleaved_system")
        else:
            conversation_started = True
        if message.name and message.role != "tool" and not caps["named_messages"]:
            reasons.append("unsupported_named_messages")
    if caps["max_input_tokens"] is not None and needs.input_tokens > caps["max_input_tokens"]:
        reasons.append("input_token_estimate_exceeded")
    if caps["max_output_tokens"] is not None and needs.output_tokens > caps["max_output_tokens"]:
        reasons.append("output_token_limit_exceeded")
    if caps["context_window_tokens"] is not None and needs.input_tokens + needs.output_tokens > caps["context_window_tokens"]:
        reasons.append("context_window_estimate_exceeded")
    stops = len(request.stop) if isinstance(request.stop, list) else int(request.stop is not None)
    if caps["max_stop_sequences"] is not None and stops > caps["max_stop_sequences"]:
        reasons.append("stop_sequence_limit_exceeded")
    return list(dict.fromkeys(reasons))


@dataclass(frozen=True)
class Eligibility:
    eligible: tuple[str, ...]
    rejected: dict[str, list[str]]
    needs: Requirements


class CapabilityRegistry:
    def __init__(self, tiers: list[Tier], reliability: ReliabilitySettings, router: RouterSettings | None = None):
        self.tiers = {tier.name: tier for tier in tiers}
        self.reliability, self.router = reliability, router

    def validate_aliases(self, request: ChatRequest):
        if request.model != "auto" and request.model not in self.tiers:
            raise SelectionError("model_not_found", "Unknown model alias; see /v1/models", status=404)
        if any(name not in self.tiers for name in request.routing.fallbacks or []):
            raise SelectionError("invalid_routing_policy", "Fallback policy contains an unknown model alias")

    def allowed(self, request: ChatRequest, name: str) -> bool:
        if request.routing.mode == "strict":
            return name == request.model
        if request.routing.fallbacks is None:
            return True
        return name == request.model or name in request.routing.fallbacks

    def retry_order(self, request: ChatRequest, preferred: str) -> list[str]:
        order = [preferred]
        if request.routing.mode != "strict" and self.reliability.fallback_enabled and self.reliability.max_retries > 0:
            order += self.reliability.fallback_order
        return [name for name in dict.fromkeys(order) if self.allowed(request, name)]

    def initial_order(self, request: ChatRequest) -> list[str]:
        if request.model != "auto":
            return self.retry_order(request, request.model)
        if self.router:
            order = [self.router.small_tier, self.router.medium_tier, self.router.large_tier, *self.reliability.fallback_order]
        else:
            order = list(self.tiers)
        return [name for name in dict.fromkeys(order) if self.allowed(request, name)]

    def evaluate(self, request: ChatRequest, order: list[str]) -> Eligibility:
        self.validate_aliases(request)
        needs = requirements(request)
        eligible, rejected = [], {}
        for name in order:
            if name not in self.tiers or not self.allowed(request, name):
                continue
            reasons = incompatibilities(self.tiers[name], request, needs)
            if reasons:
                rejected[name] = reasons
            else:
                eligible.append(name)
        return Eligibility(tuple(eligible), rejected, needs)

    @staticmethod
    def require_eligible(report: Eligibility):
        if not report.eligible:
            raise SelectionError(
                "no_compatible_tier", "No configured tier supports this request under its routing policy",
                details={"incompatible_tiers": report.rejected},
            )

    def preflight(self, request: ChatRequest) -> Eligibility:
        self.validate_aliases(request)
        report = self.evaluate(request, self.initial_order(request))
        self.require_eligible(report)
        return report
