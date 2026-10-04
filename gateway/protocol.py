import json

from gateway.backends import BackendEvent


def sse(payload: dict | str) -> bytes:
    return ("data: " + (payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))) + "\n\n").encode()


def chunk(request_id: str, created: int, model: str, delta: dict | None = None, reason=None, usage=None):
    value = {
        "id": request_id, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": reason}] if usage is None else [],
    }
    if usage is not None:
        value["usage"] = usage
    return value


class Accumulator:
    def __init__(self):
        self.content = ""
        self.reasoning = ""
        self.tools: dict[int, dict] = {}
        self.reason = "stop"
        self.finished = False
        self.usage = None
        self.provider_metadata = {}

    def add(self, event: BackendEvent):
        self.content += event.delta.get("content", "")
        self.reasoning += event.delta.get("reasoning_content", "")
        if event.delta.get("provider_metadata"):
            self.provider_metadata.update(event.delta["provider_metadata"])
        for call in event.delta.get("tool_calls", []):
            target = self.tools.setdefault(call["index"], {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
            if call.get("id"):
                target["id"] = call["id"]
            if call.get("extra_content"):
                target.setdefault("extra_content", {}).update(call["extra_content"])
            for field in ("name", "arguments"):
                target["function"][field] += call.get("function", {}).get(field, "")
        if event.finish_reason:
            self.reason, self.finished = event.finish_reason, True
        if event.usage:
            self.usage = event.usage

    def message(self):
        value = {"role": "assistant", "content": self.content or (None if self.tools else "")}
        if self.reasoning:
            value["reasoning_content"] = self.reasoning
        if self.provider_metadata:
            value["provider_metadata"] = self.provider_metadata
        if self.tools:
            value["tool_calls"] = [self.tools[index] for index in sorted(self.tools)]
        return value

    def effective_usage(self, prompt: str):
        if self.usage:
            return self.usage, "provider"
        incoming = max(1, (len(prompt) + 3) // 4)
        outgoing = (len(self.content) + len(self.reasoning) + len(json.dumps(self.tools) if self.tools else "") + 3) // 4
        return {"prompt_tokens": incoming, "completion_tokens": outgoing, "total_tokens": incoming + outgoing}, "estimated"

    def response(self, request_id: str, created: int, model: str, prompt: str):
        return {
            "id": request_id, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": self.message(), "finish_reason": self.reason}],
            "usage": self.effective_usage(prompt)[0],
        }
