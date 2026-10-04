from typing import Any, Literal

from pydantic import Field, model_validator

from gateway.config import StrictModel


class Message(StrictModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    reasoning_content: str | None = None
    provider_metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def meaningful(self):
        if self.content is None and not self.tool_calls:
            raise ValueError("message needs content or tool_calls")
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
        return self


class StreamOptions(StrictModel):
    include_usage: bool = False


class RoutingOptions(StrictModel):
    mode: Literal["prefer", "strict"] = "prefer"
    fallbacks: list[str] | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def valid_policy(self):
        if self.fallbacks is not None and len(self.fallbacks) != len(set(self.fallbacks)):
            raise ValueError("fallback aliases must be unique")
        if self.mode == "strict" and self.fallbacks:
            raise ValueError("strict routing cannot include fallback aliases")
        return self


class ChatRequest(StrictModel):
    model: str = "auto"
    messages: list[Message] = Field(min_length=1, max_length=256)
    stream: bool = False
    stream_options: StreamOptions | None = None
    temperature: float = Field(default=0, ge=0, le=2)
    top_p: float = Field(default=1, ge=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1, le=32768)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=32768)
    stop: str | list[str] | None = None
    seed: int | None = None
    frequency_penalty: float = Field(default=0, ge=-2, le=2)
    presence_penalty: float = Field(default=0, ge=-2, le=2)
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    response_format: dict[str, Any] | None = None
    user: str | None = None
    n: Literal[1] = 1
    routing: RoutingOptions = Field(default_factory=RoutingOptions)

    @model_validator(mode="after")
    def token_budget(self):
        if self.model == "auto" and self.routing.mode == "strict":
            raise ValueError("strict routing requires an explicit model alias")
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            raise ValueError("use max_tokens or max_completion_tokens, not both")
        if self.max_tokens is None and self.max_completion_tokens is None:
            self.max_tokens = 256
        return self

    def canonical(self) -> dict:
        return self.model_dump(exclude={"stream", "stream_options"}, exclude_none=True)

    def upstream_payload(self) -> dict:
        return self.model_dump(exclude={"stream", "stream_options", "routing"}, exclude_none=True)

    def prompt_text(self) -> str:
        return "\n".join(message.content or "" for message in self.messages)

    def semantic_text(self) -> str | None:
        users = [m for m in self.messages if m.role == "user"]
        if (
            self.temperature != 0
            or self.tools
            or self.tool_choice
            or self.response_format
            or len(users) != 1
            or any(m.role not in {"system", "developer", "user"} for m in self.messages)
        ):
            return None
        return users[0].content or None
