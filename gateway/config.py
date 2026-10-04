from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Price(StrictModel):
    input_per_million: float = Field(default=0, ge=0)
    output_per_million: float = Field(default=0, ge=0)
    source: str = "unverified"
    verified_on: str | None = None


class ModelCapabilities(StrictModel):
    """Deployment restrictions, intersected with the adapter's implementation limits."""

    tools: bool = True
    strict_tools: bool = True
    tool_choice: bool = True
    parallel_tool_calls: bool = True
    disable_parallel_tool_calls: bool = True
    json_object: bool = True
    json_schema: bool = True
    tools_with_response_format: bool = True
    seed: bool = True
    stop: bool = True
    temperature: bool = True
    top_p: bool = True
    frequency_penalty: bool = True
    presence_penalty: bool = True
    temperature_with_top_p: bool = True
    named_messages: bool = True
    interleaved_system: bool = True
    max_temperature: float = Field(default=2, ge=0, le=2)
    max_input_tokens: int | None = Field(default=None, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)
    context_window_tokens: int | None = Field(default=None, ge=1)
    max_stop_sequences: int | None = Field(default=None, ge=1)


class Tier(StrictModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,31}$")
    provider: Literal["ollama", "openai", "openai_compatible", "anthropic", "gemini", "mock"]
    model: str
    base_url: str = ""
    base_url_env: str | None = None
    api_key_env: str | None = None
    ttft_timeout: float = Field(default=30, gt=0)
    idle_timeout: float = Field(default=20, gt=0)
    total_timeout: float = Field(default=180, gt=0)
    price: Price = Field(default_factory=Price)
    mock_delay: float = Field(default=0.001, ge=0)
    send_stream_usage: bool = True
    token_limit_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    omit_parameters: list[Literal["temperature", "top_p", "frequency_penalty", "presence_penalty"]] = Field(default_factory=list)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)

    @model_validator(mode="after")
    def provider_defaults(self):
        if self.provider == "openai_compatible":
            # A wire-compatible endpoint does not imply model feature support.
            for feature in ("tools", "strict_tools", "json_object", "json_schema", "seed"):
                if feature not in self.capabilities.model_fields_set:
                    setattr(self.capabilities, feature, False)
        if self.base_url_env and os.getenv(self.base_url_env):
            self.base_url = os.environ[self.base_url_env]
        if not self.base_url:
            defaults = {
                "openai": "https://api.openai.com/v1",
                "anthropic": "https://api.anthropic.com/v1",
                "gemini": "https://generativelanguage.googleapis.com/v1beta",
                "ollama": "http://localhost:11434",
                "mock": "http://localhost:11434",
            }
            if self.provider not in defaults:
                raise ValueError("openai_compatible tiers require a base_url")
            self.base_url = defaults[self.provider]
        return self


class CacheSettings(StrictModel):
    enabled: bool = True
    semantic_enabled: bool = False
    threshold: float = Field(default=0.98, ge=0.8, le=1)
    ttl_seconds: int = Field(default=3600, ge=1)
    max_entries_per_namespace: int = Field(default=256, ge=1, le=10000)
    embedding_url: str = "http://localhost:11434"
    embedding_url_env: str | None = None
    embedding_model: str = "all-minilm"
    embedding_timeout: float = Field(default=10, gt=0)
    policy_revision: str = "v1"

    @model_validator(mode="after")
    def embedding_endpoint(self):
        if self.embedding_url_env and os.getenv(self.embedding_url_env):
            self.embedding_url = os.environ[self.embedding_url_env]
        return self


class RouterSettings(StrictModel):
    version: Literal["v0", "v1"] = "v0"
    small_tier: str = "local"
    medium_tier: str = "small"
    large_tier: str = "large"
    short_chars: int = Field(default=600, ge=1)
    long_chars: int = Field(default=2400, ge=1)
    probability_threshold: float = Field(default=0.8, ge=0, le=1)
    artifact: str | None = None


class ReliabilitySettings(StrictModel):
    fallback_enabled: bool = True
    max_retries: int = Field(default=2, ge=0, le=2)
    fallback_order: list[str] = Field(default_factory=list)
    failure_threshold: int = Field(default=3, ge=1)
    cooldown_seconds: float = Field(default=30, gt=0)
    request_timeout: float = Field(default=240, gt=0)


class Settings(StrictModel):
    mode: Literal["production", "demo"] = "production"
    state: Literal["redis", "memory"] = "redis"
    redis_url_env: str = "REDIS_URL"
    keys_file: str = "config/keys.json"
    bucket_capacity: int = Field(default=20, ge=1)
    bucket_refill_rate: float = Field(default=2, gt=0)
    max_request_bytes: int = Field(default=1048576, ge=1024)
    max_concurrent_requests: int = Field(default=100, ge=1)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    router: RouterSettings = Field(default_factory=RouterSettings)
    reliability: ReliabilitySettings = Field(default_factory=ReliabilitySettings)
    tiers: list[Tier]

    @model_validator(mode="after")
    def valid_deployment(self):
        names = [tier.name for tier in self.tiers]
        if len(names) != len(set(names)) or not names:
            raise ValueError("tiers must have unique names")
        selected = [self.router.small_tier, self.router.medium_tier, self.router.large_tier]
        if any(name not in names for name in selected + self.reliability.fallback_order):
            raise ValueError("router and fallback tiers must exist")
        if self.mode == "production" and (
            self.state != "redis" or any(t.provider == "mock" for t in self.tiers)
        ):
            raise ValueError("production requires Redis and real providers")
        if self.router.version == "v1" and not self.router.artifact:
            raise ValueError("v1 requires a trained router artifact")
        if self.cache.semantic_enabled and any("temperature" in tier.omit_parameters for tier in self.tiers):
            raise ValueError("disable semantic caching when a tier omits temperature control")
        return self

    def policy_hash(self) -> str:
        policy = self.model_dump(include={"tiers", "router", "cache", "reliability"})
        if self.router.artifact:
            policy["artifact_hash"] = hashlib.sha256(Path(self.router.artifact).read_bytes()).hexdigest()
        return hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()[:20]


def load_settings(path: str | None = None) -> Settings:
    path = path or os.getenv("GATEWAY_CONFIG", "config/gateway.yaml")
    return Settings.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
