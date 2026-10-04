import json
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

from gateway.capabilities import CapabilityRegistry, Eligibility
from gateway.config import RouterSettings
from gateway.models import ChatRequest


@dataclass
class Decision:
    tier: str
    reason: str
    probability: float | None = None
    preferred_tier: str | None = None


def baseline(text: str, config: RouterSettings) -> Decision:
    lowered = text.lower()
    hard = any(word in lowered for word in ("prove", "reason", "derive", "debug", "algorithm"))
    if hard or len(text) > config.long_chars:
        return Decision(config.large_tier, "complexity_rule")
    if len(text) <= config.short_chars:
        return Decision(config.small_tier, "short_prompt")
    return Decision(config.medium_tier, "medium_prompt")


class Router:
    def __init__(self, config: RouterSettings, embedder, capabilities: CapabilityRegistry):
        self.config, self.embedder, self.capabilities = config, embedder, capabilities
        self.artifact = None
        if config.version == "v1":
            self.artifact = json.loads(Path(config.artifact).read_text(encoding="utf-8"))
            if self.artifact["embedding_model"] != embedder.model:
                raise ValueError("classifier embedding model differs from runtime model")
            if self.artifact.get("schema_version") != 1:
                raise ValueError("unsupported router artifact")
            vectors = [np.asarray(self.artifact[field], dtype=float) for field in ("mean", "scale", "coef")]
            if (
                any(vector.ndim != 1 or not vector.size or not np.isfinite(vector).all() for vector in vectors)
                or len({vector.shape for vector in vectors}) != 1
                or np.any(vectors[1] <= 0)
                or not np.isfinite(self.artifact["intercept"])
            ):
                raise ValueError("invalid classifier coefficients")

    async def choose(self, request: ChatRequest, report: Eligibility | None = None) -> Decision:
        report = report or self.capabilities.preflight(request)
        if request.model != "auto":
            decision = Decision(request.model, "explicit_model")
        elif report.needs.tools or report.needs.format != "text":
            decision = Decision(self.config.large_tier, "structured_request")
        elif self.artifact:
            try:
                text = request.prompt_text()
                vector = await self.embedder.embed(text)
                probability = predict(self.artifact, vector, len(text))
                tier = self.config.small_tier if probability >= self.config.probability_threshold else self.config.large_tier
                decision = Decision(tier, "classifier", probability)
            except (httpx.HTTPError, ValueError):
                decision = Decision(self.config.large_tier, "classifier_unavailable")
        else:
            decision = baseline(request.prompt_text(), self.config)
        decision.preferred_tier = decision.tier
        if decision.tier not in report.eligible:
            order = [*self.capabilities.reliability.fallback_order, self.config.large_tier, self.config.medium_tier, self.config.small_tier]
            decision.tier = next(name for name in order if name in report.eligible)
            decision.reason = "capability_reroute"
        return decision


def predict(artifact: dict, vector: list[float], length: int) -> float:
    features = np.asarray([*vector, np.log1p(length)])
    mean, scale = np.asarray(artifact["mean"]), np.asarray(artifact["scale"])
    if features.shape != mean.shape or features.shape != scale.shape:
        raise ValueError("classifier embedding dimensions do not match")
    z = np.dot((features - mean) / scale, artifact["coef"]) + artifact["intercept"]
    return float(1 / (1 + np.exp(-np.clip(z, -700, 700))))
