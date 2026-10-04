import hashlib
import json
from dataclasses import dataclass

from gateway.config import CacheSettings
from gateway.embeddings import cosine
from gateway.models import ChatRequest


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class CacheContext:
    namespace: str
    key: str
    vector: list[float] | None = None


class ResponseCache:
    def __init__(self, state, embedder, settings: CacheSettings, policy: str):
        self.state, self.embedder, self.settings, self.policy = state, embedder, settings, policy

    def context(self, request: ChatRequest, tenant: str) -> CacheContext:
        full = request.canonical()
        scope = request.canonical()
        # A semantic namespace preserves message positions, system text and every model setting.
        if request.semantic_text() is not None:
            scope["messages"] = [
                {**message, "content": "<semantic-user>"} if message["role"] == "user" else message
                for message in scope["messages"]
            ]
        # A tier can omit implicit sampling defaults, but rejects explicitly requested
        # unsupported parameters. Keep that distinction through cache lookup as well.
        explicit_sampling = sorted(request.model_fields_set & {"temperature", "top_p", "frequency_penalty", "presence_penalty"})
        return CacheContext(digest([tenant, self.policy, scope, explicit_sampling]), digest(full))

    async def lookup(self, request: ChatRequest, tenant: str, allowed_tiers: set[str] | None = None):
        context = self.context(request, tenant)
        if not self.settings.enabled:
            return None, "disabled", context
        exact = await self.state.get(context.namespace, context.key)
        if exact and (allowed_tiers is None or exact["response"].get("model") in allowed_tiers):
            return exact["response"], "exact", context
        text = request.semantic_text()
        if self.settings.semantic_enabled and text:
            context.vector = await self.embedder.embed(text)
            entries = await self.state.candidates(
                context.namespace, self.settings.max_entries_per_namespace
            )
            best, score = None, -1.0
            for entry in entries:
                if allowed_tiers is not None and entry["response"].get("model") not in allowed_tiers:
                    continue
                if entry.get("vector"):
                    similarity = cosine(context.vector, entry["vector"])
                    if similarity > score:
                        best, score = entry, similarity
            if best and score >= self.settings.threshold:
                return best["response"], "semantic", context
        return None, "miss", context

    async def store(self, context: CacheContext, response: dict):
        if self.settings.enabled:
            await self.state.put(
                context.namespace, context.key,
                {"vector": context.vector, "response": response},
                self.settings.ttl_seconds, self.settings.max_entries_per_namespace,
            )
