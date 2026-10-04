import hashlib
import re

import httpx
import numpy as np


class OllamaEmbedder:
    def __init__(self, client: httpx.AsyncClient, url: str, model: str, timeout: float = 10):
        self.client, self.url, self.model, self.timeout = client, url.rstrip("/"), model, timeout

    async def embed(self, text: str) -> list[float]:
        response = await self.client.post(
            f"{self.url}/api/embed", json={"model": self.model, "input": text},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return normalize(response.json()["embeddings"][0])


class DemoEmbedder:
    """Stable bag-of-words hashing for smoke tests, NOT a semantic embedding model."""

    model = "demo-hash-64"

    async def embed(self, text: str) -> list[float]:
        vector = np.zeros(64)
        for word in re.findall(r"\w+", text.lower()):
            vector[int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % 64] += 1
        return normalize(vector)


def normalize(vector) -> list[float]:
    array = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(array)
    if array.ndim != 1 or not np.isfinite(array).all() or not norm:
        raise ValueError("embedding must be a finite, nonzero vector")
    return (array / norm).tolist()


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return -1
    return float(np.dot(normalize(left), normalize(right)))
