from typing import List
import hashlib
import math
import re
import httpx

from backend.cache import cache, text_hash
from backend.config import settings


class EmbeddingProvider:
    """OpenAI-compatible embedding client."""

    def __init__(self, provider: str = None):
        self.provider = provider or settings.embedding_provider
        self.config = settings.get_embedding_config()

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if self.provider and self.provider.lower() in {"local", "local_hash", "hash"}:
            return [self._local_hash_embedding(text) for text in texts]

        keys = [
            await cache.make_key(
                "embedding",
                {
                    "kind": "embedding",
                    "provider": self.provider,
                    "base_url": self.config.get("base_url", ""),
                    "model": self.config.get("model", ""),
                    "dim": settings.embedding_dim,
                    "text_hash": text_hash(text),
                },
            )
            for text in texts
        ]

        results: List[List[float]] = [None] * len(texts)
        misses = []

        for index, key in enumerate(keys):
            cached = await cache.get_json(key)
            if cached is None:
                misses.append((index, key, texts[index]))
            else:
                results[index] = cached

        if misses:
            vectors = await self._request_embeddings([text for _, _, text in misses])
            for (index, key, _), vector in zip(misses, vectors):
                results[index] = vector
                await cache.set_json(
                    key,
                    vector,
                    settings.cache_embedding_ttl_seconds,
                )

        return results

    async def _request_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        base_url = self.config.get("base_url", "")
        model = self.config.get("model", "")

        if not base_url:
            raise RuntimeError(
                "Embedding base URL is not configured. Set EMBEDDING_BASE_URL "
                "or configure the provider named by EMBEDDING_PROVIDER."
            )
        if not model:
            raise RuntimeError("Embedding model is not configured. Set EMBEDDING_MODEL.")

        url = base_url if "/embeddings" in base_url else f"{base_url.rstrip('/')}/embeddings"
        headers = {
            "Content-Type": "application/json",
            **self.config.get("headers", {}),
        }
        if self.config.get("api_key"):
            headers["Authorization"] = f"Bearer {self.config['api_key']}"

        payload = {"model": model, "input": texts}

        try:
            async with httpx.AsyncClient(timeout=settings.embedding_timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Embedding API error {response.status_code}: {response.text}"
                    )
                data = response.json()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Embedding API request failed: {exc}") from exc

        embeddings = data.get("data", [])
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Embedding API returned {len(embeddings)} vectors for {len(texts)} texts."
            )

        embeddings = sorted(embeddings, key=lambda item: item.get("index", 0))
        vectors = [item.get("embedding") for item in embeddings]
        if any(not vector for vector in vectors):
            raise RuntimeError("Embedding API returned an empty embedding vector.")

        return vectors

    def _local_hash_embedding(self, text: str) -> List[float]:
        """Deterministic dense vector for local tests and CI.

        This is not a replacement for semantic embeddings in production. It lets
        the Milvus dense+sparse pipeline run without external API keys.
        """
        dim = settings.embedding_dim
        vector = [0.0] * dim
        tokens = re.findall(r"[a-z0-9]+", text.lower())

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dim
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if not norm:
            return vector
        return [value / norm for value in vector]
