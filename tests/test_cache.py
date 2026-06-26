import asyncio
import tempfile
import unittest
from pathlib import Path

from backend.cache import cache, normalize_file_paths, stable_hash
from backend.config import settings
from backend.embedding_provider import EmbeddingProvider
from backend.knowledge_base import KnowledgeBase
from backend.milvus_hybrid import HybridHit, MilvusHybridRetriever


CACHE_SETTINGS = [
    "cache_enabled",
    "cache_backend",
    "cache_redis_url",
    "cache_key_prefix",
    "cache_default_ttl_seconds",
    "cache_summary_ttl_seconds",
    "cache_file_ttl_seconds",
    "cache_embedding_ttl_seconds",
    "cache_hybrid_search_ttl_seconds",
    "cache_max_memory_items",
    "cache_lock_ttl_seconds",
    "cache_lock_wait_seconds",
]


class CacheTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._old_settings = {name: getattr(settings, name) for name in CACHE_SETTINGS}
        settings.cache_enabled = True
        settings.cache_backend = "memory"
        settings.cache_redis_url = "redis://cache-test/0"
        settings.cache_key_prefix = "deep-rag-test"
        settings.cache_default_ttl_seconds = 300
        settings.cache_summary_ttl_seconds = 300
        settings.cache_file_ttl_seconds = 300
        settings.cache_embedding_ttl_seconds = 300
        settings.cache_hybrid_search_ttl_seconds = 300
        settings.cache_max_memory_items = 2048
        settings.cache_lock_ttl_seconds = 1
        settings.cache_lock_wait_seconds = 0.05
        cache.reset_for_tests()

    async def asyncTearDown(self):
        for name, value in self._old_settings.items():
            setattr(settings, name, value)
        cache.reset_for_tests()


class CacheKeyHelpersTest(CacheTestCase):
    async def test_normalized_file_paths_make_stable_keys(self):
        payload_a = {"paths": normalize_file_paths(["/Products/A.md", "Reports\\"])}
        payload_b = {"paths": normalize_file_paths(["Reports/", "Products/A.md"])}

        self.assertEqual(payload_a, payload_b)
        self.assertEqual(stable_hash(payload_a), stable_hash(payload_b))

    async def test_cache_key_changes_with_namespace_version(self):
        key_a = await cache.make_key("summary", {"kind": "demo", "value": 1})
        await cache.bump_namespace("summary")
        key_b = await cache.make_key("summary", {"kind": "demo", "value": 1})

        self.assertNotEqual(key_a, key_b)


class MemoryCacheBackendTest(CacheTestCase):
    async def test_memory_cache_ttl_expires_values(self):
        await cache.set_json("ttl-key", {"value": 1}, ttl_seconds=0.01)
        self.assertEqual(await cache.get_json("ttl-key"), {"value": 1})

        await asyncio.sleep(0.03)

        self.assertIsNone(await cache.get_json("ttl-key"))

    async def test_memory_cache_evicts_lru_items(self):
        settings.cache_max_memory_items = 2
        await cache.set_json("one", 1, ttl_seconds=60)
        await cache.set_json("two", 2, ttl_seconds=60)
        self.assertEqual(await cache.get_json("one"), 1)

        await cache.set_json("three", 3, ttl_seconds=60)

        self.assertEqual(await cache.get_json("one"), 1)
        self.assertIsNone(await cache.get_json("two"))
        self.assertEqual(await cache.get_json("three"), 3)

    async def test_get_or_set_uses_single_flight_for_memory_backend(self):
        calls = 0

        async def factory():
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.01)
            return {"ok": True}

        results = await asyncio.gather(
            cache.get_or_set_json("single-flight", 60, factory),
            cache.get_or_set_json("single-flight", 60, factory),
            cache.get_or_set_json("single-flight", 60, factory),
        )

        self.assertEqual(results, [{"ok": True}, {"ok": True}, {"ok": True}])
        self.assertEqual(calls, 1)


class BrokenRedis:
    async def get(self, *args, **kwargs):
        raise RuntimeError("redis unavailable")

    async def set(self, *args, **kwargs):
        raise RuntimeError("redis unavailable")

    async def incr(self, *args, **kwargs):
        raise RuntimeError("redis unavailable")


class RedisDegradationTest(CacheTestCase):
    async def test_redis_errors_bypass_cache(self):
        settings.cache_backend = "redis"
        cache._redis = BrokenRedis()
        cache._redis_url = settings.cache_redis_url

        async def factory():
            return {"fallback": True}

        value = await cache.get_or_set_json("redis-fallback", 60, factory)

        self.assertEqual(value, {"fallback": True})


class CountingEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        self.provider = "remote-test"
        self.config = {
            "base_url": "http://embedding.test/v1",
            "model": "test-embedding",
            "headers": {},
        }
        self.calls = []

    async def _request_embeddings(self, texts):
        self.calls.append(list(texts))
        return [[float(len(text)), 1.0] for text in texts]


class EmbeddingCacheTest(CacheTestCase):
    async def test_embedding_cache_supports_full_and_partial_hits(self):
        provider = CountingEmbeddingProvider()

        first = await provider.embed_texts(["alpha", "beta"])
        second = await provider.embed_texts(["alpha", "beta"])
        third = await provider.embed_texts(["alpha", "gamma"])

        self.assertEqual(first, [[5.0, 1.0], [4.0, 1.0]])
        self.assertEqual(second, first)
        self.assertEqual(third, [[5.0, 1.0], [5.0, 1.0]])
        self.assertEqual(provider.calls, [["alpha", "beta"], ["gamma"]])


class CountingRetriever(MilvusHybridRetriever):
    def __init__(self):
        self.client = None
        self.embedder = EmbeddingProvider(provider="local_hash")
        self.collection_name = "cache_test_collection"
        self.calls = 0

    async def _search(self, query, file_paths=None, top_k=None, modes=("dense", "sparse")):
        self.calls += 1
        return [
            HybridHit(
                rank=1,
                path="Products/A.md",
                source_path="Products/A.md",
                score=0.5,
                chunk_id=str(self.calls),
            )
        ]


class MilvusSearchCacheTest(CacheTestCase):
    async def test_hybrid_search_cache_normalizes_paths_and_invalidates(self):
        retriever = CountingRetriever()

        first = await retriever.search("query", file_paths=["Reports/", "Products/A.md"], top_k=5)
        second = await retriever.search("query", file_paths=["Products/A.md", "Reports/"], top_k=5)
        third = await retriever.search("query", file_paths=["Products/A.md", "Reports/"], top_k=6)
        await cache.bump_namespace("hybrid_search")
        fourth = await retriever.search("query", file_paths=["Reports/", "Products/A.md"], top_k=5)

        self.assertEqual(retriever.calls, 3)
        self.assertEqual(first[0].chunk_id, "1")
        self.assertEqual(second[0].chunk_id, "1")
        self.assertEqual(third[0].chunk_id, "2")
        self.assertEqual(fourth[0].chunk_id, "3")


class KnowledgeBaseFileCacheTest(CacheTestCase):
    async def test_file_cache_key_changes_when_file_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "doc.md"
            file_path.write_text("first", encoding="utf-8")
            kb = KnowledgeBase(base_path=tmp)

            first = await kb._read_file(file_path)
            second = await kb._read_file(file_path)

            file_path.write_text("updated content", encoding="utf-8")
            third = await kb._read_file(file_path)

        self.assertEqual(first, "first")
        self.assertEqual(second, "first")
        self.assertEqual(third, "updated content")


if __name__ == "__main__":
    unittest.main()
