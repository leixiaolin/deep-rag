import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from backend import config as config_module


logger = logging.getLogger(__name__)

CACHE_NAMESPACES = ("summary", "file", "file_tree", "embedding", "hybrid_search")


def _settings():
    return config_module.settings


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_file_paths(file_paths: Optional[List[str]]) -> List[str]:
    normalized = []

    for raw_path in file_paths or []:
        path = (raw_path or "").strip().replace("\\", "/")
        while "//" in path:
            path = path.replace("//", "/")

        if path in ("", ".", "/"):
            return []

        path = path.lstrip("/")
        if path in ("", "."):
            return []

        if path.endswith("/"):
            path = path.rstrip("/") + "/"

        normalized.append(path)

    return sorted(set(normalized))


@dataclass
class _MemoryEntry:
    value: Any
    expires_at: Optional[float]


class CacheManager:
    def __init__(self):
        self._memory: OrderedDict[str, _MemoryEntry] = OrderedDict()
        self._memory_versions: Dict[str, int] = defaultdict(int)
        self._locks: Dict[str, asyncio.Lock] = {}
        self._redis = None
        self._redis_url = None
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._errors = 0

    async def make_key(self, namespace: str, payload: Dict[str, Any]) -> str:
        self._validate_namespace(namespace)
        cfg = _settings()
        version = await self.namespace_version(namespace)
        digest = stable_hash(payload)
        return f"{cfg.cache_key_prefix}:v1:{namespace}:{version}:{digest}"

    async def get_json(self, key: str) -> Any:
        if not self._enabled:
            return None

        if self._backend == "redis":
            value = await self._redis_get_json(key)
        else:
            value = self._memory_get(key)

        if value is None:
            self._misses += 1
        else:
            self._hits += 1
        return value

    async def set_json(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        if not self._enabled:
            return

        try:
            stable_json_dumps(value)
        except Exception as exc:
            self._record_error("cache value is not JSON serializable", exc)
            return

        if self._backend == "redis":
            await self._redis_set_json(key, value, ttl_seconds)
        else:
            self._memory_set(key, value, ttl_seconds)

    async def get_or_set_json(
        self,
        key: str,
        ttl_seconds: Optional[float],
        factory: Callable[[], Awaitable[Any]],
    ) -> Any:
        cached = await self.get_json(key)
        if cached is not None:
            return cached

        if not self._enabled:
            return await factory()

        if self._backend == "redis":
            return await self._redis_get_or_set_json(key, ttl_seconds, factory)

        return await self._memory_get_or_set_json(key, ttl_seconds, factory)

    async def bump_namespace(self, namespace: Optional[str] = None) -> None:
        names = CACHE_NAMESPACES if namespace in (None, "all") else (namespace,)
        for name in names:
            self._validate_namespace(name)
            self._memory_versions[name] += 1

        if not self._enabled or self._backend != "redis":
            return

        client = await self._redis_client()
        if client is None:
            return

        for name in names:
            try:
                await client.incr(self._namespace_version_key(name))
            except Exception as exc:
                self._record_error(f"failed to bump cache namespace {name}", exc)

    async def namespace_version(self, namespace: str) -> int:
        self._validate_namespace(namespace)
        if not self._enabled or self._backend != "redis":
            return self._memory_versions[namespace]

        client = await self._redis_client()
        if client is None:
            return self._memory_versions[namespace]

        try:
            value = await client.get(self._namespace_version_key(namespace))
            return int(value or 0)
        except Exception as exc:
            self._record_error(f"failed to read cache namespace {namespace}", exc)
            return self._memory_versions[namespace]

    async def stats(self) -> Dict[str, Any]:
        namespaces = {
            namespace: await self.namespace_version(namespace)
            for namespace in CACHE_NAMESPACES
        }
        return {
            "enabled": self._enabled,
            "backend": self._backend,
            "hits": self._hits,
            "misses": self._misses,
            "sets": self._sets,
            "errors": self._errors,
            "namespaces": namespaces,
        }

    def reset_for_tests(self) -> None:
        self._memory.clear()
        self._memory_versions.clear()
        self._locks.clear()
        self._redis = None
        self._redis_url = None
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._errors = 0

    async def _memory_get_or_set_json(
        self,
        key: str,
        ttl_seconds: Optional[float],
        factory: Callable[[], Awaitable[Any]],
    ) -> Any:
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached = await self.get_json(key)
            if cached is not None:
                return cached

            value = await factory()
            await self.set_json(key, value, ttl_seconds)
            return value

    async def _redis_get_or_set_json(
        self,
        key: str,
        ttl_seconds: Optional[float],
        factory: Callable[[], Awaitable[Any]],
    ) -> Any:
        client = await self._redis_client()
        if client is None:
            return await factory()

        cfg = _settings()
        lock_key = f"{key}:lock"
        token = uuid.uuid4().hex
        lock_ttl = max(1, int(cfg.cache_lock_ttl_seconds))

        try:
            acquired = await client.set(lock_key, token, nx=True, ex=lock_ttl)
        except Exception as exc:
            self._record_error("failed to acquire redis cache lock", exc)
            return await factory()

        if acquired:
            try:
                cached = await self.get_json(key)
                if cached is not None:
                    return cached

                value = await factory()
                await self.set_json(key, value, ttl_seconds)
                return value
            finally:
                await self._release_redis_lock(client, lock_key, token)

        deadline = time.monotonic() + max(0.0, float(cfg.cache_lock_wait_seconds))
        while time.monotonic() < deadline:
            await asyncio.sleep(0.05)
            cached = await self.get_json(key)
            if cached is not None:
                return cached

        value = await factory()
        await self.set_json(key, value, ttl_seconds)
        return value

    async def _release_redis_lock(self, client: Any, lock_key: str, token: str) -> None:
        try:
            current = await client.get(lock_key)
            if current == token:
                await client.delete(lock_key)
        except Exception as exc:
            self._record_error("failed to release redis cache lock", exc)

    def _memory_get(self, key: str) -> Any:
        entry = self._memory.get(key)
        if entry is None:
            return None

        if entry.expires_at is not None and entry.expires_at <= time.monotonic():
            self._memory.pop(key, None)
            return None

        self._memory.move_to_end(key)
        return entry.value

    def _memory_set(self, key: str, value: Any, ttl_seconds: Optional[float]) -> None:
        ttl_seconds = self._effective_ttl(ttl_seconds)
        expires_at = None
        if ttl_seconds and ttl_seconds > 0:
            expires_at = time.monotonic() + float(ttl_seconds)

        self._memory[key] = _MemoryEntry(value=value, expires_at=expires_at)
        self._memory.move_to_end(key)
        self._sets += 1

        max_items = max(0, int(_settings().cache_max_memory_items))
        while len(self._memory) > max_items:
            self._memory.popitem(last=False)

    async def _redis_get_json(self, key: str) -> Any:
        client = await self._redis_client()
        if client is None:
            return None

        try:
            value = await client.get(key)
            if value is None:
                return None
            return json.loads(value)
        except Exception as exc:
            self._record_error("failed to read redis cache value", exc)
            return None

    async def _redis_set_json(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[float],
    ) -> None:
        client = await self._redis_client()
        if client is None:
            return

        try:
            payload = stable_json_dumps(value)
            ttl_seconds = self._effective_ttl(ttl_seconds)
            if ttl_seconds and ttl_seconds > 0:
                await client.set(key, payload, ex=max(1, int(ttl_seconds)))
            else:
                await client.set(key, payload)
            self._sets += 1
        except Exception as exc:
            self._record_error("failed to write redis cache value", exc)

    async def _redis_client(self) -> Any:
        cfg = _settings()
        if self._redis is not None and self._redis_url == cfg.cache_redis_url:
            return self._redis

        try:
            import redis.asyncio as redis
        except Exception as exc:
            self._record_error("redis package is not available", exc)
            return None

        try:
            self._redis = redis.from_url(cfg.cache_redis_url, decode_responses=True)
            self._redis_url = cfg.cache_redis_url
            return self._redis
        except Exception as exc:
            self._record_error("failed to create redis cache client", exc)
            return None

    def _namespace_version_key(self, namespace: str) -> str:
        cfg = _settings()
        return f"{cfg.cache_key_prefix}:v1:namespace:{namespace}:version"

    def _effective_ttl(self, ttl_seconds: Optional[float]) -> Optional[float]:
        if ttl_seconds is None:
            return _settings().cache_default_ttl_seconds
        return ttl_seconds

    def _validate_namespace(self, namespace: str) -> None:
        if namespace not in CACHE_NAMESPACES:
            raise ValueError(f"Unknown cache namespace: {namespace}")

    def _record_error(self, message: str, exc: Exception) -> None:
        self._errors += 1
        logger.warning("%s: %s", message, exc)

    @property
    def _enabled(self) -> bool:
        return bool(_settings().cache_enabled)

    @property
    def _backend(self) -> str:
        backend = (_settings().cache_backend or "memory").lower()
        return "redis" if backend == "redis" else "memory"


cache = CacheManager()
