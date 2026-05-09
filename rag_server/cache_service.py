from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
from typing import Any

CACHE_SCHEMA_VERSION = "v1"
DEFAULT_CACHE_NAMESPACE = "rag-server"


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def stable_json_dumps(value: Any) -> str:
    """Serialize values deterministically for cache keys and payloads."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def stable_cache_digest(value: Any) -> str:
    return sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheTTLs:
    query_rewrite_ttl_s: int = 86400
    embedding_ttl_s: int = 604800
    retrieval_ttl_s: int = 3600
    rerank_ttl_s: int = 86400
    memory_ttl_s: int = 300

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> CacheTTLs:
        if not value:
            return cls()
        default = cls()
        return cls(
            query_rewrite_ttl_s=int(
                value.get("query_rewrite_ttl_s", default.query_rewrite_ttl_s)
            ),
            embedding_ttl_s=int(value.get("embedding_ttl_s", default.embedding_ttl_s)),
            retrieval_ttl_s=int(value.get("retrieval_ttl_s", default.retrieval_ttl_s)),
            rerank_ttl_s=int(value.get("rerank_ttl_s", default.rerank_ttl_s)),
            memory_ttl_s=int(value.get("memory_ttl_s", default.memory_ttl_s)),
        )


class JsonCache:
    """Small JSON cache abstraction shared by Redis and tests."""

    def __init__(self, namespace: str = DEFAULT_CACHE_NAMESPACE) -> None:
        self.namespace = namespace.strip() or DEFAULT_CACHE_NAMESPACE

    def make_key(self, category: str, payload: Any) -> str:
        digest = stable_cache_digest(payload)
        return f"{self.namespace}:{CACHE_SCHEMA_VERSION}:{category}:{digest}"

    def get_json(self, key: str) -> Any | None:
        raise NotImplementedError

    def set_json(self, key: str, value: Any, ttl_s: int | None = None) -> None:
        raise NotImplementedError


class RedisJsonCache(JsonCache):
    def __init__(self, client: Any, namespace: str = DEFAULT_CACHE_NAMESPACE) -> None:
        super().__init__(namespace=namespace)
        self.client = client
        self._available = True

    def get_json(self, key: str) -> Any | None:
        if not self._available:
            return None
        try:
            raw = self.client.get(key)
        except Exception:
            self._available = False
            return None
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            return json.loads(str(raw))
        except json.JSONDecodeError:
            return None

    def set_json(self, key: str, value: Any, ttl_s: int | None = None) -> None:
        if not self._available:
            return
        if ttl_s is not None and ttl_s <= 0:
            return
        kwargs: dict[str, Any] = {}
        if ttl_s is not None:
            kwargs["ex"] = int(ttl_s)
        try:
            self.client.set(key, stable_json_dumps(value), **kwargs)
        except Exception:
            self._available = False


class InMemoryJsonCache(JsonCache):
    """In-process cache used by unit tests and lightweight integrations."""

    def __init__(self, namespace: str = DEFAULT_CACHE_NAMESPACE) -> None:
        super().__init__(namespace=namespace)
        self._values: dict[str, tuple[str, float | None]] = {}

    def get_json(self, key: str) -> Any | None:
        stored = self._values.get(key)
        if stored is None:
            return None
        raw, expires_at = stored
        if expires_at is not None and expires_at <= time.time():
            self._values.pop(key, None)
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def set_json(self, key: str, value: Any, ttl_s: int | None = None) -> None:
        if ttl_s is not None and ttl_s <= 0:
            return
        expires_at = time.time() + ttl_s if ttl_s is not None and ttl_s > 0 else None
        self._values[key] = (stable_json_dumps(value), expires_at)

    def clear(self) -> None:
        self._values.clear()


def create_redis_cache(
    *,
    enabled: bool,
    redis_url: str,
    namespace: str = DEFAULT_CACHE_NAMESPACE,
    socket_timeout_s: float = 0.2,
) -> RedisJsonCache | None:
    if not enabled:
        return None

    try:
        import redis

        client = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=socket_timeout_s,
            socket_connect_timeout=socket_timeout_s,
        )
        client.ping()
    except Exception:
        return None
    return RedisJsonCache(client, namespace=namespace)
