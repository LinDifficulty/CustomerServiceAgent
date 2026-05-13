"""测试内存 JSON 缓存服务 (InMemoryJsonCache) 的键稳定性、过期机制和零 TTL 行为。"""

from __future__ import annotations

import time
import unittest

from rag_server.cache_service import InMemoryJsonCache


class CacheServiceTests(unittest.TestCase):
    """InMemoryJsonCache 单元测试：验证缓存键生成、TTL 过期、零 TTL 跳过写入。"""

    def test_keys_are_stable_for_equivalent_payloads(self) -> None:
        """相同语义的 payload（键顺序不同）应生成相同的缓存键。"""
        cache = InMemoryJsonCache(namespace="test")

        first = cache.make_key("demo", {"b": 2, "a": 1})
        second = cache.make_key("demo", {"a": 1, "b": 2})

        self.assertEqual(first, second)

    def test_in_memory_cache_expires_values(self) -> None:
        """缓存项在设定的 TTL 之后应自动过期，读取返回 None。"""
        cache = InMemoryJsonCache(namespace="test")
        key = cache.make_key("demo", {"id": 1})

        # 写入缓存，TTL 为 1 秒
        cache.set_json(key, {"value": "cached"}, ttl_s=1)
        self.assertEqual(cache.get_json(key), {"value": "cached"})

        # 等待超过 TTL 后读取，使用 monotonic 确保精确等待
        deadline = time.monotonic() + 1.1
        while time.monotonic() < deadline:
            time.sleep(0.05)

        self.assertIsNone(cache.get_json(key))

    def test_zero_ttl_skips_write(self) -> None:
        """TTL 为 0 时应跳过写入，数据不会进入缓存。"""
        cache = InMemoryJsonCache(namespace="test")
        key = cache.make_key("demo", {"id": 2})

        cache.set_json(key, {"value": "cached"}, ttl_s=0)

        self.assertIsNone(cache.get_json(key))


if __name__ == "__main__":
    unittest.main()
