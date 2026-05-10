from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
from typing import Any

# 缓存键的 schema 版本号，便于未来升级缓存格式
CACHE_SCHEMA_VERSION = "v1"
# 默认的 Redis/缓存命名空间，用于区分不同应用的缓存键
DEFAULT_CACHE_NAMESPACE = "rag-server"


def _json_default(value: Any) -> Any:
    """为 json.dumps 提供默认的序列化行为，支持 dataclass 和类数组对象。"""
    # 如果是 dataclass 实例，转换为 dict
    if is_dataclass(value):
        return asdict(value)
    # 如果对象有 tolist 方法（如 numpy 数组），转为 list
    if hasattr(value, "tolist"):
        return value.tolist()
    # 兜底：转为字符串
    return str(value)


def stable_json_dumps(value: Any) -> str:
    """
    对值进行「确定性的」JSON 序列化。

    用途：为缓存键和缓存负载生成稳定的字符串。
    通过 sort_keys 和固定 separators 确保不同调用产生相同的序列化结果。
    """
    return json.dumps(
        value,
        ensure_ascii=False,       # 保留中文字符，避免转义
        sort_keys=True,           # 按 key 排序，保证输出稳定
        separators=(",", ":"),    # 紧凑分隔符，消除空格差异
        default=_json_default,    # 自定义序列化 fallback
    )


def stable_cache_digest(value: Any) -> str:
    """
    计算值的 SHA256 哈希摘要。

    用于生成缓存键的 digest 部分，相同逻辑的值生成相同的缓存键。
    """
    return sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheTTLs:
    """
    缓存 TTL（Time-To-Live）配置，各字段单位为秒（s）。

    TTL 默认值：
    - 查询改写: 86400 秒（24小时），改写结果相对稳定
    - 嵌入向量: 604800 秒（7天），向量不会频繁变化
    - 检索结果: 3600 秒（1小时），知识库可能更新
    - 重排序: 86400 秒（24小时）
    - 用户记忆: 300 秒（5分钟），记忆提取有时效性
    """
    query_rewrite_ttl_s: int = 86400
    embedding_ttl_s: int = 604800
    retrieval_ttl_s: int = 3600
    rerank_ttl_s: int = 86400
    memory_ttl_s: int = 300

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> CacheTTLs:
        """从字典创建 CacheTTLs 实例，未提供的字段使用默认值。"""
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
    """
    JSON 缓存的抽象基类，供 Redis 和内存测试实现共享。

    定义统一的缓存键生成规则和 get/set 接口，子类只需实现具体的存储逻辑。
    """

    def __init__(self, namespace: str = DEFAULT_CACHE_NAMESPACE) -> None:
        # 缓存命名空间，用于分组和隔离不同来源的缓存数据
        self.namespace = namespace.strip() or DEFAULT_CACHE_NAMESPACE

    def make_key(self, category: str, payload: Any) -> str:
        """
        生成缓存键。

        格式: {namespace}:{schema_version}:{category}:{sha256_digest}
        包含命名空间用于隔离，包含版本号用于兼容升级，包含摘要用于内容寻址。
        """
        digest = stable_cache_digest(payload)
        return f"{self.namespace}:{CACHE_SCHEMA_VERSION}:{category}:{digest}"

    def get_json(self, key: str) -> Any | None:
        """获取缓存值，返回 Python 对象或 None（未命中/异常）。"""
        raise NotImplementedError

    def set_json(self, key: str, value: Any, ttl_s: int | None = None) -> None:
        """设置缓存值，可选 TTL 过期时间（秒）。"""
        raise NotImplementedError


class RedisJsonCache(JsonCache):
    """
    基于 Redis 的 JSON 缓存实现。

    特性：
    - 自动降级：Redis 连接异常时标记为不可用，后续操作直接跳过
    - 紧凑存储：使用 stable_json_dumps 序列化，节省空间
    - 异常安全：get/set 操作失败不抛异常，静默降级
    """

    def __init__(self, client: Any, namespace: str = DEFAULT_CACHE_NAMESPACE) -> None:
        super().__init__(namespace=namespace)
        self.client = client  # Redis 客户端实例
        self._available = True  # 标记 Redis 是否可用，异常时自动降级

    def get_json(self, key: str) -> Any | None:
        """从 Redis 获取并反序列化 JSON 值，失败时静默返回 None。"""
        # 如果之前已标记不可用，直接跳过
        if not self._available:
            return None
        try:
            raw = self.client.get(key)
        except Exception:
            # 异常时标记不可用，后续不再尝试
            self._available = False
            return None
        if raw is None:
            return None
        # Redis decode_responses=True 应返回 str，但兼容 bytes 情况
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            return json.loads(str(raw))
        except json.JSONDecodeError:
            # JSON 格式损坏，返回 None（视为缓存未命中）
            return None

    def set_json(self, key: str, value: Any, ttl_s: int | None = None) -> None:
        """将值序列化为 JSON 并存入 Redis，可选设置过期时间。"""
        # 不可用时静默跳过
        if not self._available:
            return
        # TTL 为 0 或负数时忽略写入（无意义）
        if ttl_s is not None and ttl_s <= 0:
            return
        kwargs: dict[str, Any] = {}
        if ttl_s is not None:
            kwargs["ex"] = int(ttl_s)  # Redis SET 命令的 EX 参数
        try:
            self.client.set(key, stable_json_dumps(value), **kwargs)
        except Exception:
            # 写入异常时标记不可用
            self._available = False


class InMemoryJsonCache(JsonCache):
    """
    基于内存字典的 JSON 缓存实现。

    用于单元测试和轻量级集成场景，不依赖外部服务。
    支持 TTL 过期清理（惰性删除：仅在 get 时检查）。
    """

    def __init__(self, namespace: str = DEFAULT_CACHE_NAMESPACE) -> None:
        super().__init__(namespace=namespace)
        # 存储结构: key -> (序列化后的 JSON 字符串, 过期时间戳或 None)
        self._values: dict[str, tuple[str, float | None]] = {}

    def get_json(self, key: str) -> Any | None:
        """从内存获取缓存值，按需检查过期。"""
        stored = self._values.get(key)
        if stored is None:
            return None
        raw, expires_at = stored
        # 检查是否已过期（惰性删除策略）
        if expires_at is not None and expires_at <= time.time():
            # 过期则删除条目并返回未命中
            self._values.pop(key, None)
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def set_json(self, key: str, value: Any, ttl_s: int | None = None) -> None:
        """将值序列化后存入内存字典，可选设置 TTL。"""
        # TTL 为 0 或负数时忽略写入
        if ttl_s is not None and ttl_s <= 0:
            return
        # 计算过期时间：当前时间 + TTL；TTL 为 None 表示永不过期
        expires_at = time.time() + ttl_s if ttl_s is not None and ttl_s > 0 else None
        self._values[key] = (stable_json_dumps(value), expires_at)

    def clear(self) -> None:
        """清空所有内存缓存条目。"""
        self._values.clear()


def create_redis_cache(
    *,
    enabled: bool,
    redis_url: str,
    namespace: str = DEFAULT_CACHE_NAMESPACE,
    socket_timeout_s: float = 0.2,
) -> RedisJsonCache | None:
    """
    工厂函数：创建并验证 Redis 缓存连接。

    参数：
    - enabled: 全局开关，为 False 时跳过创建
    - redis_url: Redis 连接地址（格式: redis://host:port/db）
    - namespace: 缓存键命名空间
    - socket_timeout_s: Socket 超时时间（秒），较小值避免长时间阻塞

    返回：
    - 成功的 RedisJsonCache 实例
    - 连接失败或未启用时返回 None

    行为：创建后会执行 ping 命令验证连接，失败则自动降级。
    """
    # 全局开关关闭时直接返回 None，跳过 Redis 初始化
    if not enabled:
        return None

    try:
        # 延迟导入 redis 库，避免硬依赖
        import redis

        # 从 URL 创建 Redis 客户端
        client = redis.Redis.from_url(
            redis_url,
            decode_responses=True,           # 自动解码响应为 Python str
            socket_timeout=socket_timeout_s,  # 读写超时
            socket_connect_timeout=socket_timeout_s,  # 连接超时
        )
        # 执行 ping 验证连接可用性
        client.ping()
    except Exception:
        # 连接失败（如 Redis 未启动），返回 None，自动降级
        return None
    return RedisJsonCache(client, namespace=namespace)
