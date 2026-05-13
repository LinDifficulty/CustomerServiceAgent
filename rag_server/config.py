"""
应用配置管理模块。

职责：
1. 定义配置数据类（Settings Dataclasses），每个 section 一个
2. 聚合为顶层 AppConfig 容器
3. 提供四层优先级配置加载：默认值 < 配置文件 < 环境变量 < 代码覆写
4. 支持 TOML / JSON 格式的配置文件

所有 Settings Dataclass 均为 frozen，表示运行时不可变的已验证配置。
"""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .model_factory import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHAT_PROVIDER,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANKER_PROVIDER,
)
from .utils import coerce_bool

# ═══════════════════════════════════════════════════════════════
# 全局默认常量（被 cli.py 函数签名使用，保持集中管理）
# ═══════════════════════════════════════════════════════════════

DEFAULT_AGENT_MODEL = DEFAULT_CHAT_MODEL
DEFAULT_USER_ID = "default_user"
DEFAULT_QUERY_REWRITE_MODE = "on"
DEFAULT_MAX_TOOL_ROUNDS = 6
DEFAULT_MAX_REPEATED_TOOL_CALLS = 2
DEFAULT_REFLECTION_ENABLED = True
DEFAULT_LIVE_EVENTS_ENABLED = False
DEFAULT_CLI_CONFIG_OUTPUT_ENABLED = False
DEFAULT_STREAM_OUTPUT_ENABLED = True

# 查询改写模式枚举
QUERY_REWRITE_MODES = ("on", "off", "rewrite_only", "multi_query")

# 缓存默认值
DEFAULT_DATA_DIR = "data"
DEFAULT_MEMORY_DIR = "memory"
DEFAULT_TRACE_DIR = "traces"
DEFAULT_MCP_CONFIG_PATH = "mcp_servers.json"
DEFAULT_CACHE_ENABLED = True
DEFAULT_REDIS_URL = "redis://localhost:6379/0"
DEFAULT_CACHE_NAMESPACE = "rag-server"
DEFAULT_CACHE_SOCKET_TIMEOUT_S = 0.2
DEFAULT_QUERY_REWRITE_CACHE_TTL_S = 86400  # 24小时
DEFAULT_EMBEDDING_CACHE_TTL_S = 604800  # 7天
DEFAULT_RETRIEVAL_CACHE_TTL_S = 3600  # 1小时
DEFAULT_RERANK_CACHE_TTL_S = 86400  # 24小时
DEFAULT_MEMORY_CACHE_TTL_S = 300  # 5分钟


class ConfigError(ValueError):
    """配置无效时抛出的异常。"""


# ═══════════════════════════════════════════════════════════════
# Settings Dataclasses
# ═══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PathSettings:
    """文件路径配置。"""

    data_dir: str = DEFAULT_DATA_DIR
    memory_dir: str = DEFAULT_MEMORY_DIR
    trace_dir: str = DEFAULT_TRACE_DIR
    mcp_config_path: str = DEFAULT_MCP_CONFIG_PATH


@dataclass(frozen=True)
class AgentSettings:
    """Agent 行为配置：模型、用户、工具调用限制、反思开关。"""

    provider: str = DEFAULT_CHAT_PROVIDER
    model: str = DEFAULT_CHAT_MODEL
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    user_id: str = DEFAULT_USER_ID
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS
    max_repeated_tool_calls: int = DEFAULT_MAX_REPEATED_TOOL_CALLS
    reflection_enabled: bool = DEFAULT_REFLECTION_ENABLED


@dataclass(frozen=True)
class RetrievalSettings:
    """检索配置：查询改写、BM25、CrossEncoder、嵌入和重排序模型。"""

    query_rewrite: str = DEFAULT_QUERY_REWRITE_MODE
    bm25: bool = True
    cross_encoder: bool = False
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_kwargs: dict[str, Any] = field(default_factory=dict)
    reranker_provider: str = DEFAULT_RERANKER_PROVIDER
    reranker_model: str = DEFAULT_RERANKER_MODEL
    reranker_kwargs: dict[str, Any] = field(default_factory=dict)
    reranker_device: str | None = None
    reranker_batch_size: int = 16


@dataclass(frozen=True)
class LLMSettings:
    """LLM 调用配置：查询改写、记忆提取的独立 provider/model 及重试策略。"""

    rewrite_provider: str | None = None
    rewrite_model: str | None = None
    rewrite_kwargs: dict[str, Any] = field(default_factory=dict)
    memory_provider: str | None = None
    memory_model: str | None = None
    memory_kwargs: dict[str, Any] = field(default_factory=dict)
    retry_attempts: int = 3
    timeout_s: float | None = 30.0
    retry_backoff_s: float = 1.0


@dataclass(frozen=True)
class MemorySettings:
    """用户记忆配置。"""

    enabled: bool = True
    top_k: int = 5


@dataclass(frozen=True)
class SkillsSettings:
    """Skills 能力注册配置。"""

    enabled: bool = True
    dirs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MCPSettings:
    """MCP 客户端配置。"""

    enabled: bool = False


@dataclass(frozen=True)
class TraceSettings:
    """链路追踪配置。"""

    enabled: bool = False
    live: bool = DEFAULT_LIVE_EVENTS_ENABLED


@dataclass(frozen=True)
class CLISettings:
    """CLI 交互界面配置。"""

    show_config: bool = DEFAULT_CLI_CONFIG_OUTPUT_ENABLED
    stream_output: bool = DEFAULT_STREAM_OUTPUT_ENABLED


@dataclass(frozen=True)
class CacheSettings:
    """缓存配置（Redis）。"""

    enabled: bool = DEFAULT_CACHE_ENABLED
    redis_url: str = DEFAULT_REDIS_URL
    namespace: str = DEFAULT_CACHE_NAMESPACE
    socket_timeout_s: float = DEFAULT_CACHE_SOCKET_TIMEOUT_S
    query_rewrite_ttl_s: int = DEFAULT_QUERY_REWRITE_CACHE_TTL_S
    embedding_ttl_s: int = DEFAULT_EMBEDDING_CACHE_TTL_S
    retrieval_ttl_s: int = DEFAULT_RETRIEVAL_CACHE_TTL_S
    rerank_ttl_s: int = DEFAULT_RERANK_CACHE_TTL_S
    memory_ttl_s: int = DEFAULT_MEMORY_CACHE_TTL_S


@dataclass(frozen=True)
class AppConfig:
    """应用总配置容器，聚合所有子配置。"""

    paths: PathSettings = field(default_factory=PathSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    skills: SkillsSettings = field(default_factory=SkillsSettings)
    mcp: MCPSettings = field(default_factory=MCPSettings)
    trace: TraceSettings = field(default_factory=TraceSettings)
    cli: CLISettings = field(default_factory=CLISettings)
    cache: CacheSettings = field(default_factory=CacheSettings)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AppConfig:
        raw = _normalize_mapping(value)
        _ensure_known_sections(raw)
        return cls(
            paths=_path_settings(raw.get("paths", {})),
            agent=_agent_settings(raw.get("agent", {})),
            retrieval=_retrieval_settings(raw.get("retrieval", {})),
            llm=_llm_settings(raw.get("llm", {})),
            memory=_memory_settings(raw.get("memory", {})),
            skills=_skills_settings(raw.get("skills", {})),
            mcp=_mcp_settings(raw.get("mcp", {})),
            trace=_trace_settings(raw.get("trace", {})),
            cli=_cli_settings(raw.get("cli", {})),
            cache=_cache_settings(raw.get("cache", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_runtime_kwargs(self) -> dict[str, Any]:
        """转换为扁平化的运行时 kwargs，供 RAGService、Agent 等组件直接消费。

        LLM 继承规则：查询改写和记忆提取的 provider/model 未指定时回退到 agent 配置。
        """
        rewrite_provider = self.llm.rewrite_provider or self.agent.provider
        rewrite_model = self.llm.rewrite_model or self.agent.model
        rewrite_kwargs = (
            self.agent.model_kwargs
            if (self.llm.rewrite_provider is None and not self.llm.rewrite_kwargs)
            else self.llm.rewrite_kwargs
        )
        memory_provider = self.llm.memory_provider or self.agent.provider
        memory_model = self.llm.memory_model or self.agent.model
        memory_kwargs = (
            self.agent.model_kwargs
            if (self.llm.memory_provider is None and not self.llm.memory_kwargs)
            else self.llm.memory_kwargs
        )
        return {
            # 检索
            "query_rewrite_mode": self.retrieval.query_rewrite,
            "rewrite_provider": rewrite_provider,
            "rewrite_model_name": rewrite_model,
            "rewrite_model_kwargs": dict(rewrite_kwargs),
            "bm25_enabled": self.retrieval.bm25,
            "cross_encoder_enabled": self.retrieval.cross_encoder,
            "embedding_provider": self.retrieval.embedding_provider,
            "embedding_model_name": self.retrieval.embedding_model,
            "embedding_model_kwargs": dict(self.retrieval.embedding_kwargs),
            "reranker_provider": self.retrieval.reranker_provider,
            "reranker_model_name": self.retrieval.reranker_model,
            "reranker_model_kwargs": dict(self.retrieval.reranker_kwargs),
            "reranker_device": self.retrieval.reranker_device,
            "reranker_batch_size": self.retrieval.reranker_batch_size,
            # Agent & 用户
            "user_id": self.agent.user_id,
            "memory_enabled": self.memory.enabled,
            "memory_provider": memory_provider,
            "memory_model_name": memory_model,
            "memory_model_kwargs": dict(memory_kwargs),
            "memory_top_k": self.memory.top_k,
            "skills_enabled": self.skills.enabled,
            "skill_dirs": self.skills.dirs or None,
            "mcp_enabled": self.mcp.enabled,
            "mcp_config_path": self.paths.mcp_config_path,
            # 追踪
            "trace_enabled": self.trace.enabled,
            "live_events_enabled": self.trace.live,
            "trace_dir": self.paths.trace_dir,
            # CLI
            "show_config": self.cli.show_config,
            "stream_output_enabled": self.cli.stream_output,
            # LLM 重试
            "llm_retry_attempts": self.llm.retry_attempts,
            "llm_timeout_s": self.llm.timeout_s,
            "llm_retry_backoff_s": self.llm.retry_backoff_s,
            # Agent 流程控制
            "max_tool_rounds": self.agent.max_tool_rounds,
            "max_repeated_tool_calls": self.agent.max_repeated_tool_calls,
            "reflection_enabled": self.agent.reflection_enabled,
            # 路径
            "data_dir": self.paths.data_dir,
            "memory_dir": self.paths.memory_dir,
            # Agent LLM
            "agent_provider": self.agent.provider,
            "agent_model_name": self.agent.model,
            "agent_model_kwargs": dict(self.agent.model_kwargs),
            # 缓存
            "cache_enabled": self.cache.enabled,
            "cache_redis_url": self.cache.redis_url,
            "cache_namespace": self.cache.namespace,
            "cache_socket_timeout_s": self.cache.socket_timeout_s,
            "cache_query_rewrite_ttl_s": self.cache.query_rewrite_ttl_s,
            "cache_embedding_ttl_s": self.cache.embedding_ttl_s,
            "cache_retrieval_ttl_s": self.cache.retrieval_ttl_s,
            "cache_rerank_ttl_s": self.cache.rerank_ttl_s,
            "cache_memory_ttl_s": self.cache.memory_ttl_s,
        }


# ═══════════════════════════════════════════════════════════════
# 公开 API
# ═══════════════════════════════════════════════════════════════


def load_app_config(
    config_path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> AppConfig:
    """加载应用配置，四层优先级（从低到高）：

    1. 默认值
    2. 配置文件（JSON / TOML，路径由 config_path 或环境变量 RAG_SERVER_CONFIG 指定）
    3. 环境变量（RAG_SERVER_* 前缀）
    4. 代码覆写（overrides 参数）
    """
    actual_env = os.environ if env is None else env
    selected_config_path = config_path or actual_env.get("RAG_SERVER_CONFIG")
    merged: dict[str, Any] = AppConfig().to_dict()

    if selected_config_path:
        loaded = load_config_file(selected_config_path)
        merged = _deep_merge(merged, loaded)

    env_overrides = build_env_overrides(actual_env)
    if env_overrides:
        merged = _deep_merge(merged, env_overrides)

    if overrides:
        merged = _deep_merge(merged, _normalize_mapping(overrides))

    return AppConfig.from_mapping(merged)


def load_config_file(config_path: str | Path) -> dict[str, Any]:
    """加载并解析配置文件，支持 .json 和 .toml 格式。"""
    path = Path(config_path).expanduser()
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    if not path.is_file():
        raise ConfigError(f"Config path is not a file: {path}")

    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        elif suffix in {".toml", ".tml"}:
            payload = tomllib.loads(path.read_text(encoding="utf-8"))
        else:
            raise ConfigError("Config file must be .toml or .json")
    except json.JSONDecodeError as error:
        raise ConfigError(f"Invalid JSON config: {error}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"Invalid TOML config: {error}") from error

    if not isinstance(payload, dict):
        raise ConfigError("Config file must contain an object")
    return _normalize_mapping(payload)


def build_env_overrides(env: Mapping[str, str]) -> dict[str, Any]:
    """从环境变量构建配置覆写字典。

    环境变量命名规则：RAG_SERVER_<SECTION>_<KEY>（全大写，下划线分隔）
    例如：RAG_SERVER_DATA_DIR → paths.data_dir
          RAG_SERVER_AGENT_PROVIDER → agent.provider
          RAG_SERVER_BM25 → retrieval.bm25
    """
    mapping: dict[str, tuple[str, str]] = {
        # Paths
        "RAG_SERVER_DATA_DIR": ("paths", "data_dir"),
        "RAG_SERVER_MEMORY_DIR": ("paths", "memory_dir"),
        "RAG_SERVER_TRACE_DIR": ("paths", "trace_dir"),
        "RAG_SERVER_MCP_CONFIG": ("paths", "mcp_config_path"),
        # Agent
        "RAG_SERVER_AGENT_PROVIDER": ("agent", "provider"),
        "RAG_SERVER_AGENT_MODEL": ("agent", "model"),
        "RAG_SERVER_AGENT_MODEL_KWARGS": ("agent", "model_kwargs"),
        "RAG_SERVER_USER_ID": ("agent", "user_id"),
        "RAG_SERVER_MAX_TOOL_ROUNDS": ("agent", "max_tool_rounds"),
        "RAG_SERVER_MAX_REPEATED_TOOL_CALLS": ("agent", "max_repeated_tool_calls"),
        "RAG_SERVER_REFLECTION": ("agent", "reflection_enabled"),
        # Retrieval
        "RAG_SERVER_QUERY_REWRITE": ("retrieval", "query_rewrite"),
        "RAG_SERVER_BM25": ("retrieval", "bm25"),
        "RAG_SERVER_CROSS_ENCODER": ("retrieval", "cross_encoder"),
        "RAG_SERVER_EMBEDDING_PROVIDER": ("retrieval", "embedding_provider"),
        "RAG_SERVER_EMBEDDING_MODEL": ("retrieval", "embedding_model"),
        "RAG_SERVER_EMBEDDING_KWARGS": ("retrieval", "embedding_kwargs"),
        "RAG_SERVER_RERANKER_PROVIDER": ("retrieval", "reranker_provider"),
        "RAG_SERVER_RERANKER_MODEL": ("retrieval", "reranker_model"),
        "RAG_SERVER_RERANKER_KWARGS": ("retrieval", "reranker_kwargs"),
        "RAG_SERVER_RERANKER_DEVICE": ("retrieval", "reranker_device"),
        "RAG_SERVER_RERANKER_BATCH_SIZE": ("retrieval", "reranker_batch_size"),
        # LLM
        "RAG_SERVER_REWRITE_PROVIDER": ("llm", "rewrite_provider"),
        "RAG_SERVER_REWRITE_MODEL": ("llm", "rewrite_model"),
        "RAG_SERVER_REWRITE_KWARGS": ("llm", "rewrite_kwargs"),
        "RAG_SERVER_MEMORY_PROVIDER": ("llm", "memory_provider"),
        "RAG_SERVER_MEMORY_MODEL": ("llm", "memory_model"),
        "RAG_SERVER_MEMORY_KWARGS": ("llm", "memory_kwargs"),
        "RAG_SERVER_LLM_RETRY_ATTEMPTS": ("llm", "retry_attempts"),
        "RAG_SERVER_LLM_TIMEOUT": ("llm", "timeout_s"),
        "RAG_SERVER_LLM_RETRY_BACKOFF": ("llm", "retry_backoff_s"),
        # Memory
        "RAG_SERVER_MEMORY": ("memory", "enabled"),
        "RAG_SERVER_MEMORY_TOP_K": ("memory", "top_k"),
        # Skills
        "RAG_SERVER_SKILLS": ("skills", "enabled"),
        "RAG_SERVER_SKILLS_DIRS": ("skills", "dirs"),
        # MCP
        "RAG_SERVER_MCP": ("mcp", "enabled"),
        # Trace
        "RAG_SERVER_TRACE": ("trace", "enabled"),
        "RAG_SERVER_LIVE_EVENTS": ("trace", "live"),
        # CLI
        "RAG_SERVER_SHOW_CONFIG": ("cli", "show_config"),
        "RAG_SERVER_STREAM_OUTPUT": ("cli", "stream_output"),
        # Cache
        "RAG_SERVER_CACHE": ("cache", "enabled"),
        "RAG_SERVER_REDIS_URL": ("cache", "redis_url"),
        "RAG_SERVER_CACHE_NAMESPACE": ("cache", "namespace"),
        "RAG_SERVER_CACHE_SOCKET_TIMEOUT": ("cache", "socket_timeout_s"),
        "RAG_SERVER_CACHE_QUERY_REWRITE_TTL": ("cache", "query_rewrite_ttl_s"),
        "RAG_SERVER_CACHE_EMBEDDING_TTL": ("cache", "embedding_ttl_s"),
        "RAG_SERVER_CACHE_RETRIEVAL_TTL": ("cache", "retrieval_ttl_s"),
        "RAG_SERVER_CACHE_RERANK_TTL": ("cache", "rerank_ttl_s"),
        "RAG_SERVER_CACHE_MEMORY_TTL": ("cache", "memory_ttl_s"),
    }
    result: dict[str, Any] = {}
    for env_name, (section, key) in mapping.items():
        if env_name in env:
            result.setdefault(section, {})[key] = env[env_name]
    return result


# ═══════════════════════════════════════════════════════════════
# 别名规范化
# ═══════════════════════════════════════════════════════════════


def _normalize_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """规范化配置字典，将常见别名映射到标准键名。

    保留少量常用别名，帮助减少配置文件中的拼写困惑：
    - agent.chat_provider → agent.provider
    - retrieval.bm25_enabled → retrieval.bm25
    - retrieval.cross_encoder_enabled → retrieval.cross_encoder
    - cli.streaming → cli.stream_output
    """
    aliases: dict[tuple[str, str], tuple[str, str]] = {
        # Agent
        ("agent", "chat_provider"): ("agent", "provider"),
        ("agent", "chat_model"): ("agent", "model"),
        ("agent", "chat_model_kwargs"): ("agent", "model_kwargs"),
        ("agent", "reflection"): ("agent", "reflection_enabled"),
        # Retrieval
        ("retrieval", "query_rewrite_mode"): ("retrieval", "query_rewrite"),
        ("retrieval", "bm25_enabled"): ("retrieval", "bm25"),
        ("retrieval", "cross_encoder_enabled"): ("retrieval", "cross_encoder"),
        # LLM
        ("llm", "query_rewrite_provider"): ("llm", "rewrite_provider"),
        ("llm", "query_rewrite_model"): ("llm", "rewrite_model"),
        ("llm", "query_rewrite_kwargs"): ("llm", "rewrite_kwargs"),
        ("llm", "llm_timeout_s"): ("llm", "timeout_s"),
        # Memory
        ("memory", "memory_top_k"): ("memory", "top_k"),
        # Trace
        ("trace", "live_events"): ("trace", "live"),
        ("trace", "live_logs"): ("trace", "live"),
        ("cli", "live_events"): ("trace", "live"),
        ("cli", "live_logs"): ("trace", "live"),
        # CLI
        ("cli", "streaming"): ("cli", "stream_output"),
        ("cli", "stream_output_enabled"): ("cli", "stream_output"),
        ("cli", "show_startup_config"): ("cli", "show_config"),
        # Cache
        ("cache", "url"): ("cache", "redis_url"),
    }
    normalized: dict[str, Any] = {}
    for section_name, raw_section in value.items():
        if not isinstance(raw_section, Mapping):
            normalized[str(section_name)] = raw_section
            continue
        normalized_section: dict[str, Any] = {}
        for key, item in raw_section.items():
            target_section, target_key = aliases.get(
                (str(section_name), str(key)),
                (str(section_name), str(key)),
            )
            if target_section != str(section_name):
                normalized.setdefault(target_section, {})[target_key] = item
            else:
                normalized_section[target_key] = item
        existing = normalized.get(str(section_name))
        if isinstance(existing, Mapping):
            normalized[str(section_name)] = _deep_merge(existing, normalized_section)
        else:
            normalized[str(section_name)] = normalized_section
    return normalized


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """深层合并两个字典。双方同一键的值都是字典时递归合并，否则 overlay 覆盖。"""
    merged = {str(key): value for key, value in base.items()}
    for key, value in overlay.items():
        key = str(key)
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# ═══════════════════════════════════════════════════════════════
# Section 解析函数
# ═══════════════════════════════════════════════════════════════

_ALLOWED_SECTIONS = {
    "paths",
    "agent",
    "retrieval",
    "llm",
    "memory",
    "skills",
    "mcp",
    "trace",
    "cli",
    "cache",
}


def _ensure_known_sections(raw: Mapping[str, Any]) -> None:
    unknown = sorted(set(raw) - _ALLOWED_SECTIONS)
    if unknown:
        raise ConfigError(f"Unknown config section(s): {', '.join(unknown)}")


def _ensure_known_keys(section_name: str, raw: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"Unknown key(s) in {section_name}: {', '.join(unknown)}")


def _init_settings(raw: Any, section_name: str, settings_cls: type, allowed_keys: set[str]) -> tuple[dict[str, Any], Any]:
    """解析配置 section 并返回 (原始字典, 默认值实例)。"""
    section = _section(raw, section_name)
    _ensure_known_keys(section_name, section, allowed_keys)
    return section, settings_cls()


def _path_settings(raw: Any) -> PathSettings:
    s, d = _init_settings(raw, "paths", PathSettings, {"data_dir", "memory_dir", "trace_dir", "mcp_config_path"})
    return PathSettings(
        data_dir=_non_empty_str(s.get("data_dir", d.data_dir), "paths.data_dir"),
        memory_dir=_non_empty_str(s.get("memory_dir", d.memory_dir), "paths.memory_dir"),
        trace_dir=_non_empty_str(s.get("trace_dir", d.trace_dir), "paths.trace_dir"),
        mcp_config_path=_non_empty_str(s.get("mcp_config_path", d.mcp_config_path), "paths.mcp_config_path"),
    )


def _agent_settings(raw: Any) -> AgentSettings:
    s, d = _init_settings(raw, "agent", AgentSettings, {
        "provider", "model", "model_kwargs", "user_id",
        "max_tool_rounds", "max_repeated_tool_calls", "reflection_enabled",
    })
    return AgentSettings(
        provider=_non_empty_str(s.get("provider", d.provider), "agent.provider"),
        model=_non_empty_str(s.get("model", d.model), "agent.model"),
        model_kwargs=_coerce_mapping(s.get("model_kwargs", d.model_kwargs), "agent.model_kwargs"),
        user_id=_non_empty_str(s.get("user_id", d.user_id), "agent.user_id"),
        max_tool_rounds=_coerce_int(s.get("max_tool_rounds", d.max_tool_rounds), "agent.max_tool_rounds", minimum=0),
        max_repeated_tool_calls=_coerce_int(s.get("max_repeated_tool_calls", d.max_repeated_tool_calls), "agent.max_repeated_tool_calls", minimum=1),
        reflection_enabled=_coerce_bool(s.get("reflection_enabled", d.reflection_enabled), "agent.reflection_enabled"),
    )


def _retrieval_settings(raw: Any) -> RetrievalSettings:
    s, d = _init_settings(raw, "retrieval", RetrievalSettings, {
        "query_rewrite", "bm25", "cross_encoder",
        "embedding_provider", "embedding_model", "embedding_kwargs",
        "reranker_provider", "reranker_model", "reranker_kwargs",
        "reranker_device", "reranker_batch_size",
    })
    query_rewrite = _non_empty_str(s.get("query_rewrite", d.query_rewrite), "retrieval.query_rewrite")
    if query_rewrite not in QUERY_REWRITE_MODES:
        raise ConfigError(f"retrieval.query_rewrite must be one of: {', '.join(QUERY_REWRITE_MODES)}")
    return RetrievalSettings(
        query_rewrite=query_rewrite,
        bm25=_coerce_bool(s.get("bm25", d.bm25), "retrieval.bm25"),
        cross_encoder=_coerce_bool(s.get("cross_encoder", d.cross_encoder), "retrieval.cross_encoder"),
        embedding_provider=_non_empty_str(s.get("embedding_provider", d.embedding_provider), "retrieval.embedding_provider"),
        embedding_model=_non_empty_str(s.get("embedding_model", d.embedding_model), "retrieval.embedding_model"),
        embedding_kwargs=_coerce_mapping(s.get("embedding_kwargs", d.embedding_kwargs), "retrieval.embedding_kwargs"),
        reranker_provider=_non_empty_str(s.get("reranker_provider", d.reranker_provider), "retrieval.reranker_provider"),
        reranker_model=_non_empty_str(s.get("reranker_model", d.reranker_model), "retrieval.reranker_model"),
        reranker_kwargs=_coerce_mapping(s.get("reranker_kwargs", d.reranker_kwargs), "retrieval.reranker_kwargs"),
        reranker_device=_optional_str(s.get("reranker_device", d.reranker_device)),
        reranker_batch_size=_coerce_int(s.get("reranker_batch_size", d.reranker_batch_size), "retrieval.reranker_batch_size", minimum=1),
    )


def _llm_settings(raw: Any) -> LLMSettings:
    s, d = _init_settings(raw, "llm", LLMSettings, {
        "rewrite_provider", "rewrite_model", "rewrite_kwargs",
        "memory_provider", "memory_model", "memory_kwargs",
        "retry_attempts", "timeout_s", "retry_backoff_s",
    })
    return LLMSettings(
        rewrite_provider=_optional_str(s.get("rewrite_provider", d.rewrite_provider)),
        rewrite_model=_optional_str(s.get("rewrite_model", d.rewrite_model)),
        rewrite_kwargs=_coerce_mapping(s.get("rewrite_kwargs", d.rewrite_kwargs), "llm.rewrite_kwargs"),
        memory_provider=_optional_str(s.get("memory_provider", d.memory_provider)),
        memory_model=_optional_str(s.get("memory_model", d.memory_model)),
        memory_kwargs=_coerce_mapping(s.get("memory_kwargs", d.memory_kwargs), "llm.memory_kwargs"),
        retry_attempts=_coerce_int(s.get("retry_attempts", d.retry_attempts), "llm.retry_attempts", minimum=1),
        timeout_s=_coerce_optional_float(s.get("timeout_s", d.timeout_s), "llm.timeout_s"),
        retry_backoff_s=_coerce_float(s.get("retry_backoff_s", d.retry_backoff_s), "llm.retry_backoff_s", minimum=0.0),
    )


def _memory_settings(raw: Any) -> MemorySettings:
    s, d = _init_settings(raw, "memory", MemorySettings, {"enabled", "top_k"})
    return MemorySettings(
        enabled=_coerce_bool(s.get("enabled", d.enabled), "memory.enabled"),
        top_k=_coerce_int(s.get("top_k", d.top_k), "memory.top_k", minimum=1),
    )


def _skills_settings(raw: Any) -> SkillsSettings:
    s, d = _init_settings(raw, "skills", SkillsSettings, {"enabled", "dirs"})
    return SkillsSettings(
        enabled=_coerce_bool(s.get("enabled", d.enabled), "skills.enabled"),
        dirs=_coerce_str_list(s.get("dirs", d.dirs), "skills.dirs"),
    )


def _mcp_settings(raw: Any) -> MCPSettings:
    s, d = _init_settings(raw, "mcp", MCPSettings, {"enabled"})
    return MCPSettings(enabled=_coerce_bool(s.get("enabled", d.enabled), "mcp.enabled"))


def _trace_settings(raw: Any) -> TraceSettings:
    s, d = _init_settings(raw, "trace", TraceSettings, {"enabled", "live"})
    return TraceSettings(
        enabled=_coerce_bool(s.get("enabled", d.enabled), "trace.enabled"),
        live=_coerce_bool(s.get("live", d.live), "trace.live"),
    )


def _cli_settings(raw: Any) -> CLISettings:
    s, d = _init_settings(raw, "cli", CLISettings, {"show_config", "stream_output"})
    return CLISettings(
        show_config=_coerce_bool(s.get("show_config", d.show_config), "cli.show_config"),
        stream_output=_coerce_bool(s.get("stream_output", d.stream_output), "cli.stream_output"),
    )


def _cache_settings(raw: Any) -> CacheSettings:
    s, d = _init_settings(raw, "cache", CacheSettings, {
        "enabled", "redis_url", "namespace", "socket_timeout_s",
        "query_rewrite_ttl_s", "embedding_ttl_s", "retrieval_ttl_s",
        "rerank_ttl_s", "memory_ttl_s",
    })
    return CacheSettings(
        enabled=_coerce_bool(s.get("enabled", d.enabled), "cache.enabled"),
        redis_url=_non_empty_str(s.get("redis_url", d.redis_url), "cache.redis_url"),
        namespace=_non_empty_str(s.get("namespace", d.namespace), "cache.namespace"),
        socket_timeout_s=_coerce_float(s.get("socket_timeout_s", d.socket_timeout_s), "cache.socket_timeout_s", minimum=0.0),
        query_rewrite_ttl_s=_coerce_int(s.get("query_rewrite_ttl_s", d.query_rewrite_ttl_s), "cache.query_rewrite_ttl_s", minimum=0),
        embedding_ttl_s=_coerce_int(s.get("embedding_ttl_s", d.embedding_ttl_s), "cache.embedding_ttl_s", minimum=0),
        retrieval_ttl_s=_coerce_int(s.get("retrieval_ttl_s", d.retrieval_ttl_s), "cache.retrieval_ttl_s", minimum=0),
        rerank_ttl_s=_coerce_int(s.get("rerank_ttl_s", d.rerank_ttl_s), "cache.rerank_ttl_s", minimum=0),
        memory_ttl_s=_coerce_int(s.get("memory_ttl_s", d.memory_ttl_s), "cache.memory_ttl_s", minimum=0),
    )


# ═══════════════════════════════════════════════════════════════
# 类型转换工具函数
# ═══════════════════════════════════════════════════════════════


def _section(raw: Any, name: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{name} must be an object")
    return {str(key): value for key, value in raw.items()}


def _non_empty_str(value: Any, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ConfigError(f"{field_name} must not be empty")
    return text


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_bool(value: Any, field_name: str) -> bool:
    try:
        return coerce_bool(value, strict=True)
    except ValueError as exc:
        raise ConfigError(f"{field_name}: {exc}") from exc


def _coerce_int(value: Any, field_name: str, *, minimum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{field_name} must be an integer") from error
    if number < minimum:
        raise ConfigError(f"{field_name} must be >= {minimum}")
    return number


def _coerce_float(value: Any, field_name: str, *, minimum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{field_name} must be a number") from error
    if number < minimum:
        raise ConfigError(f"{field_name} must be >= {minimum:g}")
    return number


def _coerce_optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    number = _coerce_float(value, field_name, minimum=0.0)
    return None if number <= 0 else number


def _coerce_str_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, list | tuple):
        items = list(value)
    else:
        raise ConfigError(f"{field_name} must be a list of strings")
    return [str(item).strip() for item in items if str(item).strip()]


def _coerce_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            raise ConfigError(f"{field_name} must be a JSON object") from error
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field_name} must be an object")
    return {str(key): item for key, item in value.items()}
