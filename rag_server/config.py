from __future__ import annotations

import json
import os
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

# 从 model_factory 导入各模型类型的默认 provider 和 model 名称
from .model_factory import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHAT_PROVIDER,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANKER_PROVIDER,
)

# ── 全局默认常量 ──
# Agent 对话模型默认与 Chat 模型一致
DEFAULT_AGENT_MODEL = DEFAULT_CHAT_MODEL
# 默认用户 ID，单用户场景使用
DEFAULT_USER_ID = "default_user"
# 默认开启查询改写
DEFAULT_QUERY_REWRITE_MODE = "on"
# 查询改写的四种模式：开启、关闭、仅改写、多查询融合
QUERY_REWRITE_MODES = ("on", "off", "rewrite_only", "multi_query")
# Agent 最大工具调用轮次（防止无限循环）
DEFAULT_MAX_TOOL_ROUNDS = 6
# 同一工具最大重复调用次数（防止 Agent 死循环调用同一工具）
DEFAULT_MAX_REPEATED_TOOL_CALLS = 2
# 默认开启反思（Reflection）代理
DEFAULT_REFLECTION_ENABLED = True
# 知识库数据目录
DEFAULT_DATA_DIR = "data"
# 用户记忆存储目录
DEFAULT_MEMORY_DIR = "memory"
# 追踪日志输出目录
DEFAULT_TRACE_DIR = "traces"
# MCP 服务器配置文件路径
DEFAULT_MCP_CONFIG_PATH = "mcp_servers.json"
# 默认关闭实时事件推送
DEFAULT_LIVE_EVENTS_ENABLED = False
# 默认关闭 CLI 启动时的配置输出
DEFAULT_CLI_CONFIG_OUTPUT_ENABLED = False
# 默认开启 Agent 流式输出
DEFAULT_STREAM_OUTPUT_ENABLED = True

# ── 缓存相关默认常量 ──
# 默认关闭缓存（需要用户显式开启 Redis）
DEFAULT_CACHE_ENABLED = True
# 默认 Redis 连接地址
DEFAULT_REDIS_URL = "redis://localhost:6379/0"
# 缓存键命名空间
DEFAULT_CACHE_NAMESPACE = "rag-server"
# Redis Socket 超时（秒），较小值避免长时间阻塞
DEFAULT_CACHE_SOCKET_TIMEOUT_S = 0.2
# 查询改写缓存 TTL: 86400 秒 = 24 小时
DEFAULT_QUERY_REWRITE_CACHE_TTL_S = 86400
# 嵌入向量缓存 TTL: 604800 秒 = 7 天
DEFAULT_EMBEDDING_CACHE_TTL_S = 604800
# 检索结果缓存 TTL: 3600 秒 = 1 小时
DEFAULT_RETRIEVAL_CACHE_TTL_S = 3600
# 重排序缓存 TTL: 86400 秒 = 24 小时
DEFAULT_RERANK_CACHE_TTL_S = 86400
# 用户记忆缓存 TTL: 300 秒 = 5 分钟
DEFAULT_MEMORY_CACHE_TTL_S = 300


class ConfigError(ValueError):
    """配置无效时抛出的异常，继承自 ValueError。"""


# ═══════════════════════════════════════════════════════════════
# Settings Dataclasses（配置数据类）
# 每个数据类对应配置文件中的一个 section，均为 frozen 不可变
# ═══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PathSettings:
    """文件路径相关配置。

    管理知识库、记忆、追踪、MCP 配置等文件的存放路径。
    """
    data_dir: str = DEFAULT_DATA_DIR
    memory_dir: str = DEFAULT_MEMORY_DIR
    trace_dir: str = DEFAULT_TRACE_DIR
    mcp_config_path: str = DEFAULT_MCP_CONFIG_PATH


@dataclass(frozen=True)
class AgentSettings:
    """Agent 行为相关配置。

    控制 Agent 使用的 LLM 模型、用户 ID、工具调用限制、反思开关等。
    """
    provider: str = DEFAULT_CHAT_PROVIDER
    model: str = DEFAULT_AGENT_MODEL
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    user_id: str = DEFAULT_USER_ID
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS
    max_repeated_tool_calls: int = DEFAULT_MAX_REPEATED_TOOL_CALLS
    reflection_enabled: bool = DEFAULT_REFLECTION_ENABLED


@dataclass(frozen=True)
class RetrievalSettings:
    """检索相关配置。

    控制查询改写模式、BM25 混合检索开关、CrossEncoder 重排序开关、
    嵌入模型、重排序模型等参数。
    """
    query_rewrite: str = DEFAULT_QUERY_REWRITE_MODE
    bm25: bool = True
    cross_encoder: bool = False
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_kwargs: dict[str, Any] = field(default_factory=dict)
    reranker_provider: str = DEFAULT_RERANKER_PROVIDER
    reranker_model: str = DEFAULT_RERANKER_MODEL
    reranker_kwargs: dict[str, Any] = field(default_factory=dict)
    reranker_device: str | None = None  # 设备选择（如 "cuda", "cpu"），None 表示自动
    reranker_batch_size: int = 16


@dataclass(frozen=True)
class LLMSettings:
    """LLM 调用相关配置。

    独立控制查询改写和记忆提取使用的 LLM provider/model，
    以及重试策略（次数、超时、退避时间）。
    如果不指定，则回退到 AgentSettings 中的 provider/model。
    """
    rewrite_provider: str | None = None
    rewrite_model: str | None = None
    rewrite_kwargs: dict[str, Any] = field(default_factory=dict)
    memory_provider: str | None = None
    memory_model: str | None = None
    memory_kwargs: dict[str, Any] = field(default_factory=dict)
    retry_attempts: int = 3       # 失败重试次数
    timeout_s: float | None = 30.0  # 调用超时（秒），None 表示不限制
    retry_backoff_s: float = 1.0  # 失败重试等待时间（秒）


@dataclass(frozen=True)
class MemorySettings:
    """用户记忆相关配置。"""
    enabled: bool = True  # 是否启用记忆功能
    top_k: int = 5        # 每次召回的记忆条目数


@dataclass(frozen=True)
class SkillsSettings:
    """Skills 能力注册相关配置。"""
    enabled: bool = True      # 是否启用 Skills 功能
    dirs: list[str] = field(default_factory=list)  # 额外的 Skills 目录列表


@dataclass(frozen=True)
class MCPSettings:
    """MCP 客户端相关配置。"""
    enabled: bool = False  # 是否启用 MCP（默认关闭，需显式开启）


@dataclass(frozen=True)
class TraceSettings:
    """链路追踪相关配置。"""
    enabled: bool = False                 # 是否启用追踪日志
    live: bool = DEFAULT_LIVE_EVENTS_ENABLED  # 是否启用实时事件推送


@dataclass(frozen=True)
class CLISettings:
    """CLI 交互界面相关配置。"""
    show_config: bool = DEFAULT_CLI_CONFIG_OUTPUT_ENABLED  # 启动时是否打印配置
    stream_output: bool = DEFAULT_STREAM_OUTPUT_ENABLED    # 是否启用流式输出


@dataclass(frozen=True)
class CacheSettings:
    """缓存相关配置。

    控制 Redis 缓存的行为，包括开关、连接参数和各项 TTL 设置。
    """
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
    """
    应用总配置，聚合所有子配置的顶级容器。

    包含九个子配置块：
    - paths:   文件路径
    - agent:   Agent 行为
    - retrieval: 检索参数
    - llm:     LLM 调用策略
    - memory:  用户记忆
    - skills:  Skills 能力
    - mcp:     MCP 客户端
    - trace:   链路追踪
    - cli:     CLI 界面
    - cache:   缓存
    """
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
        """
        从字典/映射创建 AppConfig 实例。

        执行步骤：
        1. 规范化键名（处理别名映射）
        2. 检查是否有未知的 section
        3. 逐 section 解析为对应的 Settings dataclass
        4. 组装成 AppConfig
        """
        raw = _normalize_mapping(value)
        _ensure_known_sections(raw)  # 校验不存在未知配置段
        paths = _path_settings(raw.get("paths", {}))
        agent = _agent_settings(raw.get("agent", {}))
        retrieval = _retrieval_settings(raw.get("retrieval", {}))
        llm = _llm_settings(raw.get("llm", {}))
        memory = _memory_settings(raw.get("memory", {}))
        skills = _skills_settings(raw.get("skills", {}))
        mcp = _mcp_settings(raw.get("mcp", {}))
        trace = _trace_settings(raw.get("trace", {}))
        cli = _cli_settings(raw.get("cli", {}))
        cache = _cache_settings(raw.get("cache", {}))
        return cls(
            paths=paths,
            agent=agent,
            retrieval=retrieval,
            llm=llm,
            memory=memory,
            skills=skills,
            mcp=mcp,
            trace=trace,
            cli=cli,
            cache=cache,
        )

    def to_dict(self) -> dict[str, Any]:
        """将 AppConfig 序列化为嵌套字典（使用 dataclasses.asdict）。"""
        return asdict(self)

    def to_runtime_kwargs(self) -> dict[str, Any]:
        """
        将配置转换为运行时可用的扁平化 kwargs 字典。

        此方法是配置系统和运行时组件之间的桥梁：
        - 处理 LLM provider/model 的继承逻辑（子配置未指定时回退到 agent 配置）
        - 将嵌套的 Settings 对象展平为扁平的键值对
        - 确保 dict 类型的值通过 dict() 复制，避免共享引用

        返回值被 RAGService、Agent 等运行时组件直接消费。
        """
        # 查询改写 LLM 配置：优先用 llm.rewrite_*，未指定则回退到 agent 配置
        rewrite_provider = self.llm.rewrite_provider or self.agent.provider
        rewrite_model = self.llm.rewrite_model or self.agent.model
        rewrite_kwargs = (
            self.agent.model_kwargs
            if (
                self.llm.rewrite_provider is None
                and not self.llm.rewrite_kwargs  # llm 层也未配置 kwargs 时才用 agent 的
            )
            else self.llm.rewrite_kwargs
        )
        # 记忆提取 LLM 配置：逻辑同查询改写
        memory_provider = self.llm.memory_provider or self.agent.provider
        memory_model = self.llm.memory_model or self.agent.model
        memory_kwargs = (
            self.agent.model_kwargs
            if (
                self.llm.memory_provider is None
                and not self.llm.memory_kwargs
            )
            else self.llm.memory_kwargs
        )
        return {
            # ── 检索 ──
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
            # ── Agent & 用户 ──
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
            # ── 追踪 & 日志 ──
            "trace_enabled": self.trace.enabled,
            "live_events_enabled": self.trace.live,
            "trace_dir": self.paths.trace_dir,
            "show_config": self.cli.show_config,
            "stream_output_enabled": self.cli.stream_output,
            # ── LLM 重试 ──
            "llm_retry_attempts": self.llm.retry_attempts,
            "llm_timeout_s": self.llm.timeout_s,
            "llm_retry_backoff_s": self.llm.retry_backoff_s,
            # ── Agent 流程控制 ──
            "max_tool_rounds": self.agent.max_tool_rounds,
            "max_repeated_tool_calls": self.agent.max_repeated_tool_calls,
            "reflection_enabled": self.agent.reflection_enabled,
            # ── 数据路径 ──
            "data_dir": self.paths.data_dir,
            "memory_dir": self.paths.memory_dir,
            # ── Agent LLM ──
            "agent_provider": self.agent.provider,
            "agent_model_name": self.agent.model,
            "agent_model_kwargs": dict(self.agent.model_kwargs),
            # ── 缓存 ──
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
# 公共 API 函数
# ═══════════════════════════════════════════════════════════════


def load_app_config(
    config_path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> AppConfig:
    """
    加载应用配置，应用四层叠加优先级（从低到高）：

    1. 默认值 (AppConfig 的 default_factory)
    2. 配置文件 (JSON 或 TOML，路径由 config_path 或环境变量 RAG_SERVER_CONFIG 指定)
    3. 环境变量覆盖 (RAG_SERVER_* 前缀的环境变量)
    4. 代码级别覆盖 (overrides 参数)

    层级越高，优先级越高，即后面会覆盖前面的值。
    """
    # 使用传入的 env 或默认的 os.environ
    actual_env = os.environ if env is None else env
    # 确定配置文件路径：参数 > 环境变量 RAG_SERVER_CONFIG
    selected_config_path = config_path or actual_env.get("RAG_SERVER_CONFIG")
    # 从默认值开始构建合并后的配置
    merged: dict[str, Any] = AppConfig().to_dict()

    # 第 2 层：加载配置文件并深层合并
    if selected_config_path:
        loaded = load_config_file(selected_config_path)
        merged = _deep_merge(merged, loaded)

    # 第 3 层：环境变量覆盖并深层合并
    env_overrides = build_env_overrides(actual_env)
    if env_overrides:
        merged = _deep_merge(merged, env_overrides)

    # 第 4 层：代码级别覆盖（最优先）
    if overrides:
        merged = _deep_merge(merged, _normalize_mapping(overrides))

    return AppConfig.from_mapping(merged)


def load_config_file(config_path: str | Path) -> dict[str, Any]:
    """
    加载并解析配置文件。

    支持两种格式：
    - .json: JSON 格式
    - .toml / .tml: TOML 格式
    """
    # 将路径转为 Path 对象，并展开 ~ 等用户目录符号
    path = Path(config_path).expanduser()
    # 检查文件是否存在
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    if not path.is_file():
        raise ConfigError(f"Config path is not a file: {path}")

    suffix = path.suffix.lower()
    try:
        # 根据后缀选择解析器
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

    # 配置文件顶层必须是对象/字典
    if not isinstance(payload, dict):
        raise ConfigError("Config file must contain an object")
    return _normalize_mapping(payload)


def build_env_overrides(env: Mapping[str, str]) -> dict[str, Any]:
    """
    从环境变量构建配置覆盖字典。

    映射规则：环境变量名 -> (配置段名, 键名)
    例如: RAG_SERVER_DATA_DIR -> ("paths", "data_dir")
          RAG_SERVER_BM25 -> ("retrieval", "bm25")
          RAG_SERVER_CACHE -> ("cache", "enabled")

    注意：同一字段有多个环境变量别名（旧名称的兼容项），
    后出现的可能会覆盖先出现的（取决于 dict 遍历顺序），
    但 setdefault 保证第一次设置的值不会被后续同名 section 覆盖。
    """
    mapping = {
        # ── Paths ──
        "RAG_SERVER_DATA_DIR": ("paths", "data_dir"),
        "RAG_SERVER_MEMORY_DIR": ("paths", "memory_dir"),
        "RAG_SERVER_TRACE_DIR": ("paths", "trace_dir"),
        "RAG_SERVER_MCP_CONFIG": ("paths", "mcp_config_path"),
        # ── Agent ──
        "RAG_SERVER_AGENT_PROVIDER": ("agent", "provider"),
        "RAG_SERVER_CHAT_PROVIDER": ("agent", "provider"),  # 别名兼容
        "RAG_SERVER_AGENT_MODEL": ("agent", "model"),
        "RAG_SERVER_CHAT_MODEL": ("agent", "model"),        # 别名兼容
        "RAG_SERVER_AGENT_MODEL_KWARGS": ("agent", "model_kwargs"),
        "RAG_SERVER_CHAT_MODEL_KWARGS": ("agent", "model_kwargs"),  # 别名兼容
        "RAG_SERVER_USER_ID": ("agent", "user_id"),
        "RAG_SERVER_MAX_TOOL_ROUNDS": ("agent", "max_tool_rounds"),
        "RAG_SERVER_MAX_REPEATED_TOOL_CALLS": (
            "agent",
            "max_repeated_tool_calls",
        ),
        "RAG_SERVER_REFLECTION": ("agent", "reflection_enabled"),
        # ── Retrieval ──
        "RAG_SERVER_QUERY_REWRITE": ("retrieval", "query_rewrite"),
        "RAG_SERVER_BM25": ("retrieval", "bm25"),
        "RAG_SERVER_CROSS_ENCODER": ("retrieval", "cross_encoder"),
        "RAG_SERVER_EMBEDDING_PROVIDER": ("retrieval", "embedding_provider"),
        "RAG_SERVER_EMBEDDING_MODEL": ("retrieval", "embedding_model"),
        "RAG_SERVER_EMBEDDING_KWARGS": ("retrieval", "embedding_kwargs"),
        "RAG_SERVER_EMBEDDING_MODEL_KWARGS": ("retrieval", "embedding_kwargs"),  # 别名兼容
        "RAG_SERVER_RERANKER_PROVIDER": ("retrieval", "reranker_provider"),
        "RAG_SERVER_RERANKER_MODEL": ("retrieval", "reranker_model"),
        "RAG_SERVER_RERANKER_KWARGS": ("retrieval", "reranker_kwargs"),
        "RAG_SERVER_RERANKER_MODEL_KWARGS": ("retrieval", "reranker_kwargs"),  # 别名兼容
        "RAG_SERVER_RERANKER_DEVICE": ("retrieval", "reranker_device"),
        "RAG_SERVER_RERANKER_BATCH_SIZE": ("retrieval", "reranker_batch_size"),
        # ── Rerank 旧命名兼容 ──
        "RAG_SERVER_RERANK_PROVIDER": ("retrieval", "reranker_provider"),
        "RAG_SERVER_RERANK_MODEL": ("retrieval", "reranker_model"),
        "RAG_SERVER_RERANK_KWARGS": ("retrieval", "reranker_kwargs"),
        "RAG_SERVER_RERANK_MODEL_KWARGS": ("retrieval", "reranker_kwargs"),
        "RAG_SERVER_RERANK_DEVICE": ("retrieval", "reranker_device"),
        "RAG_SERVER_RERANK_BATCH_SIZE": ("retrieval", "reranker_batch_size"),
        # ── LLM 查询改写 ──
        "RAG_SERVER_QUERY_REWRITE_PROVIDER": ("llm", "rewrite_provider"),
        "RAG_SERVER_QUERY_REWRITE_MODEL": ("llm", "rewrite_model"),
        "RAG_SERVER_QUERY_REWRITE_KWARGS": ("llm", "rewrite_kwargs"),
        "RAG_SERVER_QUERY_REWRITE_MODEL_KWARGS": ("llm", "rewrite_kwargs"),  # 别名兼容
        "RAG_SERVER_REWRITE_PROVIDER": ("llm", "rewrite_provider"),  # 短别名
        "RAG_SERVER_REWRITE_MODEL": ("llm", "rewrite_model"),
        "RAG_SERVER_REWRITE_KWARGS": ("llm", "rewrite_kwargs"),
        "RAG_SERVER_REWRITE_MODEL_KWARGS": ("llm", "rewrite_kwargs"),
        # ── LLM 记忆 ──
        "RAG_SERVER_MEMORY_PROVIDER": ("llm", "memory_provider"),
        "RAG_SERVER_MEMORY_MODEL": ("llm", "memory_model"),
        "RAG_SERVER_MEMORY_KWARGS": ("llm", "memory_kwargs"),
        "RAG_SERVER_MEMORY_MODEL_KWARGS": ("llm", "memory_kwargs"),  # 别名兼容
        # ── LLM 重试 ──
        "RAG_SERVER_LLM_RETRY_ATTEMPTS": ("llm", "retry_attempts"),
        "RAG_SERVER_LLM_TIMEOUT": ("llm", "timeout_s"),
        "RAG_SERVER_LLM_RETRY_BACKOFF": ("llm", "retry_backoff_s"),
        # ── Memory ──
        "RAG_SERVER_MEMORY": ("memory", "enabled"),
        "RAG_SERVER_MEMORY_TOP_K": ("memory", "top_k"),
        # ── Skills ──
        "RAG_SERVER_SKILLS": ("skills", "enabled"),
        "RAG_SERVER_SKILLS_DIRS": ("skills", "dirs"),
        # ── MCP ──
        "RAG_SERVER_MCP": ("mcp", "enabled"),
        # ── Trace ──
        "RAG_SERVER_TRACE": ("trace", "enabled"),
        "RAG_SERVER_LIVE_EVENTS": ("trace", "live"),
        "RAG_SERVER_LIVE_LOGS": ("trace", "live"),  # 别名兼容
        "RAG_SERVER_CLI_LIVE_EVENTS": ("trace", "live"),  # 别名兼容
        "RAG_SERVER_CLI_LIVE_LOGS": ("trace", "live"),    # 别名兼容
        # ── CLI ──
        "RAG_SERVER_SHOW_CONFIG": ("cli", "show_config"),
        "RAG_SERVER_CLI_SHOW_CONFIG": ("cli", "show_config"),  # 别名兼容
        "RAG_SERVER_CLI_CONFIG_OUTPUT": ("cli", "show_config"),  # 别名兼容
        "RAG_SERVER_STREAM_OUTPUT": ("cli", "stream_output"),
        "RAG_SERVER_CLI_STREAM_OUTPUT": ("cli", "stream_output"),  # 别名兼容
        # ── Cache ──
        "RAG_SERVER_CACHE": ("cache", "enabled"),
        "RAG_SERVER_CACHE_ENABLED": ("cache", "enabled"),  # 别名兼容
        "RAG_SERVER_REDIS_URL": ("cache", "redis_url"),
        "RAG_SERVER_CACHE_REDIS_URL": ("cache", "redis_url"),  # 别名兼容
        "RAG_SERVER_CACHE_NAMESPACE": ("cache", "namespace"),
        "RAG_SERVER_CACHE_SOCKET_TIMEOUT": ("cache", "socket_timeout_s"),
        "RAG_SERVER_CACHE_QUERY_REWRITE_TTL": ("cache", "query_rewrite_ttl_s"),
        "RAG_SERVER_CACHE_EMBEDDING_TTL": ("cache", "embedding_ttl_s"),
        "RAG_SERVER_CACHE_RETRIEVAL_TTL": ("cache", "retrieval_ttl_s"),
        "RAG_SERVER_CACHE_RERANK_TTL": ("cache", "rerank_ttl_s"),
        "RAG_SERVER_CACHE_MEMORY_TTL": ("cache", "memory_ttl_s"),
    }

    result: dict[str, Any] = {}
    for env_name, path in mapping.items():
        # 跳过未设置的环境变量
        if env_name not in env:
            continue
        section, key = path
        # 使用 setdefault 确保同一 section 的键被收集在一起
        result.setdefault(section, {})[key] = env[env_name]
    return result


# ═══════════════════════════════════════════════════════════════
# 内部辅助函数：各 Section 的配置解析
# 每个 _xxx_settings 函数负责：
#   1. 提取对应 section 的原始数据
#   2. 校验是否存在未知键
#   3. 逐个字段类型转换/校验后返回 Settings dataclass
# ═══════════════════════════════════════════════════════════════


def _path_settings(raw: Any) -> PathSettings:
    """解析 [paths] 配置段。"""
    section = _section(raw, "paths")
    _ensure_known_keys(
        "paths",
        section,
        {"data_dir", "memory_dir", "trace_dir", "mcp_config_path"},
    )
    default = PathSettings()
    return PathSettings(
        data_dir=_non_empty_str(section.get("data_dir", default.data_dir), "paths.data_dir"),
        memory_dir=_non_empty_str(
            section.get("memory_dir", default.memory_dir),
            "paths.memory_dir",
        ),
        trace_dir=_non_empty_str(
            section.get("trace_dir", default.trace_dir),
            "paths.trace_dir",
        ),
        mcp_config_path=_non_empty_str(
            section.get("mcp_config_path", default.mcp_config_path),
            "paths.mcp_config_path",
        ),
    )


def _agent_settings(raw: Any) -> AgentSettings:
    """解析 [agent] 配置段。"""
    section = _section(raw, "agent")
    _ensure_known_keys(
        "agent",
        section,
        {
            "provider",
            "model",
            "model_kwargs",
            "user_id",
            "max_tool_rounds",
            "max_repeated_tool_calls",
            "reflection_enabled",
        },
    )
    default = AgentSettings()
    return AgentSettings(
        provider=_non_empty_str(
            section.get("provider", default.provider),
            "agent.provider",
        ),
        model=_non_empty_str(section.get("model", default.model), "agent.model"),
        model_kwargs=_coerce_mapping(
            section.get("model_kwargs", default.model_kwargs),
            "agent.model_kwargs",
        ),
        user_id=_non_empty_str(section.get("user_id", default.user_id), "agent.user_id"),
        max_tool_rounds=_coerce_int(
            section.get("max_tool_rounds", default.max_tool_rounds),
            "agent.max_tool_rounds",
            minimum=0,  # 0 表示不限制轮次
        ),
        max_repeated_tool_calls=_coerce_int(
            section.get(
                "max_repeated_tool_calls",
                default.max_repeated_tool_calls,
            ),
            "agent.max_repeated_tool_calls",
            minimum=1,  # 至少允许 1 次重复调用
        ),
        reflection_enabled=_coerce_bool(
            section.get("reflection_enabled", default.reflection_enabled),
            "agent.reflection_enabled",
        ),
    )


def _retrieval_settings(raw: Any) -> RetrievalSettings:
    """解析 [retrieval] 配置段。"""
    section = _section(raw, "retrieval")
    _ensure_known_keys(
        "retrieval",
        section,
        {
            "query_rewrite",
            "bm25",
            "cross_encoder",
            "embedding_provider",
            "embedding_model",
            "embedding_kwargs",
            "reranker_provider",
            "reranker_model",
            "reranker_kwargs",
            "reranker_device",
            "reranker_batch_size",
        },
    )
    default = RetrievalSettings()
    # 读取查询改写模式并校验其值必须在允许的范围内
    query_rewrite = _non_empty_str(
        section.get("query_rewrite", default.query_rewrite),
        "retrieval.query_rewrite",
    )
    if query_rewrite not in QUERY_REWRITE_MODES:
        allowed = ", ".join(QUERY_REWRITE_MODES)
        raise ConfigError(f"retrieval.query_rewrite must be one of: {allowed}")
    return RetrievalSettings(
        query_rewrite=query_rewrite,
        bm25=_coerce_bool(section.get("bm25", default.bm25), "retrieval.bm25"),
        cross_encoder=_coerce_bool(
            section.get("cross_encoder", default.cross_encoder),
            "retrieval.cross_encoder",
        ),
        embedding_provider=_non_empty_str(
            section.get("embedding_provider", default.embedding_provider),
            "retrieval.embedding_provider",
        ),
        embedding_model=_non_empty_str(
            section.get("embedding_model", default.embedding_model),
            "retrieval.embedding_model",
        ),
        embedding_kwargs=_coerce_mapping(
            section.get("embedding_kwargs", default.embedding_kwargs),
            "retrieval.embedding_kwargs",
        ),
        reranker_provider=_non_empty_str(
            section.get("reranker_provider", default.reranker_provider),
            "retrieval.reranker_provider",
        ),
        reranker_model=_non_empty_str(
            section.get("reranker_model", default.reranker_model),
            "retrieval.reranker_model",
        ),
        reranker_kwargs=_coerce_mapping(
            section.get("reranker_kwargs", default.reranker_kwargs),
            "retrieval.reranker_kwargs",
        ),
        reranker_device=_optional_str(
            section.get("reranker_device", default.reranker_device)
        ),
        reranker_batch_size=_coerce_int(
            section.get("reranker_batch_size", default.reranker_batch_size),
            "retrieval.reranker_batch_size",
            minimum=1,  # 批处理大小至少为 1
        ),
    )


def _llm_settings(raw: Any) -> LLMSettings:
    """解析 [llm] 配置段。"""
    section = _section(raw, "llm")
    _ensure_known_keys(
        "llm",
        section,
        {
            "rewrite_provider",
            "rewrite_model",
            "rewrite_kwargs",
            "memory_provider",
            "memory_model",
            "memory_kwargs",
            "retry_attempts",
            "timeout_s",
            "retry_backoff_s",
        },
    )
    default = LLMSettings()
    return LLMSettings(
        rewrite_provider=_optional_str(
            section.get("rewrite_provider", default.rewrite_provider)
        ),
        rewrite_model=_optional_str(section.get("rewrite_model", default.rewrite_model)),
        rewrite_kwargs=_coerce_mapping(
            section.get("rewrite_kwargs", default.rewrite_kwargs),
            "llm.rewrite_kwargs",
        ),
        memory_provider=_optional_str(
            section.get("memory_provider", default.memory_provider)
        ),
        memory_model=_optional_str(section.get("memory_model", default.memory_model)),
        memory_kwargs=_coerce_mapping(
            section.get("memory_kwargs", default.memory_kwargs),
            "llm.memory_kwargs",
        ),
        retry_attempts=_coerce_int(
            section.get("retry_attempts", default.retry_attempts),
            "llm.retry_attempts",
            minimum=1,  # 至少重试 1 次
        ),
        timeout_s=_coerce_optional_float(
            section.get("timeout_s", default.timeout_s),
            "llm.timeout_s",
        ),
        retry_backoff_s=_coerce_float(
            section.get("retry_backoff_s", default.retry_backoff_s),
            "llm.retry_backoff_s",
            minimum=0.0,  # 退避时间最小为 0
        ),
    )


def _memory_settings(raw: Any) -> MemorySettings:
    """解析 [memory] 配置段。"""
    section = _section(raw, "memory")
    _ensure_known_keys("memory", section, {"enabled", "top_k"})
    default = MemorySettings()
    return MemorySettings(
        enabled=_coerce_bool(section.get("enabled", default.enabled), "memory.enabled"),
        top_k=_coerce_int(section.get("top_k", default.top_k), "memory.top_k", minimum=1),
    )


def _skills_settings(raw: Any) -> SkillsSettings:
    """解析 [skills] 配置段。"""
    section = _section(raw, "skills")
    _ensure_known_keys("skills", section, {"enabled", "dirs"})
    default = SkillsSettings()
    return SkillsSettings(
        enabled=_coerce_bool(section.get("enabled", default.enabled), "skills.enabled"),
        dirs=_coerce_str_list(section.get("dirs", default.dirs), "skills.dirs"),
    )


def _mcp_settings(raw: Any) -> MCPSettings:
    """解析 [mcp] 配置段。"""
    section = _section(raw, "mcp")
    _ensure_known_keys("mcp", section, {"enabled"})
    default = MCPSettings()
    return MCPSettings(
        enabled=_coerce_bool(section.get("enabled", default.enabled), "mcp.enabled"),
    )


def _trace_settings(raw: Any) -> TraceSettings:
    """解析 [trace] 配置段。"""
    section = _section(raw, "trace")
    _ensure_known_keys("trace", section, {"enabled", "live"})
    default = TraceSettings()
    return TraceSettings(
        enabled=_coerce_bool(section.get("enabled", default.enabled), "trace.enabled"),
        live=_coerce_bool(section.get("live", default.live), "trace.live"),
    )


def _cli_settings(raw: Any) -> CLISettings:
    """解析 [cli] 配置段。"""
    section = _section(raw, "cli")
    _ensure_known_keys("cli", section, {"show_config", "stream_output"})
    default = CLISettings()
    return CLISettings(
        show_config=_coerce_bool(
            section.get("show_config", default.show_config),
            "cli.show_config",
        ),
        stream_output=_coerce_bool(
            section.get("stream_output", default.stream_output),
            "cli.stream_output",
        ),
    )


def _cache_settings(raw: Any) -> CacheSettings:
    """解析 [cache] 配置段。"""
    section = _section(raw, "cache")
    _ensure_known_keys(
        "cache",
        section,
        {
            "enabled",
            "redis_url",
            "namespace",
            "socket_timeout_s",
            "query_rewrite_ttl_s",
            "embedding_ttl_s",
            "retrieval_ttl_s",
            "rerank_ttl_s",
            "memory_ttl_s",
        },
    )
    default = CacheSettings()
    return CacheSettings(
        enabled=_coerce_bool(section.get("enabled", default.enabled), "cache.enabled"),
        redis_url=_non_empty_str(
            section.get("redis_url", default.redis_url),
            "cache.redis_url",
        ),
        namespace=_non_empty_str(
            section.get("namespace", default.namespace),
            "cache.namespace",
        ),
        socket_timeout_s=_coerce_float(
            section.get("socket_timeout_s", default.socket_timeout_s),
            "cache.socket_timeout_s",
            minimum=0.0,
        ),
        query_rewrite_ttl_s=_coerce_int(
            section.get("query_rewrite_ttl_s", default.query_rewrite_ttl_s),
            "cache.query_rewrite_ttl_s",
            minimum=0,  # 0 表示永不过期
        ),
        embedding_ttl_s=_coerce_int(
            section.get("embedding_ttl_s", default.embedding_ttl_s),
            "cache.embedding_ttl_s",
            minimum=0,
        ),
        retrieval_ttl_s=_coerce_int(
            section.get("retrieval_ttl_s", default.retrieval_ttl_s),
            "cache.retrieval_ttl_s",
            minimum=0,
        ),
        rerank_ttl_s=_coerce_int(
            section.get("rerank_ttl_s", default.rerank_ttl_s),
            "cache.rerank_ttl_s",
            minimum=0,
        ),
        memory_ttl_s=_coerce_int(
            section.get("memory_ttl_s", default.memory_ttl_s),
            "cache.memory_ttl_s",
            minimum=0,
        ),
    )


# ═══════════════════════════════════════════════════════════════
# 内部辅助函数：通用工具函数
# ═══════════════════════════════════════════════════════════════


def _normalize_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """
    规范化配置字典：处理键名别名映射。

    支持以下别名/旧名称到新名称的转换：
    - agent.chat_provider -> agent.provider
    - retrieval.bm25_enabled -> retrieval.bm25
    - retrieval.rerank_provider -> retrieval.reranker_provider
    - cli.streaming -> cli.stream_output
    - 等等...

    此外，也会将跨 section 的别名移动到正确的 section 下，
    例如 cli.live_events 会被重定向到 trace.live。
    """
    # 别名映射表: (section, old_key) -> (target_section, target_key)
    aliases = {
        # ── Paths 别名 ──
        ("paths", "mcp_config"): ("paths", "mcp_config_path"),
        # ── Agent 别名 ──
        ("agent", "chat_provider"): ("agent", "provider"),
        ("agent", "chat_model"): ("agent", "model"),
        ("agent", "chat_model_kwargs"): ("agent", "model_kwargs"),
        ("agent", "reflection"): ("agent", "reflection_enabled"),
        # ── Retrieval 别名 ──
        ("retrieval", "query_rewrite_mode"): ("retrieval", "query_rewrite"),
        ("retrieval", "bm25_enabled"): ("retrieval", "bm25"),
        ("retrieval", "cross_encoder_enabled"): ("retrieval", "cross_encoder"),
        ("retrieval", "embedding_model_name"): ("retrieval", "embedding_model"),
        ("retrieval", "embedding_model_kwargs"): ("retrieval", "embedding_kwargs"),
        ("retrieval", "rerank_provider"): ("retrieval", "reranker_provider"),
        ("retrieval", "reranker_model_name"): ("retrieval", "reranker_model"),
        ("retrieval", "rerank_model"): ("retrieval", "reranker_model"),
        ("retrieval", "rerank_model_name"): ("retrieval", "reranker_model"),
        ("retrieval", "reranker_model_kwargs"): ("retrieval", "reranker_kwargs"),
        ("retrieval", "rerank_kwargs"): ("retrieval", "reranker_kwargs"),
        ("retrieval", "rerank_model_kwargs"): ("retrieval", "reranker_kwargs"),
        ("retrieval", "rerank_device"): ("retrieval", "reranker_device"),
        ("retrieval", "rerank_batch_size"): ("retrieval", "reranker_batch_size"),
        # ── LLM 别名 ──
        ("llm", "query_rewrite_provider"): ("llm", "rewrite_provider"),
        ("llm", "query_rewrite_model"): ("llm", "rewrite_model"),
        ("llm", "query_rewrite_kwargs"): ("llm", "rewrite_kwargs"),
        ("llm", "llm_retry_attempts"): ("llm", "retry_attempts"),
        ("llm", "llm_timeout_s"): ("llm", "timeout_s"),
        ("llm", "llm_retry_backoff_s"): ("llm", "retry_backoff_s"),
        # ── Memory 别名 ──
        ("memory", "memory_top_k"): ("memory", "top_k"),
        # ── Trace 别名 ──
        ("trace", "live_events"): ("trace", "live"),
        ("trace", "live_logs"): ("trace", "live"),
        # ── CLI 别名 ──
        ("cli", "startup_config"): ("cli", "show_config"),
        ("cli", "show_startup_config"): ("cli", "show_config"),
        ("cli", "config_output"): ("cli", "show_config"),
        ("cli", "streaming"): ("cli", "stream_output"),
        ("cli", "stream_output_enabled"): ("cli", "stream_output"),
        ("cli", "live_events"): ("trace", "live"),  # 跨 section 别名
        ("cli", "live_logs"): ("trace", "live"),    # 跨 section 别名
        # ── Cache 别名 ──
        ("cache", "url"): ("cache", "redis_url"),
        ("cache", "ttl_query_rewrite_s"): ("cache", "query_rewrite_ttl_s"),
        ("cache", "ttl_embedding_s"): ("cache", "embedding_ttl_s"),
        ("cache", "ttl_retrieval_s"): ("cache", "retrieval_ttl_s"),
        ("cache", "ttl_rerank_s"): ("cache", "rerank_ttl_s"),
        ("cache", "ttl_memory_s"): ("cache", "memory_ttl_s"),
    }
    normalized: dict[str, Any] = {}
    for section_name, raw_section in value.items():
        # 如果 section 的值不是字典，直接保留原值
        if not isinstance(raw_section, Mapping):
            normalized[str(section_name)] = raw_section
            continue
        normalized_section: dict[str, Any] = {}
        for key, item in raw_section.items():
            # 查找别名映射，决定键的目标 section 和 key
            target_section, target_key = aliases.get(
                (str(section_name), str(key)),
                (str(section_name), str(key)),  # 未匹配别名则保持原样
            )
            if target_section != str(section_name):
                # 跨 section 别名：将值放到目标 section 下
                normalized.setdefault(target_section, {})[target_key] = item
            else:
                # 同 section 别名或非别名：放入当前 section
                normalized_section[target_key] = item
        # 如果该 section 已有跨 section 别名合并过来的值，进行深层合并
        existing = normalized.get(str(section_name))
        if isinstance(existing, Mapping):
            normalized[str(section_name)] = _deep_merge(existing, normalized_section)
        else:
            normalized[str(section_name)] = normalized_section
    return normalized


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """
    深层合并两个字典。

    规则：
    - 如果 base 和 overlay 同一键的值都是字典，则递归合并
    - 否则 overlay 的值完全覆盖 base 的值
    """
    merged = {str(key): value for key, value in base.items()}
    for key, value in overlay.items():
        key = str(key)
        # 双方都是字典时递归合并
        if (
            isinstance(value, Mapping)
            and isinstance(merged.get(key), Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            # 否则直接覆盖
            merged[key] = value
    return merged


# ═══════════════════════════════════════════════════════════════
# 内部辅助函数：类型校验与转换（Coercion Helpers）
# 每个函数负责从原始输入提取并验证特定类型的值
# ═══════════════════════════════════════════════════════════════


def _section(raw: Any, name: str) -> dict[str, Any]:
    """提取并验证配置段必须是字典类型。

    Args:
        raw: 原始输入值
        name: 配置段名称（用于错误信息）

    Returns:
        规范化后的字典（所有键转为字符串）

    Raises:
        ConfigError: 如果 raw 不是 Mapping 类型
    """
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{name} must be an object")
    return {str(key): value for key, value in raw.items()}


def _ensure_known_sections(raw: Mapping[str, Any]) -> None:
    """检查配置文件顶层是否包含未知的 section 名称。

    防止用户因拼写错误导致配置被静默忽略。
    """
    allowed = {
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
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"Unknown config section(s): {', '.join(unknown)}")


def _ensure_known_keys(section_name: str, raw: Mapping[str, Any], allowed: set[str]) -> None:
    """检查配置段中是否包含未知的键名。

    Args:
        section_name: 配置段名称（用于错误信息）
        raw: 该段的原始字典
        allowed: 允许的键名集合

    Raises:
        ConfigError: 如果存在未知键
    """
    unknown = sorted(set(raw) - allowed)
    if unknown:
        keys = ", ".join(unknown)
        raise ConfigError(f"Unknown key(s) in {section_name}: {keys}")


def _non_empty_str(value: Any, field_name: str) -> str:
    """校验并返回非空字符串。

    Raises:
        ConfigError: 如果值为空或仅包含空白字符
    """
    text = str(value).strip()
    if not text:
        raise ConfigError(f"{field_name} must not be empty")
    return text


def _optional_str(value: Any) -> str | None:
    """将值转为可选字符串，None 或空串返回 None。"""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_bool(value: Any, field_name: str) -> bool:
    """将多种格式的布尔值统一转为 Python bool。

    支持的格式（不区分大小写）:
    - True: true, yes, on, 1
    - False: false, no, off, 0

    Raises:
        ConfigError: 如果值无法解析为布尔值
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
    raise ConfigError(f"{field_name} must be a boolean")


def _coerce_int(value: Any, field_name: str, *, minimum: int) -> int:
    """校验并返回整数，同时检查最小值约束。

    Args:
        value: 待转换的值
        field_name: 字段名（用于错误信息）
        minimum: 允许的最小值（含）

    Raises:
        ConfigError: 值不是整数或小于最小值
    """
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{field_name} must be an integer") from error
    if number < minimum:
        raise ConfigError(f"{field_name} must be >= {minimum}")
    return number


def _coerce_float(value: Any, field_name: str, *, minimum: float) -> float:
    """校验并返回浮点数，同时检查最小值约束。

    Args:
        value: 待转换的值
        field_name: 字段名（用于错误信息）
        minimum: 允许的最小值（含）

    Raises:
        ConfigError: 值不是数字或小于最小值
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{field_name} must be a number") from error
    if number < minimum:
        raise ConfigError(f"{field_name} must be >= {minimum:g}")
    return number


def _coerce_optional_float(value: Any, field_name: str) -> float | None:
    """将值转为可选浮点数。

    None 或 <= 0 的值返回 None（表示不限制超时时间）。
    """
    if value is None:
        return None
    number = _coerce_float(value, field_name, minimum=0.0)
    if number <= 0:
        return None  # <= 0 视为未设置超时
    return number


def _coerce_str_list(value: Any, field_name: str) -> list[str]:
    """将值转为字符串列表。

    支持输入格式：
    - None -> []
    - 逗号分隔字符串 "a,b,c" -> ["a", "b", "c"]
    - 列表 ["a", "b"] -> ["a", "b"]

    Raises:
        ConfigError: 如果值不是字符串或列表
    """
    if value is None:
        return []
    if isinstance(value, str):
        # 字符串按逗号分割
        items = value.split(",")
    elif isinstance(value, list | tuple):
        items = list(value)
    else:
        raise ConfigError(f"{field_name} must be a list of strings")
    # 去除空白和空字符串
    return [str(item).strip() for item in items if str(item).strip()]


def _coerce_mapping(value: Any, field_name: str) -> dict[str, Any]:
    """将值转为字符串键的字典。

    支持输入格式：
    - None or 空串 -> {}
    - JSON 字符串 -> 解析后的字典
    - dict/其他 Mapping -> 规范化后的字典

    Raises:
        ConfigError: 如果不是有效的对象
    """
    if value is None:
        return {}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            # JSON 字符串需要解析
            value = json.loads(text)
        except json.JSONDecodeError as error:
            raise ConfigError(f"{field_name} must be a JSON object") from error
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field_name} must be an object")
    # 返回键名转为字符串的干净字典
    return {str(key): item for key, item in value.items()}
