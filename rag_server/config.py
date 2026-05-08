from __future__ import annotations

import json
import os
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .model_factory import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHAT_PROVIDER,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANKER_PROVIDER,
)

DEFAULT_AGENT_MODEL = DEFAULT_CHAT_MODEL
DEFAULT_USER_ID = "default_user"
DEFAULT_QUERY_REWRITE_MODE = "on"
QUERY_REWRITE_MODES = ("on", "off", "rewrite_only", "multi_query")
DEFAULT_MAX_TOOL_ROUNDS = 6
DEFAULT_MAX_REPEATED_TOOL_CALLS = 2
DEFAULT_REFLECTION_ENABLED = True
DEFAULT_DATA_DIR = "data"
DEFAULT_MEMORY_DIR = "memory"
DEFAULT_TRACE_DIR = "traces"
DEFAULT_MCP_CONFIG_PATH = "mcp_servers.json"
DEFAULT_LIVE_EVENTS_ENABLED = True
DEFAULT_CLI_CONFIG_OUTPUT_ENABLED = True


class ConfigError(ValueError):
    """Raised when the application configuration is invalid."""


@dataclass(frozen=True)
class PathSettings:
    data_dir: str = DEFAULT_DATA_DIR
    memory_dir: str = DEFAULT_MEMORY_DIR
    trace_dir: str = DEFAULT_TRACE_DIR
    mcp_config_path: str = DEFAULT_MCP_CONFIG_PATH


@dataclass(frozen=True)
class AgentSettings:
    provider: str = DEFAULT_CHAT_PROVIDER
    model: str = DEFAULT_AGENT_MODEL
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    user_id: str = DEFAULT_USER_ID
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS
    max_repeated_tool_calls: int = DEFAULT_MAX_REPEATED_TOOL_CALLS
    reflection_enabled: bool = DEFAULT_REFLECTION_ENABLED


@dataclass(frozen=True)
class RetrievalSettings:
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
    enabled: bool = True
    top_k: int = 5


@dataclass(frozen=True)
class SkillsSettings:
    enabled: bool = True
    dirs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MCPSettings:
    enabled: bool = False


@dataclass(frozen=True)
class TraceSettings:
    enabled: bool = False
    live: bool = DEFAULT_LIVE_EVENTS_ENABLED


@dataclass(frozen=True)
class CLISettings:
    show_config: bool = DEFAULT_CLI_CONFIG_OUTPUT_ENABLED


@dataclass(frozen=True)
class AppConfig:
    paths: PathSettings = field(default_factory=PathSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    skills: SkillsSettings = field(default_factory=SkillsSettings)
    mcp: MCPSettings = field(default_factory=MCPSettings)
    trace: TraceSettings = field(default_factory=TraceSettings)
    cli: CLISettings = field(default_factory=CLISettings)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AppConfig:
        raw = _normalize_mapping(value)
        _ensure_known_sections(raw)
        paths = _path_settings(raw.get("paths", {}))
        agent = _agent_settings(raw.get("agent", {}))
        retrieval = _retrieval_settings(raw.get("retrieval", {}))
        llm = _llm_settings(raw.get("llm", {}))
        memory = _memory_settings(raw.get("memory", {}))
        skills = _skills_settings(raw.get("skills", {}))
        mcp = _mcp_settings(raw.get("mcp", {}))
        trace = _trace_settings(raw.get("trace", {}))
        cli = _cli_settings(raw.get("cli", {}))
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
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_runtime_kwargs(self) -> dict[str, Any]:
        rewrite_provider = self.llm.rewrite_provider or self.agent.provider
        rewrite_model = self.llm.rewrite_model or self.agent.model
        rewrite_kwargs = (
            self.agent.model_kwargs
            if (
                self.llm.rewrite_provider is None
                and not self.llm.rewrite_kwargs
            )
            else self.llm.rewrite_kwargs
        )
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
            "trace_enabled": self.trace.enabled,
            "live_events_enabled": self.trace.live,
            "trace_dir": self.paths.trace_dir,
            "show_config": self.cli.show_config,
            "llm_retry_attempts": self.llm.retry_attempts,
            "llm_timeout_s": self.llm.timeout_s,
            "llm_retry_backoff_s": self.llm.retry_backoff_s,
            "max_tool_rounds": self.agent.max_tool_rounds,
            "max_repeated_tool_calls": self.agent.max_repeated_tool_calls,
            "reflection_enabled": self.agent.reflection_enabled,
            "data_dir": self.paths.data_dir,
            "memory_dir": self.paths.memory_dir,
            "agent_provider": self.agent.provider,
            "agent_model_name": self.agent.model,
            "agent_model_kwargs": dict(self.agent.model_kwargs),
        }


def load_app_config(
    config_path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> AppConfig:
    """Load the app config using defaults, file, environment, then overrides."""
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
    mapping = {
        "RAG_SERVER_DATA_DIR": ("paths", "data_dir"),
        "RAG_SERVER_MEMORY_DIR": ("paths", "memory_dir"),
        "RAG_SERVER_TRACE_DIR": ("paths", "trace_dir"),
        "RAG_SERVER_MCP_CONFIG": ("paths", "mcp_config_path"),
        "RAG_SERVER_AGENT_PROVIDER": ("agent", "provider"),
        "RAG_SERVER_CHAT_PROVIDER": ("agent", "provider"),
        "RAG_SERVER_AGENT_MODEL": ("agent", "model"),
        "RAG_SERVER_CHAT_MODEL": ("agent", "model"),
        "RAG_SERVER_AGENT_MODEL_KWARGS": ("agent", "model_kwargs"),
        "RAG_SERVER_CHAT_MODEL_KWARGS": ("agent", "model_kwargs"),
        "RAG_SERVER_USER_ID": ("agent", "user_id"),
        "RAG_SERVER_MAX_TOOL_ROUNDS": ("agent", "max_tool_rounds"),
        "RAG_SERVER_MAX_REPEATED_TOOL_CALLS": (
            "agent",
            "max_repeated_tool_calls",
        ),
        "RAG_SERVER_REFLECTION": ("agent", "reflection_enabled"),
        "RAG_SERVER_QUERY_REWRITE": ("retrieval", "query_rewrite"),
        "RAG_SERVER_BM25": ("retrieval", "bm25"),
        "RAG_SERVER_CROSS_ENCODER": ("retrieval", "cross_encoder"),
        "RAG_SERVER_EMBEDDING_PROVIDER": ("retrieval", "embedding_provider"),
        "RAG_SERVER_EMBEDDING_MODEL": ("retrieval", "embedding_model"),
        "RAG_SERVER_EMBEDDING_KWARGS": ("retrieval", "embedding_kwargs"),
        "RAG_SERVER_EMBEDDING_MODEL_KWARGS": ("retrieval", "embedding_kwargs"),
        "RAG_SERVER_RERANKER_PROVIDER": ("retrieval", "reranker_provider"),
        "RAG_SERVER_RERANKER_MODEL": ("retrieval", "reranker_model"),
        "RAG_SERVER_RERANKER_KWARGS": ("retrieval", "reranker_kwargs"),
        "RAG_SERVER_RERANKER_MODEL_KWARGS": ("retrieval", "reranker_kwargs"),
        "RAG_SERVER_RERANKER_DEVICE": ("retrieval", "reranker_device"),
        "RAG_SERVER_RERANKER_BATCH_SIZE": ("retrieval", "reranker_batch_size"),
        "RAG_SERVER_RERANK_PROVIDER": ("retrieval", "reranker_provider"),
        "RAG_SERVER_RERANK_MODEL": ("retrieval", "reranker_model"),
        "RAG_SERVER_RERANK_KWARGS": ("retrieval", "reranker_kwargs"),
        "RAG_SERVER_RERANK_MODEL_KWARGS": ("retrieval", "reranker_kwargs"),
        "RAG_SERVER_RERANK_DEVICE": ("retrieval", "reranker_device"),
        "RAG_SERVER_RERANK_BATCH_SIZE": ("retrieval", "reranker_batch_size"),
        "RAG_SERVER_QUERY_REWRITE_PROVIDER": ("llm", "rewrite_provider"),
        "RAG_SERVER_QUERY_REWRITE_MODEL": ("llm", "rewrite_model"),
        "RAG_SERVER_QUERY_REWRITE_KWARGS": ("llm", "rewrite_kwargs"),
        "RAG_SERVER_QUERY_REWRITE_MODEL_KWARGS": ("llm", "rewrite_kwargs"),
        "RAG_SERVER_REWRITE_PROVIDER": ("llm", "rewrite_provider"),
        "RAG_SERVER_REWRITE_MODEL": ("llm", "rewrite_model"),
        "RAG_SERVER_REWRITE_KWARGS": ("llm", "rewrite_kwargs"),
        "RAG_SERVER_REWRITE_MODEL_KWARGS": ("llm", "rewrite_kwargs"),
        "RAG_SERVER_MEMORY_PROVIDER": ("llm", "memory_provider"),
        "RAG_SERVER_MEMORY_MODEL": ("llm", "memory_model"),
        "RAG_SERVER_MEMORY_KWARGS": ("llm", "memory_kwargs"),
        "RAG_SERVER_MEMORY_MODEL_KWARGS": ("llm", "memory_kwargs"),
        "RAG_SERVER_LLM_RETRY_ATTEMPTS": ("llm", "retry_attempts"),
        "RAG_SERVER_LLM_TIMEOUT": ("llm", "timeout_s"),
        "RAG_SERVER_LLM_RETRY_BACKOFF": ("llm", "retry_backoff_s"),
        "RAG_SERVER_MEMORY": ("memory", "enabled"),
        "RAG_SERVER_MEMORY_TOP_K": ("memory", "top_k"),
        "RAG_SERVER_SKILLS": ("skills", "enabled"),
        "RAG_SERVER_SKILLS_DIRS": ("skills", "dirs"),
        "RAG_SERVER_MCP": ("mcp", "enabled"),
        "RAG_SERVER_TRACE": ("trace", "enabled"),
        "RAG_SERVER_LIVE_EVENTS": ("trace", "live"),
        "RAG_SERVER_LIVE_LOGS": ("trace", "live"),
        "RAG_SERVER_CLI_LIVE_EVENTS": ("trace", "live"),
        "RAG_SERVER_CLI_LIVE_LOGS": ("trace", "live"),
        "RAG_SERVER_SHOW_CONFIG": ("cli", "show_config"),
        "RAG_SERVER_CLI_SHOW_CONFIG": ("cli", "show_config"),
        "RAG_SERVER_CLI_CONFIG_OUTPUT": ("cli", "show_config"),
    }

    result: dict[str, Any] = {}
    for env_name, path in mapping.items():
        if env_name not in env:
            continue
        section, key = path
        result.setdefault(section, {})[key] = env[env_name]
    return result


def _path_settings(raw: Any) -> PathSettings:
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
            minimum=0,
        ),
        max_repeated_tool_calls=_coerce_int(
            section.get(
                "max_repeated_tool_calls",
                default.max_repeated_tool_calls,
            ),
            "agent.max_repeated_tool_calls",
            minimum=1,
        ),
        reflection_enabled=_coerce_bool(
            section.get("reflection_enabled", default.reflection_enabled),
            "agent.reflection_enabled",
        ),
    )


def _retrieval_settings(raw: Any) -> RetrievalSettings:
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
            minimum=1,
        ),
    )


def _llm_settings(raw: Any) -> LLMSettings:
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
            minimum=1,
        ),
        timeout_s=_coerce_optional_float(
            section.get("timeout_s", default.timeout_s),
            "llm.timeout_s",
        ),
        retry_backoff_s=_coerce_float(
            section.get("retry_backoff_s", default.retry_backoff_s),
            "llm.retry_backoff_s",
            minimum=0.0,
        ),
    )


def _memory_settings(raw: Any) -> MemorySettings:
    section = _section(raw, "memory")
    _ensure_known_keys("memory", section, {"enabled", "top_k"})
    default = MemorySettings()
    return MemorySettings(
        enabled=_coerce_bool(section.get("enabled", default.enabled), "memory.enabled"),
        top_k=_coerce_int(section.get("top_k", default.top_k), "memory.top_k", minimum=1),
    )


def _skills_settings(raw: Any) -> SkillsSettings:
    section = _section(raw, "skills")
    _ensure_known_keys("skills", section, {"enabled", "dirs"})
    default = SkillsSettings()
    return SkillsSettings(
        enabled=_coerce_bool(section.get("enabled", default.enabled), "skills.enabled"),
        dirs=_coerce_str_list(section.get("dirs", default.dirs), "skills.dirs"),
    )


def _mcp_settings(raw: Any) -> MCPSettings:
    section = _section(raw, "mcp")
    _ensure_known_keys("mcp", section, {"enabled"})
    default = MCPSettings()
    return MCPSettings(
        enabled=_coerce_bool(section.get("enabled", default.enabled), "mcp.enabled"),
    )


def _trace_settings(raw: Any) -> TraceSettings:
    section = _section(raw, "trace")
    _ensure_known_keys("trace", section, {"enabled", "live"})
    default = TraceSettings()
    return TraceSettings(
        enabled=_coerce_bool(section.get("enabled", default.enabled), "trace.enabled"),
        live=_coerce_bool(section.get("live", default.live), "trace.live"),
    )


def _cli_settings(raw: Any) -> CLISettings:
    section = _section(raw, "cli")
    _ensure_known_keys("cli", section, {"show_config"})
    default = CLISettings()
    return CLISettings(
        show_config=_coerce_bool(
            section.get("show_config", default.show_config),
            "cli.show_config",
        ),
    )


def _normalize_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    aliases = {
        ("paths", "mcp_config"): ("paths", "mcp_config_path"),
        ("agent", "chat_provider"): ("agent", "provider"),
        ("agent", "chat_model"): ("agent", "model"),
        ("agent", "chat_model_kwargs"): ("agent", "model_kwargs"),
        ("agent", "reflection"): ("agent", "reflection_enabled"),
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
        ("llm", "query_rewrite_provider"): ("llm", "rewrite_provider"),
        ("llm", "query_rewrite_model"): ("llm", "rewrite_model"),
        ("llm", "query_rewrite_kwargs"): ("llm", "rewrite_kwargs"),
        ("llm", "llm_retry_attempts"): ("llm", "retry_attempts"),
        ("llm", "llm_timeout_s"): ("llm", "timeout_s"),
        ("llm", "llm_retry_backoff_s"): ("llm", "retry_backoff_s"),
        ("memory", "memory_top_k"): ("memory", "top_k"),
        ("trace", "live_events"): ("trace", "live"),
        ("trace", "live_logs"): ("trace", "live"),
        ("cli", "startup_config"): ("cli", "show_config"),
        ("cli", "show_startup_config"): ("cli", "show_config"),
        ("cli", "config_output"): ("cli", "show_config"),
        ("cli", "live_events"): ("trace", "live"),
        ("cli", "live_logs"): ("trace", "live"),
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
    merged = {str(key): value for key, value in base.items()}
    for key, value in overlay.items():
        key = str(key)
        if (
            isinstance(value, Mapping)
            and isinstance(merged.get(key), Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _section(raw: Any, name: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{name} must be an object")
    return {str(key): value for key, value in raw.items()}


def _ensure_known_sections(raw: Mapping[str, Any]) -> None:
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
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"Unknown config section(s): {', '.join(unknown)}")


def _ensure_known_keys(section_name: str, raw: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        keys = ", ".join(unknown)
        raise ConfigError(f"Unknown key(s) in {section_name}: {keys}")


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
    if number <= 0:
        return None
    return number


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
