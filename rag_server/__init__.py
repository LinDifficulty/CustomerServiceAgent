"""
可复用的 RAG（检索增强生成）与电商智能客服组件库。

本包提供以下核心能力：
- 混合检索（FAISS 向量检索 + BM25 关键词检索 + CrossEncoder 重排序）
- 基于 LangGraph 的 CLI Agent（含工具调用与反思循环）
- 用户长期记忆管理（SQLite + 向量索引）
- Anthropic 风格的 Skills 能力注册与加载
- MCP 客户端集成（支持 stdio/HTTP/SSE/WebSocket）
- 检索效果评估（命中率、MRR、来源匹配/子串匹配）
- 查询改写与多查询融合搜索
- 缓存层（Redis / 内存）加速重复查询
"""

# ── 缓存服务 ──
from .cache_service import InMemoryJsonCache, JsonCache, RedisJsonCache

# ── 配置系统 ──
from .config import AppConfig, ConfigError, load_app_config

# ── 检索评估 ──
from .eval_service import evaluate_retrieval_dataset, load_retrieval_eval_dataset

# ── LLM 重试策略 ──
from .llm_retry import LLMRetryError, LLMRetryPolicy

# ── MCP 客户端配置与工具加载 ──
from .mcp_service import MCPConfig, MCPToolLoadResult, load_mcp_config

# ── 用户记忆服务 ──
from .memory_service import MEMORY_LAYERS, LLMMemoryExtractor, MemoryService

# ── 模型工厂 ──
from .model_factory import ModelProviderError

# ── 查询改写 ──
from .query_rewrite import LLMQueryRewriter, QueryRewriteResult

# ── RAG 核心检索服务 ──
from .rag_service import RAGService

# ── 反思代理（Agent 反思循环） ──
from .reflection_service import ReflectionAgent, ReflectionResult

# ── Skills 能力注册 ──
from .skill_service import SkillDefinition, SkillRegistry

# ── 链路追踪 ──
from .trace_service import TraceRecorder, load_trace, summarize_trace

# 明确定义包的公开 API，控制 from rag_server import * 的行为
__all__ = [
    "AppConfig",
    "ConfigError",
    "InMemoryJsonCache",
    "JsonCache",
    "LLMQueryRewriter",
    "LLMMemoryExtractor",
    "LLMRetryError",
    "LLMRetryPolicy",
    "MEMORY_LAYERS",
    "MCPConfig",
    "MCPToolLoadResult",
    "ModelProviderError",
    "MemoryService",
    "QueryRewriteResult",
    "RAGService",
    "RedisJsonCache",
    "ReflectionAgent",
    "ReflectionResult",
    "SkillDefinition",
    "SkillRegistry",
    "TraceRecorder",
    "evaluate_retrieval_dataset",
    "load_app_config",
    "load_retrieval_eval_dataset",
    "load_mcp_config",
    "load_trace",
    "summarize_trace",
]
