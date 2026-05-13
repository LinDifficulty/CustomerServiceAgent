# 用户长期记忆模块：SQLite持久化 + 按用户的FAISS语义索引 + 三层记忆体系
# 三层记忆：profile（用户画像）、episode（对话片段）、procedure（操作流程）
# 支持软删除（deleted_at时间戳）、过期管理（expires_at）、LLM记忆抽取
from __future__ import annotations

import asyncio  # 异步IO，支持并发记忆搜索
import hashlib  # SHA256哈希，为每个用户生成索引文件名
import json  # 序列化/反序列化metadata和索引ID
import os  # 文件路径和环境变量
import sqlite3  # 结构化记忆存储
import threading  # 线程锁，保证FAISS索引读写的线程安全
import time  # 性能计时
import uuid  # 为每条记忆生成全局唯一ID
from dataclasses import dataclass  # ExtractedMemory数据类
from datetime import UTC, datetime
from pathlib import Path  # 跨平台路径处理
from typing import Any

# 解决macOS上OpenMP库冲突问题，避免FAISS初始化时报错
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss  # Facebook AI Similarity Search，高性能向量索引和相似度搜索
import numpy as np  # 数值计算库，用于向量矩阵操作
from langchain_core.messages import HumanMessage, SystemMessage  # LLM消息类型

from .cache_service import CacheTTLs, JsonCache, read_cached_list, stable_cache_digest, write_cached_list
from .llm_retry import LLMRetryPolicy, ainvoke_with_retry, invoke_with_retry  # LLM调用重试策略
from .model_factory import (
    DEFAULT_CHAT_MODEL,  # 默认聊天模型
    DEFAULT_CHAT_PROVIDER,  # 默认聊天模型提供商（DashScope）
    DEFAULT_EMBEDDING_MODEL,  # 默认Embedding模型
    DEFAULT_EMBEDDING_PROVIDER,  # 默认Embedding提供商
    create_chat_model,  # 模型工厂函数
    create_embeddings,  # Embeddings工厂函数
    model_config_fingerprint,  # 模型配置指纹，用于检测配置变更
)
from .trace_service import TraceRecorder
from .utils import (
    cache_key_or_none,
    call_async_fallback,
    coerce_message_content,
    load_prompt,
    normalize_vector_score,
    parse_json_object,
    trace_retry_failure,
    utc_now,
)

# 所有合法的记忆类型（memory_type）
# profile: 用户画像/身份信息 | preference: 偏好/喜好 | constraint: 限制条件
# instruction: 指令/要求 | episode: 对话片段/历史事件摘要 | procedure: 操作流程
MEMORY_TYPES = {
    "profile",
    "preference",
    "constraint",
    "instruction",
    "episode",
    "procedure",
}

# 三层记忆体系：将细分的记忆类型归类到三个逻辑层中，便于分层检索
# profile层：用户画像相关（画像、偏好、限制、指令）
# episode层：对话片段和历史事件摘要
# procedure层：用户明确要求的可复用操作流程
MEMORY_LAYERS = {
    "profile": {"profile", "preference", "constraint", "instruction"},
    "episode": {"episode"},
    "procedure": {"procedure"},
}

# 各层的默认召回数量：profile层4条，episode和procedure各3条
DEFAULT_MEMORY_LAYER_TOP_K = {
    "profile": 4,
    "episode": 3,
    "procedure": 3,
}


def _clamp_importance(value: Any) -> float:
    """将重要性分数夹紧到 [0.0, 1.0] 区间。
    如果传入的值无法转为float，默认返回0.5（中等重要性）。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.5  # 无法转换时使用默认中等重要性
    return max(0.0, min(1.0, number))  # 夹紧到合法范围


def memory_layer_for_type(memory_type: str) -> str:
    """根据记忆类型（如'preference'）找到所属的记忆层（如'profile'）。
    遍历 MEMORY_LAYERS 映射，如果类型不在任何层中则默认归入 profile 层。"""
    for layer, layer_types in MEMORY_LAYERS.items():
        if memory_type in layer_types:
            return layer
    return "profile"  # 未知类型默认归入画像层


class MemoryService:
    """用户维度的长期记忆存储服务。

    核心设计：
    - SQLite 负责结构化记录的持久化（CRUD、软删除、过期管理）
    - FAISS 按用户独立存储语义索引，每个用户一个 .faiss 文件 + .ids.json 文件
    - 按用户建索引的好处：避免跨用户向量竞争，删除和清空操作更简洁高效
    - 索引可重建：当数据库记录变更时，从 SQLite 重新生成 FAISS 索引
    """

    def __init__(
        self,
        data_dir: str = "memory",
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
        embedding_model_name: str | None = None,
        embedding_model_kwargs: dict[str, Any] | None = None,
        embeddings: Any | None = None,
        trace_recorder: TraceRecorder | None = None,
        cache: JsonCache | None = None,
        cache_ttls: dict[str, Any] | None = None,
    ) -> None:
        """初始化记忆服务。

        参数:
            data_dir: 数据存储目录，默认为"memory"
            model_name: Embedding模型名称，用于向量化记忆内容
            embedding_provider: Embedding服务提供商（如DashScope）
            embedding_model_name: 可覆盖 model_name 的更具体的模型名
            embedding_model_kwargs: 传递给Embedding模型的额外参数
            embeddings: 可注入的已有Embeddings实例，用于测试
            trace_recorder: 可选的追踪记录器，用于记录操作日志
            cache: 可选的JSON缓存服务，用于缓存Embedding向量和搜索结果
            cache_ttls: 缓存过期时间配置
        """
        # 初始化数据目录结构
        self.base_dir = Path(data_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # SQLite数据库文件路径
        self.sqlite_path = self.base_dir / "memory.sqlite"
        # FAISS索引文件存放目录
        self.index_dir = self.base_dir / "indexes"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        # Embedding配置记录文件，用于检测配置变更
        self.embedding_config_path = self.base_dir / "embedding_config.json"

        # Embedding模型配置
        self.embedding_provider = embedding_provider
        self.embedding_model_name = embedding_model_name or model_name
        self.embedding_model_kwargs = dict(embedding_model_kwargs or {})
        # 计算配置指纹，用于检测模型/参数是否发生变化
        self.embedding_config_hash = model_config_fingerprint(
            self.embedding_provider,
            self.embedding_model_name,
            self.embedding_model_kwargs,
        )
        # 创建或使用注入的Embeddings实例
        self.embeddings = embeddings or create_embeddings(
            provider=self.embedding_provider,
            model_name=self.embedding_model_name,
            **self.embedding_model_kwargs,
        )

        # 追踪和缓存服务
        self.trace_recorder = trace_recorder
        self.cache = cache
        self.cache_ttls = CacheTTLs.from_mapping(cache_ttls)

        # 内存中的索引缓存：{user_id: (faiss_index, id_list)}
        # 避免每次搜索都从磁盘加载FAISS索引
        self._index_cache: dict[str, tuple[faiss.Index | None, list[str]]] = {}
        # 缓存版本号 TTL 缓存：{user_id: (version_string, timestamp)}
        # 5秒内重复查询复用同一版本号，避免每层搜索都执行聚合 SQL
        self._version_cache: dict[str, tuple[str, float]] = {}
        self._version_cache_ttl: float = 5.0
        # 可重入锁，保证多线程下索引读写的安全性
        self._lock = threading.RLock()

        # 初始化SQLite连接，check_same_thread=False允许多线程访问
        self.conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        # 设置行工厂为sqlite3.Row，使查询结果可以通过列名访问
        self.conn.row_factory = sqlite3.Row
        self._init_db()  # 创建表和索引
        self._ensure_embedding_config()  # 验证或重建Embedding配置

    def __enter__(self) -> MemoryService:
        """上下文管理器入口，支持 `with MemoryService(...) as svc:` 语法。"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器出口，自动关闭数据库连接。"""
        self.close()
        return False

    def add_memory(
        self,
        user_id: str,
        content: str,
        *,
        memory_type: str = "preference",
        importance: float = 0.5,
        source: str = "conversation",
        metadata: dict[str, Any] | None = None,
        expires_at: str | None = None,
    ) -> dict:
        """添加单条记忆并重建该用户的语义索引。

        这是 add_memories 的便捷包装，内部将单条记忆转换为列表后调用批量添加。
        参数:
            user_id: 用户ID
            content: 记忆内容文本
            memory_type: 记忆类型（profile/preference/constraint/instruction/episode/procedure）
            importance: 重要性分数 [0.0, 1.0]
            source: 来源标识，如"conversation"
            metadata: 附加元数据字典
            expires_at: ISO格式的过期时间，None表示永不过期
        返回:
            插入的记忆记录字典
        异常:
            ValueError: 当记忆内容为空时抛出
        """
        # 委托给批量添加方法，单条包装为列表
        records = self.add_memories(
            user_id,
            [
                {
                    "content": content,
                    "memory_type": memory_type,
                    "importance": importance,
                    "source": source,
                    "metadata": metadata or {},
                    "expires_at": expires_at,
                }
            ],
        )
        if not records:
            raise ValueError("memory content must not be empty")
        return records[0]

    def add_memories(self, user_id: str, memories: list[dict[str, Any]]) -> list[dict]:
        """批量添加多条记忆，在单个事务中完成SQL写入和FAISS索引一次性重建。

        参数:
            user_id: 用户ID
            memories: 记忆字典列表，每条包含 content/memory_type/importance 等字段
        返回:
            插入成功的记忆记录字典列表
        注意:
            - 空内容和非法类型会被过滤或修正
            - 所有插入在同一SQLite事务中完成
            - 仅在插入至少一条记录后才重建FAISS索引
        """
        now = utc_now()  # 统一的时间戳，保证同一批次的时间一致
        inserted: list[dict[str, Any]] = []

        # 遍历并规范化每条记忆
        for item in memories:
            # 跳过空内容的记忆
            content = str(item.get("content") or "").strip()
            if not content:
                continue

            # 验证记忆类型，非法类型默认为 "preference"
            memory_type = str(item.get("memory_type") or item.get("type") or "")
            if memory_type not in MEMORY_TYPES:
                memory_type = "preference"

            # 确保 metadata 为字典类型
            metadata = item.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}

            # 处理过期时间：空字符串视为 None（永不过期）
            expires_at = item.get("expires_at")
            if expires_at is not None:
                expires_at = str(expires_at).strip() or None

            # 构建标准化的记忆记录
            inserted.append(
                {
                    "id": str(uuid.uuid4()),  # 为每条记忆生成全局唯一ID
                    "user_id": user_id,
                    "content": content,
                    "memory_type": memory_type,
                    "importance": _clamp_importance(item.get("importance", 0.5)),
                    "source": str(item.get("source") or "conversation"),
                    "metadata": metadata,
                    "created_at": now,
                    "updated_at": now,
                    "expires_at": expires_at,
                }
            )

        # 没有有效记忆时直接返回空列表
        if not inserted:
            return []

        # 加锁执行SQL插入和索引重建，保证原子性
        with self._lock:
            with self.conn:  # SQLite事务上下文
                # 批量插入所有记忆到memories表
                # deleted_at为NULL表示当前是活跃记录（未软删除）
                self.conn.executemany(
                    """
                    INSERT INTO memories (
                        id, user_id, content, memory_type, importance, source,
                        metadata_json, created_at, updated_at, expires_at, deleted_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                    """,
                    [
                        (
                            item["id"],
                            item["user_id"],
                            item["content"],
                            item["memory_type"],
                            item["importance"],
                            item["source"],
                            json.dumps(item["metadata"], ensure_ascii=False),
                            item["created_at"],
                            item["updated_at"],
                            item["expires_at"],
                        )
                        for item in inserted
                    ],
                )

            # 重建该用户的FAISS语义索引以包含新记忆
            self._rebuild_user_index(user_id)
            self._invalidate_version_cache(user_id)
            # 返回插入后的完整记录（重新从数据库读取以获取最新状态）
            return [self.get_memory(item["id"]) for item in inserted]

    def get_memory(self, memory_id: str, user_id: str | None = None) -> dict | None:
        """根据记忆ID查询单条记忆记录。

        参数:
            memory_id: 记忆的唯一ID
            user_id: 可选，用于缩小查询范围并做权限校验
        返回:
            记忆记录字典，未找到或已过期/已删除则返回None
        """
        # 构建WHERE条件：必须匹配ID且未被软删除
        where = ["id = ?", "deleted_at IS NULL"]
        params: list[Any] = [memory_id]
        if user_id is not None:
            where.append("user_id = ?")
            params.append(user_id)

        with self._lock:
            row = self.conn.execute(
                f"SELECT * FROM memories WHERE {' AND '.join(where)}",
                params,
            ).fetchone()
        # 检查是否未找到，或记录已过期
        if row is None or self._is_expired(row):
            return None
        return self._row_to_record(row)

    def _batch_get_memories(
        self,
        memory_ids: list[str],
        *,
        user_id: str | None = None,
    ) -> dict[str, dict]:
        """批量获取记忆记录，一次 SQL 查询替代 N 次单独查询。

        返回 {memory_id: record_dict} 字典，未找到或已过期/已删除的记录不包含在内。
        """
        if not memory_ids:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        params: list[Any] = list(memory_ids)
        where = ["id IN (" + placeholders + ")", "deleted_at IS NULL"]
        if user_id is not None:
            where.append("user_id = ?")
            params.append(user_id)
        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM memories WHERE {' AND '.join(where)} ORDER BY id",
                params,
            ).fetchall()
        result: dict[str, dict] = {}
        for row in rows:
            if self._is_expired(row):
                continue
            result[row["id"]] = self._row_to_record(row)
        return result

    def list_memories(
        self,
        user_id: str,
        *,
        limit: int = 50,
        include_expired: bool = False,
    ) -> list[dict]:
        """列出指定用户的活跃记忆列表。

        参数:
            user_id: 用户ID
            limit: 返回的最大记录数
            include_expired: 是否包含已过期的记忆，默认False
        返回:
            记忆记录字典列表，按更新时间降序排列
        """
        start = time.perf_counter()  # 性能计时开始
        # 基础条件：匹配用户且未被软删除
        where = ["user_id = ?", "deleted_at IS NULL"]
        params: list[Any] = [user_id]
        if not include_expired:
            # 过滤已过期的记录：expires_at为空（永不过期）或大于当前时间
            where.append("(expires_at IS NULL OR expires_at > ?)")
            params.append(utc_now())

        with self._lock:
            rows = self.conn.execute(
                f"""
                SELECT * FROM memories
                WHERE {" AND ".join(where)}
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                [*params, max(1, limit)],
            ).fetchall()
        # 将SQLite行转换为标准字典格式
        results = [self._row_to_record(row) for row in rows]
        # 记录追踪事件
        self._trace_event(
            "memory.list_memories",
            {
                "user_id": user_id,
                "limit": limit,
                "include_expired": include_expired,
                "result_count": len(results),
                "elapsed_ms": (time.perf_counter() - start) * 1000,
            },
        )
        return results

    def search_memory(
        self,
        user_id: str,
        query: str,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
        memory_types: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> list[dict]:
        """对单个用户的活跃记忆进行语义搜索。

        搜索流程：
        1. 参数校验（空查询或top_k<=0直接返回）
        2. 查询缓存，命中则直接返回
        3. 从FAISS索引中召回候选（检索量 = top_k * 50，至少100）
        4. 从SQLite验证每条候选记录的有效性（未删除、未过期）
        5. 按记忆类型过滤（如果指定了memory_types）
        6. 归一化向量分数并过滤低于阈值的记录
        7. 按分数降序排列并写入缓存

        参数:
            user_id: 用户ID
            query: 语义搜索查询文本
            top_k: 期望返回的前K条结果
            min_score: 最小相似度阈值 [0.0, 1.0]
            memory_types: 可选的记忆类型白名单，None表示不过滤类型
        返回:
            记忆记录字典列表（含score字段），按相似度降序排列
        """
        start = time.perf_counter()
        # 规范化记忆类型：过滤非法类型字符串
        allowed_types = self._normalize_memory_types(memory_types)

        # 空查询或非法的top_k：直接返回空结果
        if top_k <= 0 or not query.strip():
            self._trace_search_memory(
                user_id=user_id,
                query=query,
                top_k=top_k,
                min_score=min_score,
                allowed_types=allowed_types,
                result_count=0,
                elapsed_ms=(time.perf_counter() - start) * 1000,
                reason="empty_query_or_top_k",
            )
            return []

        # 构建缓存查询载荷，尝试从缓存中读取结果
        cache_payload = self._memory_search_cache_payload(
            user_id=user_id,
            query=query,
            top_k=top_k,
            min_score=min_score,
            allowed_types=allowed_types,
        )
        cached_results = read_cached_list(self.cache, "memory_search",cache_payload)
        if cached_results is not None:
            # 缓存命中：直接返回缓存结果，节省Embedding调用和FAISS搜索
            self._trace_search_memory(
                user_id=user_id,
                query=query,
                top_k=top_k,
                min_score=min_score,
                allowed_types=allowed_types,
                result_count=len(cached_results),
                elapsed_ms=(time.perf_counter() - start) * 1000,
                reason="cache_hit",
                results=[
                    {
                        "id": item["id"][:8],
                        "memory_type": item["memory_type"],
                        "memory_layer": item.get("memory_layer"),
                        "score": item.get("score"),
                    }
                    for item in cached_results
                ],
            )
            return cached_results

        # 获取用户的FAISS索引和对应的ID列表
        index, index_ids = self._get_user_index(user_id)
        # 索引为空或没有向量：该用户还没有记忆
        if index is None or index.ntotal == 0:
            self._trace_search_memory(
                user_id=user_id,
                query=query,
                top_k=top_k,
                min_score=min_score,
                allowed_types=allowed_types,
                result_count=0,
                elapsed_ms=(time.perf_counter() - start) * 1000,
                reason="empty_index",
            )
            return []

        # 从FAISS中检索比所需更多的候选（top_k*50），便于后续过滤
        # 至少检索100条，保证有充足候选可过滤
        limit = min(index.ntotal, max(top_k * 50, 100))
        # 将查询文本向量化并在FAISS索引中搜索
        scores, indices = index.search(self._embed_texts([query]), limit)

        # 收集所有有效候选的 memory_id（去重），用于批量验证
        candidate_scores: dict[str, float] = {}
        candidate_order: list[str] = []  # 保持 FAISS 返回的原始顺序
        for raw_score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0 or idx >= len(index_ids):
                continue
            memory_id = index_ids[int(idx)]
            if memory_id in candidate_scores:
                continue
            candidate_scores[memory_id] = normalize_vector_score(float(raw_score))
            candidate_order.append(memory_id)

        if not candidate_order:
            write_cached_list(self.cache, "memory_search", cache_payload, [], self.cache_ttls.memory_ttl_s)
            return []

        # 用 batch get 替代逐个 get_memory，大幅减少 SQL 往返
        batch_records = self._batch_get_memories(candidate_order, user_id=user_id)

        results: list[dict] = []
        seen: set[str] = set()
        for memory_id in candidate_order:
            score = candidate_scores[memory_id]
            if score < min_score:
                continue
            record = batch_records.get(memory_id)
            if record is None:
                continue
            if memory_id in seen:
                continue
            seen.add(memory_id)
            # 按记忆类型过滤
            if allowed_types is not None and record["memory_type"] not in allowed_types:
                continue
            record["score"] = score
            results.append(record)

            # 达到所需数量后停止
            if len(results) >= top_k:
                break

        # 按分数降序排列
        results.sort(key=lambda item: item["score"], reverse=True)
        # 将结果写入缓存以供后续查询复用
        write_cached_list(self.cache, "memory_search", cache_payload, results, self.cache_ttls.memory_ttl_s)
        # 记录搜索追踪事件
        self._trace_search_memory(
            user_id=user_id,
            query=query,
            top_k=top_k,
            min_score=min_score,
            allowed_types=allowed_types,
            result_count=len(results),
            elapsed_ms=(time.perf_counter() - start) * 1000,
            results=[
                {
                    "id": item["id"][:8],
                    "memory_type": item["memory_type"],
                    "memory_layer": item.get("memory_layer"),
                    "score": item.get("score"),
                }
                for item in results
            ],
        )
        return results

    def search_memory_layer(
        self,
        user_id: str,
        query: str,
        layer: str,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[dict]:
        """搜索单个记忆层（profile/episode/procedure）。

        将层名转换为对应的记忆类型集合后委托给 search_memory。
        参数:
            user_id: 用户ID
            query: 搜索查询文本
            layer: 记忆层名称（profile / episode / procedure）
            top_k: 返回的最大结果数
            min_score: 最小相似度阈值
        返回:
            该层中匹配的记忆记录列表
        """
        # 根据层名查找对应的记忆类型集合
        memory_types = MEMORY_LAYERS.get(layer)
        if memory_types is None:
            raise ValueError(f"Unknown memory layer: {layer}")
        return self.search_memory(
            user_id,
            query,
            top_k=top_k,
            min_score=min_score,
            memory_types=memory_types,
        )

    def search_memory_layers(
        self,
        user_id: str,
        query: str,
        *,
        layer_top_k: dict[str, int] | None = None,
        min_score: float = 0.0,
    ) -> dict[str, list[dict]]:
        """同步搜索全部三层记忆（profile/episode/procedure），各层独立并行检索。

        参数:
            user_id: 用户ID
            query: 搜索查询文本
            layer_top_k: 自定义各层的top_k值，会与默认值合并
            min_score: 最小相似度阈值
        返回:
            字典 {layer_name: [memory_records]}，如 {"profile": [...], "episode": [...], "procedure": [...]}
        """
        start = time.perf_counter()
        # 合并自定义top_k与默认top_k
        limits = {**DEFAULT_MEMORY_LAYER_TOP_K, **(layer_top_k or {})}
        layered: dict[str, list[dict]] = {}
        # 逐层搜索（同步串行）
        for layer in MEMORY_LAYERS:
            layered[layer] = self.search_memory_layer(
                user_id,
                query,
                layer,
                top_k=limits.get(layer, 3),
                min_score=min_score,
            )
        # 记录分层搜索追踪事件
        self._trace_event(
            "memory.search_memory_layers",
            {
                "user_id": user_id,
                "query": query,
                "layer_top_k": limits,
                "min_score": min_score,
                "layer_counts": {layer: len(items) for layer, items in layered.items()},
                "elapsed_ms": (time.perf_counter() - start) * 1000,
            },
        )
        return layered

    async def asearch_memory_layers(
        self,
        user_id: str,
        query: str,
        *,
        layer_top_k: dict[str, int] | None = None,
        min_score: float = 0.0,
    ) -> dict[str, list[dict]]:
        """异步搜索全部三层记忆，各层并发执行以提升性能。

        与 search_memory_layers 功能相同，但使用 asyncio.gather 将三层搜索并发化。
        每层通过 asyncio.to_thread 在独立线程中执行同步的 search_memory_layer。

        参数:
            user_id: 用户ID
            query: 搜索查询文本
            layer_top_k: 自定义各层top_k
            min_score: 最小相似度阈值
        返回:
            字典 {layer_name: [memory_records]}
        """
        start = time.perf_counter()
        # 合并自定义与默认的每层召回数量
        limits = {**DEFAULT_MEMORY_LAYER_TOP_K, **(layer_top_k or {})}

        # 预先嵌入查询向量：避免 3 个并发层在冷缓存时同时调用 API
        await asyncio.to_thread(self._embed_texts, [query])

        # 内嵌异步函数：将同步的分层搜索包装为线程执行
        async def search_layer(layer: str) -> tuple[str, list[dict]]:
            results = await asyncio.to_thread(
                self.search_memory_layer,
                user_id,
                query,
                layer,
                top_k=limits.get(layer, 3),
                min_score=min_score,
            )
            return layer, results

        # 三层搜索并发执行（asyncio.gather 并行化）
        pairs = await asyncio.gather(*(search_layer(layer) for layer in MEMORY_LAYERS))
        layered = dict(pairs)
        # 记录分层搜索追踪事件
        self._trace_event(
            "memory.search_memory_layers",
            {
                "user_id": user_id,
                "query": query,
                "layer_top_k": limits,
                "min_score": min_score,
                "layer_counts": {layer: len(items) for layer, items in layered.items()},
                "elapsed_ms": (time.perf_counter() - start) * 1000,
            },
        )
        return layered

    def forget_memory(self, memory_id: str, user_id: str | None = None) -> bool:
        """软删除单条记忆（设置deleted_at时间戳而非物理删除）。

        软删除的优势：保留历史记录可用于审计，同时保证索引重建时不会包含已删除记录。
        参数:
            memory_id: 要删除的记忆ID
            user_id: 可选，用于限定用户范围
        返回:
            True表示删除成功，False表示未找到匹配记录
        """
        # 构建WHERE条件：匹配ID且未被软删除（防止重复删除）
        where = ["id = ?", "deleted_at IS NULL"]
        params: list[Any] = [memory_id]
        if user_id is not None:
            where.append("user_id = ?")
            params.append(user_id)

        with self._lock:
            # 在删除前先获取受影响的用户ID，用于后续索引重建
            affected_user_ids = self._memory_user_ids(memory_id, user_id=user_id)

            with self.conn:  # SQLite事务
                # UPDATE而非DELETE：设置deleted_at和updated_at为当前时间
                cursor = self.conn.execute(
                    f"""
                    UPDATE memories
                    SET deleted_at = ?, updated_at = ?
                    WHERE {" AND ".join(where)}
                    """,
                    [utc_now(), utc_now(), *params],
                )

            # rowcount为0表示未匹配到任何活跃记录
            if cursor.rowcount <= 0:
                return False
            # 重建受影响用户的FAISS索引以排除已删除记录
            for affected_user_id in affected_user_ids:
                self._rebuild_user_index(affected_user_id)
                self._invalidate_version_cache(affected_user_id)
            return True

    def clear_user_memory(self, user_id: str) -> int:
        """清空指定用户的所有记忆（软删除该用户的所有活跃记录）。

        注意：这是软删除操作，不会物理删除SQLite中的记录，
        而是将deleted_at设置为当前时间，这样历史数据依然保留。
        参数:
            user_id: 要清空的用户ID
        返回:
            被软删除的记录数量
        """
        with self._lock:
            with self.conn:  # SQLite事务
                # 批量软删除该用户所有未删除的记录
                cursor = self.conn.execute(
                    """
                    UPDATE memories
                    SET deleted_at = ?, updated_at = ?
                    WHERE user_id = ? AND deleted_at IS NULL
                    """,
                    [utc_now(), utc_now(), user_id],
                )

            # 如果有记录被删除，重建该用户的FAISS索引
            if cursor.rowcount > 0:
                self._rebuild_user_index(user_id)
                self._invalidate_version_cache(user_id)
            return int(cursor.rowcount)

    def close(self) -> None:
        """关闭SQLite数据库连接，释放资源。
        应在MemoryService不再使用时调用，避免数据库连接泄漏。"""
        with self._lock:
            self.conn.close()

    def _trace_event(
        self,
        name: str,
        payload: dict[str, Any],
        *,
        level: str = "info",
    ) -> None:
        """记录追踪事件到trace_recorder（如果已配置）。
        用于可观测性：记录操作类型、参数、耗时和结果。
        参数:
            name: 事件名称（如 "memory.search_memory"）
            payload: 事件携带的数据
            level: 事件级别（info/warning/error）
        """
        if self.trace_recorder is None:
            return
        self.trace_recorder.event("memory", name, payload, level=level)

    def _trace_search_memory(
        self,
        *,
        user_id: str,
        query: str,
        top_k: int,
        min_score: float,
        allowed_types: set[str] | None,
        result_count: int,
        elapsed_ms: float,
        reason: str | None = None,
        results: list[dict] | None = None,
    ) -> None:
        """记录记忆搜索事件的详细追踪信息。

        统一搜索事件的数据格式，包括查询参数、过滤条件、结果数量和耗时。
        reason用于标记特殊情况（如缓存命中、空索引、空查询等）。
        """
        payload: dict[str, Any] = {
            "user_id": user_id,
            "query": query,
            "top_k": top_k,
            "min_score": min_score,
            # 记忆类型过滤条件（如果指定了）
            "memory_types": sorted(allowed_types) if allowed_types else None,
            "result_count": result_count,
            "elapsed_ms": elapsed_ms,
        }
        if reason:
            payload["reason"] = reason  # 如 "cache_hit", "empty_index" 等
        if results is not None:
            payload["results"] = results  # 前K条结果的摘要信息
        self._trace_event("memory.search_memory", payload)

    def _memory_search_cache_payload(
        self,
        *,
        user_id: str,
        query: str,
        top_k: int,
        min_score: float,
        allowed_types: set[str] | None,
    ) -> dict[str, Any]:
        """构建记忆搜索的缓存载荷，包含所有影响搜索结果的因素。
        这些字段共同决定缓存键的唯一性：任何参数变化都导致缓存未命中。
        memory_version用于在记忆内容变更（增删改）时自动使缓存失效。
        """
        return {
            "user_id": user_id,
            "query": query,
            "top_k": top_k,
            "min_score": min_score,
            "memory_types": sorted(allowed_types) if allowed_types else None,
            "embedding_provider": self.embedding_provider,
            "embedding_model_name": self.embedding_model_name,
            "embedding_config_hash": self.embedding_config_hash,  # Embedding配置变更使缓存失效
            "memory_version": self._memory_cache_version(user_id),  # 记忆变更使缓存失效
        }

    def _invalidate_version_cache(self, user_id: str) -> None:
        """使特定用户的缓存版本号失效（在写操作后调用）。"""
        self._version_cache.pop(user_id, None)

    def _memory_cache_version(self, user_id: str) -> str:
        """生成用户记忆的缓存版本号。

        通过统计该用户的总记录数、活跃记录数、最后更新时间、最后删除时间
        来生成一个稳定摘要。当任何记忆被添加、修改或删除时，摘要会变化，
        从而使所有相关缓存自动失效。

        缓存版本号在 5 秒内复用，避免频繁的聚合 SQL 查询。
        """
        now = time.time()
        cached = self._version_cache.get(user_id)
        if cached is not None and now - cached[1] < self._version_cache_ttl:
            return cached[0]
        with self._lock:
            row = self.conn.execute(
                """
                SELECT
                    COUNT(*) AS total_count,
                    SUM(
                        CASE
                            WHEN deleted_at IS NULL           -- 未软删除
                             AND (expires_at IS NULL OR expires_at > ?)  -- 未过期
                            THEN 1 ELSE 0
                        END
                    ) AS active_count,
                    MAX(updated_at) AS max_updated_at,       -- 最近更新时间
                    MAX(deleted_at) AS max_deleted_at        -- 最近删除时间
                FROM memories
                WHERE user_id = ?
                """,
                [utc_now(), user_id],
            ).fetchone()
        # 通过哈希摘要生成稳定的缓存版本字符串
        version = stable_cache_digest(
            {
                "total_count": int(row["total_count"] or 0),
                "active_count": int(row["active_count"] or 0),
                "max_updated_at": row["max_updated_at"],
                "max_deleted_at": row["max_deleted_at"],
            }
        )
        self._version_cache[user_id] = (version, now)
        return version

    def _init_db(self) -> None:
        """初始化SQLite数据库：创建memories表和索引（如果不存在）。

        memories表字段说明：
        - id: 主键，UUID格式的全局唯一标识
        - user_id: 用户ID，用于按用户隔离记忆
        - content: 记忆的文本内容，用于语义搜索
        - memory_type: 记忆类型（profile/preference/constraint/instruction/episode/procedure）
        - importance: 重要性分数 [0.0, 1.0]
        - source: 来源标识（如conversation、manual）
        - metadata_json: JSON格式的附加元数据
        - created_at/updated_at: ISO时间戳
        - expires_at: 可选的过期时间，NULL表示永不过期
        - deleted_at: 软删除时间戳，NULL表示活跃记录

        索引 idx_memories_user_active: 加速按用户查询活跃记忆（最常见查询模式）
        """
        with self._lock, self.conn:
            # 创建memories表（IF NOT EXISTS保证幂等性）
            self.conn.execute(
                """
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        memory_type TEXT NOT NULL,
                        importance REAL NOT NULL,
                        source TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        expires_at TEXT,
                        deleted_at TEXT
                    )
                    """
            )
            # 创建复合索引：加速 "查询某用户的活跃记忆" 这一核心操作
            self.conn.execute(
                """
                    CREATE INDEX IF NOT EXISTS idx_memories_user_active
                    ON memories(user_id, deleted_at, updated_at)
                    """
            )

    def _ensure_embedding_config(self) -> None:
        """验证并确保Embedding配置的一致性。

        如果Embedding模型或提供商发生变化（通过config_hash检测），
        则清除所有已有FAISS索引和内存缓存，因为旧向量与新模型不兼容。
        这防止了因模型变更导致向量维度不匹配的问题。
        """
        # 当前配置快照
        current = {
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model_name,
            "embedding_config_hash": self.embedding_config_hash,
        }
        # 配置未变更，无需重建
        if self._load_embedding_config() == current:
            return

        # 配置已变更：清除所有重建资源
        self._index_cache.clear()  # 清空内存缓存
        for path in self.index_dir.glob("*.faiss"):  # 删除所有FAISS索引文件
            path.unlink()
        for path in self.index_dir.glob("*.ids.json"):  # 删除所有ID映射文件
            path.unlink()
        # 写入新的配置快照
        self.embedding_config_path.write_text(
            json.dumps(current, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_embedding_config(self) -> dict[str, Any] | None:
        """从磁盘加载上次保存的Embedding配置快照。

        返回 None 表示配置文件不存在或损坏，等同于首次运行。
        用于与当前配置比较，检测是否需要重建索引。
        """
        if not self.embedding_config_path.exists():
            return None
        try:
            payload = json.loads(self.embedding_config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None  # 文件损坏，视为无配置
        if not isinstance(payload, dict):
            return None
        # 规范化返回：确保所有字段都是字符串类型
        return {
            "embedding_provider": str(payload.get("embedding_provider") or ""),
            "embedding_model": str(payload.get("embedding_model") or ""),
            "embedding_config_hash": str(payload.get("embedding_config_hash") or ""),
        }

    def _active_rows(self, user_id: str) -> list[sqlite3.Row]:
        """查询指定用户的所有活跃记忆行（未删除且未过期），按创建时间升序排列。
        用于重建FAISS索引时的数据源。"""
        with self._lock:
            return self.conn.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ?
                  AND deleted_at IS NULL                               -- 未被软删除
                  AND (expires_at IS NULL OR expires_at > ?)            -- 未过期或永不过期
                ORDER BY created_at ASC
                """,
                [user_id, utc_now()],
            ).fetchall()

    def _active_count(self, user_id: str) -> int:
        """快速统计指定用户的活跃记忆数量。
        用于判断FAISS索引是否与数据库一致。"""
        with self._lock:
            row = self.conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM memories
                WHERE user_id = ?
                  AND deleted_at IS NULL
                  AND (expires_at IS NULL OR expires_at > ?)
                """,
                [user_id, utc_now()],
            ).fetchone()
        return int(row["count"])

    def _memory_user_ids(self, memory_id: str, user_id: str | None = None) -> list[str]:
        """查询某条记忆所属的用户ID列表。
        用于删除操作后确定需要重建哪些用户的索引。"""
        where = ["id = ?", "deleted_at IS NULL"]
        params: list[Any] = [memory_id]
        if user_id is not None:
            where.append("user_id = ?")
            params.append(user_id)

        with self._lock:
            rows = self.conn.execute(
                f"SELECT DISTINCT user_id FROM memories WHERE {' AND '.join(where)}",
                params,
            ).fetchall()
        return [str(row["user_id"]) for row in rows]

    def _user_index_paths(self, user_id: str) -> tuple[Path, Path]:
        r"""为用户生成FAISS索引文件和ID映射文件的路径。

        使用SHA256哈希处理user_id，避免文件名中包含特殊字符（如/、\等）。
        返回: (faiss_index_path, ids_json_path)
        """
        # SHA256哈希确保文件名安全且唯一
        user_hash = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
        return (
            self.index_dir / f"{user_hash}.faiss",  # FAISS二进制索引
            self.index_dir / f"{user_hash}.ids.json",  # 索引向量对应的memory_id列表
        )

    def _get_user_index(self, user_id: str) -> tuple[faiss.Index | None, list[str]]:
        """获取用户的FAISS索引和ID列表，优先从内存缓存读取。

        如果缓存未命中则从磁盘加载。加载后检查索引是否与数据库一致，
        不一致则触发重建。最终结果会写入内存缓存。

        返回: (faiss_index, [memory_id, ...])，无数据时返回 (None, [])
        """
        with self._lock:
            cached = self._index_cache.get(user_id)
            if cached is None:
                index_path, index_ids_path = self._user_index_paths(user_id)
                index = self._load_index(index_path)
                index_ids = self._load_index_ids(index_ids_path)
            else:
                index, index_ids = cached

            if not self._index_matches_database(user_id, index, index_ids):
                return self._rebuild_user_index(user_id)

            self._index_cache[user_id] = (index, index_ids)
            return index, index_ids

    def _load_index(self, index_path: Path) -> faiss.Index | None:
        """从磁盘加载FAISS索引文件。
        如果文件不存在（用户还没有任何记忆）则返回 None。"""
        if not index_path.exists():
            return None
        return faiss.read_index(str(index_path))

    def _load_index_ids(self, index_ids_path: Path) -> list[str]:
        """从磁盘加载索引ID映射文件（JSON数组，记录每个向量对应的memory_id）。
        文件不存在或损坏时返回空列表。"""
        if not index_ids_path.exists():
            return []
        try:
            payload = json.loads(index_ids_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []  # JSON损坏，视为无效
        if not isinstance(payload, list):
            return []  # 格式不正确
        return [str(item) for item in payload]  # 标准化为字符串列表

    def _save_index_ids(self, index_ids_path: Path, index_ids: list[str]) -> None:
        """将索引ID列表持久化到磁盘JSON文件。
        写入顺序与FAISS索引中向量顺序一致，一一对应。"""
        index_ids_path.write_text(
            json.dumps(index_ids, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _index_matches_database(
        self,
        user_id: str,
        index: faiss.Index | None,
        index_ids: list[str],
    ) -> bool:
        """检查FAISS索引是否与数据库中的活跃记忆一致。

        通过比较三个维度来判断：FAISS向量数量、活跃记录数量、ID列表长度。
        三者必须完全相等才认为一致。不一致的情况包括：
        - 有新记忆添加（数据库多了一条）
        - 有记忆被软删除（数据库少了一条）
        - 索引文件损坏或过期
        """
        active_count = self._active_count(user_id)
        # 无活跃记录时，索引和ID列表也应为空
        if active_count == 0:
            return index is None and not index_ids
        # 有活跃记录但索引为空，不一致
        if index is None:
            return False
        # 三者数量必须完全相等
        return index.ntotal == active_count == len(index_ids)

    def _rebuild_user_index(self, user_id: str) -> tuple[faiss.Index | None, list[str]]:
        """从SQLite数据库重建指定用户的FAISS语义索引。

        流程:
        1. 查询该用户所有活跃记忆（未删除、未过期）
        2. 如果没有活跃记忆：清理旧索引文件和ID文件
        3. 批量向量化所有记忆内容
        4. 构建 FAISS IndexFlatIP（内积索引，配合L2归一化实现余弦相似度）
        5. 持久化索引和ID映射到磁盘
        6. 更新内存缓存

        这是保证索引与数据库一致的核心方法。任何记忆增删改后都会调用此方法。
        """
        with self._lock:
            index_path, index_ids_path = self._user_index_paths(user_id)
            # 获取所有活跃记录作为索引的数据源
            rows = self._active_rows(user_id)
            if not rows:
                # 无活跃记忆：清理磁盘和缓存
                if index_path.exists():
                    index_path.unlink()
                if index_ids_path.exists():
                    index_ids_path.unlink()
                self._index_cache[user_id] = (None, [])
                return None, []

            # 批量向量化：一次API调用处理所有文本（比逐条调用效率高）
            vectors = self._embed_texts([row["content"] for row in rows])
            # 创建内积索引：IndexFlatIP + L2归一化 = 余弦相似度检索
            index = faiss.IndexFlatIP(vectors.shape[1])  # shape[1]是向量维度
            index.add(vectors)
            # 记录向量在索引中的位置与memory_id的对应关系
            index_ids = [row["id"] for row in rows]

            # 持久化到磁盘
            faiss.write_index(index, str(index_path))
            self._save_index_ids(index_ids_path, index_ids)
            # 更新内存缓存
            self._index_cache[user_id] = (index, index_ids)
            return index, index_ids

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """批量文本向量化，支持单文本Embedding缓存。

        流程:
        1. 先查询缓存：已缓存的文本直接复用向量，避免重复API调用
        2. 缓存未命中的文本批量调用Embedding API（一次调用处理所有缺失文本）
        3. 所有向量经L2归一化后用于FAISS内积索引（等价于余弦相似度）
        4. 新生成的向量写入缓存供后续复用

        返回: shape=(n_texts, dim) 的 float32 numpy数组，已L2归一化
        """
        # 第一阶段：检查缓存，区分已缓存和缺失的文本
        cached_vectors: list[list[float] | None] = [None] * len(texts)
        missing_texts: list[str] = []
        missing_indices: list[int] = []

        for index, text in enumerate(texts):
            key = self._embedding_cache_key(text)
            # 从缓存读取向量（如果缓存服务可用）
            cached = self.cache.get_json(key) if key is not None and self.cache else None
            if isinstance(cached, list) and cached:
                cached_vectors[index] = [float(item) for item in cached]  # 缓存命中
            else:
                missing_texts.append(text)  # 缓存未命中，需要调用API
                missing_indices.append(index)

        # 第二阶段：批量调用API向量化所有缺失文本
        if missing_texts:
            vectors = self.embeddings.embed_documents(missing_texts)
            # 将API返回的向量填入cached_vectors对应位置
            for index, vector in zip(missing_indices, vectors, strict=True):
                cached_vectors[index] = [float(item) for item in vector]

        # 第三阶段：构建numpy矩阵并L2归一化
        matrix = np.asarray(cached_vectors, dtype="float32")
        # L2归一化：使每个向量模长为1，这样FAISS内积搜索等价于余弦相似度
        faiss.normalize_L2(matrix)

        # 第四阶段：将新生成的向量写入缓存
        for index in missing_indices:
            key = self._embedding_cache_key(texts[index])
            if key is not None and self.cache is not None:
                self.cache.set_json(
                    key,
                    matrix[index].tolist(),  # 保存归一化后的向量
                    ttl_s=self.cache_ttls.embedding_ttl_s,
                )
        return matrix

    def _embedding_cache_key(self, text: str) -> str | None:
        """为单个文本生成Embedding缓存的键。

        缓存键包含文本原文、Embedding提供商、模型名和配置哈希，
        任何参数变化都会导致不同的缓存键，确保Embedding一致性。
        """
        return cache_key_or_none(
            self.cache,
            "embedding",
            {
                "kind": "memory_text",
                "text": text,
                "embedding_provider": self.embedding_provider,
                "embedding_model_name": self.embedding_model_name,
                "embedding_config_hash": self.embedding_config_hash,
            },
        )

    def _normalize_memory_types(
        self,
        memory_types: set[str] | list[str] | tuple[str, ...] | None,
    ) -> set[str] | None:
        """规范化记忆类型参数：过滤掉不在MEMORY_TYPES中的非法值。
        None表示不限制类型（不过滤）。返回集合类型确保后续O(1)查找。"""
        if memory_types is None:
            return None  # None语义：不做类型过滤
        # 只保留合法的记忆类型，去掉未知类型
        return {str(item) for item in memory_types if str(item) in MEMORY_TYPES}

    def _is_expired(self, row: sqlite3.Row) -> bool:
        """检查一条记忆记录是否已过期。
        expires_at 为 NULL 表示永不过期，否则比较过期时间与当前时间。"""
        expires_at = row["expires_at"]
        return bool(expires_at and datetime.fromisoformat(str(expires_at)) <= datetime.now(UTC))

    def _row_to_record(self, row: sqlite3.Row) -> dict:
        """将SQLite行对象转换为标准的字典格式。

        职责：
        - 解析 metadata_json 中的JSON元数据
        - 自动计算 memory_layer（根据 memory_type 映射到对应层）
        - 规范化 importance 为 float 类型
        - 排除 deleted_at 字段（活跃记录才会被转换）
        """
        # 解析JSON元数据，失败时使用空字典
        try:
            metadata = json.loads(row["metadata_json"])
        except json.JSONDecodeError:
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}

        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "content": row["content"],
            "memory_type": row["memory_type"],
            "memory_layer": memory_layer_for_type(row["memory_type"]),  # 自动推导所属层
            "importance": float(row["importance"]),
            "source": row["source"],
            "metadata": metadata,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "expires_at": row["expires_at"],
        }


@dataclass
class ExtractedMemory:
    """LLM从对话中抽取出的记忆项（数据类，不可变风格）。

    字段说明:
        content:     记忆的文本内容
        memory_type: 记忆类型（profile/preference/constraint/instruction/episode/procedure）
        importance:  重要性分数 [0.0, 1.0]，影响后续检索排序
        expires_at:  可选过期时间，None表示永不过期
    """

    content: str
    memory_type: str
    importance: float
    expires_at: str | None = None


class LLMMemoryExtractor:
    """基于LLM的对话记忆抽取器。

    从前一轮客服对话中提取值得长期保存的用户信息。使用提示词工程指导LLM
    识别和分类用户相关信息，输出标准化的 ExtractedMemory 列表。

    设计原则:
    - 只保存稳定的、可复用的用户信息（如尺码、偏好、过敏等）
    - 不保存商品知识、客服回答、模型推测等非用户信息
    - 不保存敏感信息（密码、支付信息、手机号等）
    - 支持同步(extract)和异步(aextract)两种调用方式
    - LLM调用失败时自动重试
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CHAT_MODEL,
        provider: str = DEFAULT_CHAT_PROVIDER,
        model_kwargs: dict[str, Any] | None = None,
        model: Any | None = None,
        retry_policy: LLMRetryPolicy | None = None,
        trace_recorder: TraceRecorder | None = None,
    ) -> None:
        """初始化LLM记忆抽取器。

        参数:
            model_name: LLM模型名称
            provider: LLM提供商
            model_kwargs: 传递给LLM模型的额外参数
            model: 可注入的已有模型实例（用于测试）
            retry_policy: LLM调用的重试策略
            trace_recorder: 追踪记录器
        """
        self.provider = provider
        self.model_name = model_name
        self.model_kwargs = dict(model_kwargs or {})
        # 创建或使用注入的聊天模型
        self.model = model or create_chat_model(
            provider=provider,
            model_name=model_name,
            **self.model_kwargs,
        )
        self.retry_policy = retry_policy or LLMRetryPolicy()
        self.trace_recorder = trace_recorder
        # 系统提示词：指导LLM如何从对话中抽取记忆
        # 包含详细的抽取规则、记忆类型说明、输出格式要求和隐私保护指令
        self.system_prompt = SystemMessage(content=load_prompt("memory_extractor_system.txt"))

    def extract(
        self,
        *,
        user_message: str,
        assistant_message: str,
        existing_memories: list[dict] | None = None,
    ) -> list[ExtractedMemory]:
        """同步从一轮对话中抽取长期记忆。

        参数:
            user_message: 用户在本轮对话中的消息
            assistant_message: 客服助手的回复
            existing_memories: 已有的相关记忆列表，用于避免重复抽取
        返回:
            ExtractedMemory列表（最多5条），无记忆可抽取时返回空列表
        """
        # 格式化已有记忆为提示词中的上下文
        existing_block = self._format_existing_memories(existing_memories or [])
        # 构建发送给LLM的完整消息序列
        messages = self._build_messages(
            user_message=user_message,
            assistant_message=assistant_message,
            existing_block=existing_block,
        )
        # 调用LLM并解析JSON响应，失败时自动重试
        response = invoke_with_retry(
            lambda: self.model.invoke(messages),
            retry_policy=self.retry_policy,
            operation="memory_extractor.invoke",
            on_failure=self._trace_retry_failure,
        )
        # 解析LLM返回的JSON，提取memories数组并规范化
        payload = parse_json_object(coerce_message_content(response.content))
        return self._normalize_memories(payload.get("memories"))

    async def aextract(
        self,
        *,
        user_message: str,
        assistant_message: str,
        existing_memories: list[dict] | None = None,
    ) -> list[ExtractedMemory]:
        """异步从一轮对话中抽取长期记忆。

        与 extract 功能相同，但使用异步LLM调用。优先使用模型的 ainvoke 方法，
        如果模型不支持原生异步则回退到 asyncio.to_thread 在线程中执行同步调用。
        """
        existing_block = self._format_existing_memories(existing_memories or [])
        messages = self._build_messages(
            user_message=user_message,
            assistant_message=assistant_message,
            existing_block=existing_block,
        )

        async def invoke_model() -> Any:
            return await call_async_fallback(self.model, "ainvoke", "invoke", messages)

        # 异步调用LLM，失败时自动重试
        response = await ainvoke_with_retry(
            invoke_model,
            retry_policy=self.retry_policy,
            operation="memory_extractor.ainvoke",
            on_failure=self._trace_retry_failure,
        )
        # 解析JSON响应并规范化记忆列表
        payload = parse_json_object(coerce_message_content(response.content))
        return self._normalize_memories(payload.get("memories"))

    def _build_messages(
        self,
        *,
        user_message: str,
        assistant_message: str,
        existing_block: str,
    ) -> list[Any]:
        """构建发送给LLM的消息序列。

        消息结构：SystemMessage（抽取规则） + HumanMessage（对话内容）。
        在HumanMessage中包含：
        - 已有的相关记忆列表（帮助LLM判断是否已保存过，避免重复）
        - 用户消息原文
        - 客服回复原文
        """
        return [
            self.system_prompt,  # 抽取规则和格式指令
            HumanMessage(
                content=(
                    "请从下面这轮电商客服对话中抽取需要长期记住的用户信息。\n\n"
                    f"已有相关记忆：\n{existing_block}\n\n"  # 已有记忆作为上下文
                    f"用户消息：{user_message}\n\n"
                    f"客服回复：{assistant_message}"
                )
            ),
        ]

    def _trace_retry_failure(self, event: dict[str, Any]) -> None:
        """记录LLM调用的重试事件到追踪系统。"""
        trace_retry_failure(
            self.trace_recorder,
            "model",
            "memory_extractor.model_retry",
            self.provider,
            self.model_name,
            event,
        )

    def _format_existing_memories(self, memories: list[dict]) -> str:
        """将已有记忆列表格式化为提示词中的文本块。

        最多显示前8条已有记忆（避免提示词过长），
        用"- "前缀的列表格式呈现，帮助LLM判断是否要新增还是跳过。
        无已有记忆时返回"无"。
        """
        if not memories:
            return "无"
        # 只取前8条，每行用 "- 内容" 格式展示
        return "\n".join(f"- {item['content']}" for item in memories[:8])

    def _normalize_memories(self, value: Any) -> list[ExtractedMemory]:
        """将LLM输出的原始JSON解析结果规范化为 ExtractedMemory 列表。

        处理逻辑:
        - 如果LLM返回的不是列表，视为无记忆返回空列表
        - 过滤空内容的记忆项
        - 修正非法的 memory_type（默认改为 "preference"）
        - 夹紧 importance 到 [0, 1] 范围
        - 最多返回5条记忆（防止LLM输出过长的噪音结果）
        """
        if not isinstance(value, list):
            return []  # LLM输出格式异常

        memories: list[ExtractedMemory] = []
        for item in value:
            if not isinstance(item, dict):
                continue  # 跳过非字典项

            # 过滤空内容
            content = str(item.get("content") or "").strip()
            if not content:
                continue

            # 验证并修正记忆类型
            memory_type = str(item.get("memory_type") or item.get("type") or "")
            if memory_type not in MEMORY_TYPES:
                memory_type = "preference"  # 非法类型默认为偏好

            # 处理可选过期时间
            expires_at = item.get("expires_at")
            if expires_at is not None:
                expires_at = str(expires_at).strip() or None

            # 构建规范化的 ExtractedMemory 对象
            memories.append(
                ExtractedMemory(
                    content=content,
                    memory_type=memory_type,
                    importance=_clamp_importance(item.get("importance", 0.5)),
                    expires_at=expires_at,
                )
            )

        # 截断：最多返回5条，避免噪音过多
        return memories[:5]
