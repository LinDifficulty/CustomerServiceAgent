from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import numpy as np
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

from .llm_retry import LLMRetryPolicy, invoke_with_retry

MEMORY_TYPES = {
    "profile",
    "preference",
    "constraint",
    "instruction",
    "episode",
    "procedure",
}
MEMORY_LAYERS = {
    "profile": {"profile", "preference", "constraint", "instruction"},
    "episode": {"episode"},
    "procedure": {"procedure"},
}
DEFAULT_MEMORY_LAYER_TOP_K = {
    "profile": 4,
    "episode": 3,
    "procedure": 3,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp_importance(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.5
    return max(0.0, min(1.0, number))


def memory_layer_for_type(memory_type: str) -> str:
    for layer, layer_types in MEMORY_LAYERS.items():
        if memory_type in layer_types:
            return layer
    return "profile"


class MemoryService:
    """User-scoped long-term memory store.

    SQLite owns the structured records. FAISS stores one rebuildable semantic
    index per user, which avoids cross-user recall competition and keeps delete
    and clear operations straightforward.
    """

    def __init__(
        self,
        data_dir: str = "memory",
        model_name: str = "text-embedding-v4",
        embeddings: Any | None = None,
    ) -> None:
        self.base_dir = Path(data_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.sqlite_path = self.base_dir / "memory.sqlite"
        self.index_dir = self.base_dir / "indexes"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = embeddings or DashScopeEmbeddings(model=model_name)
        self._index_cache: dict[str, tuple[faiss.Index | None, list[str]]] = {}

        self.conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

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
        """Add one memory and rebuild the semantic index."""
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
        """Add multiple memories in one transaction and rebuild FAISS once."""
        now = _utc_now()
        inserted: list[dict[str, Any]] = []

        for item in memories:
            content = str(item.get("content") or "").strip()
            if not content:
                continue

            memory_type = str(item.get("memory_type") or item.get("type") or "")
            if memory_type not in MEMORY_TYPES:
                memory_type = "preference"

            metadata = item.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}

            expires_at = item.get("expires_at")
            if expires_at is not None:
                expires_at = str(expires_at).strip() or None

            inserted.append(
                {
                    "id": str(uuid.uuid4()),
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

        if not inserted:
            return []

        with self.conn:
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

        self._rebuild_user_index(user_id)
        return [self.get_memory(item["id"]) for item in inserted]

    def get_memory(self, memory_id: str, user_id: str | None = None) -> dict | None:
        where = ["id = ?", "deleted_at IS NULL"]
        params: list[Any] = [memory_id]
        if user_id is not None:
            where.append("user_id = ?")
            params.append(user_id)

        row = self.conn.execute(
            f"SELECT * FROM memories WHERE {' AND '.join(where)}",
            params,
        ).fetchone()
        if row is None or self._is_expired(row):
            return None
        return self._row_to_record(row)

    def list_memories(
        self,
        user_id: str,
        *,
        limit: int = 50,
        include_expired: bool = False,
    ) -> list[dict]:
        where = ["user_id = ?", "deleted_at IS NULL"]
        params: list[Any] = [user_id]
        if not include_expired:
            where.append("(expires_at IS NULL OR expires_at > ?)")
            params.append(_utc_now())

        rows = self.conn.execute(
            f"""
            SELECT * FROM memories
            WHERE {' AND '.join(where)}
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            [*params, max(1, limit)],
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def search_memory(
        self,
        user_id: str,
        query: str,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
        memory_types: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> list[dict]:
        """Semantic search over active memories for one user."""
        if top_k <= 0 or not query.strip():
            return []

        allowed_types = self._normalize_memory_types(memory_types)
        index, index_ids = self._get_user_index(user_id)
        if index is None or index.ntotal == 0:
            return []

        limit = min(index.ntotal, max(top_k * 50, 100))
        scores, indices = index.search(self._embed_texts([query]), limit)

        results: list[dict] = []
        seen: set[str] = set()
        for raw_score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0 or idx >= len(index_ids):
                continue

            memory_id = index_ids[int(idx)]
            if memory_id in seen:
                continue
            seen.add(memory_id)

            record = self.get_memory(memory_id, user_id=user_id)
            if record is None:
                continue
            if (
                allowed_types is not None
                and record["memory_type"] not in allowed_types
            ):
                continue

            score = self._normalize_vector_score(float(raw_score))
            if score < min_score:
                continue
            record["score"] = score
            results.append(record)

            if len(results) >= top_k:
                break

        results.sort(key=lambda item: item["score"], reverse=True)
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
        """Search one semantic memory layer such as profile, episode, or procedure."""
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
        """Search profile, episode, and procedure memories separately."""
        limits = {**DEFAULT_MEMORY_LAYER_TOP_K, **(layer_top_k or {})}
        layered: dict[str, list[dict]] = {}
        for layer in MEMORY_LAYERS:
            layered[layer] = self.search_memory_layer(
                user_id,
                query,
                layer,
                top_k=limits.get(layer, 3),
                min_score=min_score,
            )
        return layered

    def forget_memory(self, memory_id: str, user_id: str | None = None) -> bool:
        where = ["id = ?", "deleted_at IS NULL"]
        params: list[Any] = [memory_id]
        if user_id is not None:
            where.append("user_id = ?")
            params.append(user_id)

        affected_user_ids = self._memory_user_ids(memory_id, user_id=user_id)

        with self.conn:
            cursor = self.conn.execute(
                f"""
                UPDATE memories
                SET deleted_at = ?, updated_at = ?
                WHERE {' AND '.join(where)}
                """,
                [_utc_now(), _utc_now(), *params],
            )

        if cursor.rowcount <= 0:
            return False
        for affected_user_id in affected_user_ids:
            self._rebuild_user_index(affected_user_id)
        return True

    def clear_user_memory(self, user_id: str) -> int:
        with self.conn:
            cursor = self.conn.execute(
                """
                UPDATE memories
                SET deleted_at = ?, updated_at = ?
                WHERE user_id = ? AND deleted_at IS NULL
                """,
                [_utc_now(), _utc_now(), user_id],
            )

        if cursor.rowcount > 0:
            self._rebuild_user_index(user_id)
        return int(cursor.rowcount)

    def close(self) -> None:
        self.conn.close()

    def _init_db(self) -> None:
        with self.conn:
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
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_user_active
                ON memories(user_id, deleted_at, updated_at)
                """
            )

    def _active_rows(self, user_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT * FROM memories
            WHERE user_id = ?
              AND deleted_at IS NULL
              AND (expires_at IS NULL OR expires_at > ?)
            ORDER BY created_at ASC
            """,
            [user_id, _utc_now()],
        ).fetchall()

    def _active_count(self, user_id: str) -> int:
        row = self.conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM memories
            WHERE user_id = ?
              AND deleted_at IS NULL
              AND (expires_at IS NULL OR expires_at > ?)
            """,
            [user_id, _utc_now()],
        ).fetchone()
        return int(row["count"])

    def _memory_user_ids(self, memory_id: str, user_id: str | None = None) -> list[str]:
        where = ["id = ?", "deleted_at IS NULL"]
        params: list[Any] = [memory_id]
        if user_id is not None:
            where.append("user_id = ?")
            params.append(user_id)

        rows = self.conn.execute(
            f"SELECT DISTINCT user_id FROM memories WHERE {' AND '.join(where)}",
            params,
        ).fetchall()
        return [str(row["user_id"]) for row in rows]

    def _user_index_paths(self, user_id: str) -> tuple[Path, Path]:
        user_hash = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
        return (
            self.index_dir / f"{user_hash}.faiss",
            self.index_dir / f"{user_hash}.ids.json",
        )

    def _get_user_index(self, user_id: str) -> tuple[faiss.Index | None, list[str]]:
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
        if not index_path.exists():
            return None
        return faiss.read_index(str(index_path))

    def _load_index_ids(self, index_ids_path: Path) -> list[str]:
        if not index_ids_path.exists():
            return []
        try:
            payload = json.loads(index_ids_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, list):
            return []
        return [str(item) for item in payload]

    def _save_index_ids(self, index_ids_path: Path, index_ids: list[str]) -> None:
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
        active_count = self._active_count(user_id)
        if active_count == 0:
            return index is None and not index_ids
        if index is None:
            return False
        return index.ntotal == active_count == len(index_ids)

    def _rebuild_user_index(self, user_id: str) -> tuple[faiss.Index | None, list[str]]:
        index_path, index_ids_path = self._user_index_paths(user_id)
        rows = self._active_rows(user_id)
        if not rows:
            if index_path.exists():
                index_path.unlink()
            if index_ids_path.exists():
                index_ids_path.unlink()
            self._index_cache[user_id] = (None, [])
            return None, []

        vectors = self._embed_texts([row["content"] for row in rows])
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        index_ids = [row["id"] for row in rows]

        faiss.write_index(index, str(index_path))
        self._save_index_ids(index_ids_path, index_ids)
        self._index_cache[user_id] = (index, index_ids)
        return index, index_ids

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = self.embeddings.embed_documents(texts)
        matrix = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(matrix)
        return matrix

    def _normalize_vector_score(self, score: float) -> float:
        return max(0.0, min(1.0, (score + 1) / 2))

    def _normalize_memory_types(
        self,
        memory_types: set[str] | list[str] | tuple[str, ...] | None,
    ) -> set[str] | None:
        if memory_types is None:
            return None
        return {
            str(item)
            for item in memory_types
            if str(item) in MEMORY_TYPES
        }

    def _is_expired(self, row: sqlite3.Row) -> bool:
        expires_at = row["expires_at"]
        return bool(expires_at and str(expires_at) <= _utc_now())

    def _row_to_record(self, row: sqlite3.Row) -> dict:
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
            "memory_layer": memory_layer_for_type(row["memory_type"]),
            "importance": float(row["importance"]),
            "source": row["source"],
            "metadata": metadata,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "expires_at": row["expires_at"],
        }


@dataclass
class ExtractedMemory:
    content: str
    memory_type: str
    importance: float
    expires_at: str | None = None


class LLMMemoryExtractor:
    """Extract stable user memories from a finished conversation turn."""

    def __init__(
        self,
        model_name: str = "qwen3-max-2026-01-23",
        model: ChatTongyi | None = None,
        retry_policy: LLMRetryPolicy | None = None,
        trace_recorder: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.model = model or ChatTongyi(model=model_name, max_retries=0)
        self.retry_policy = retry_policy or LLMRetryPolicy()
        self.trace_recorder = trace_recorder
        self.system_prompt = SystemMessage(
            content=(
                "你是电商客服系统的长期记忆抽取器，只负责判断本轮对话中是否有值得长期保存的用户信息。"
                "只保存用户明确表达的、未来服务中可能复用且相对稳定的信息，例如尺码、身高体重、颜色偏好、"
                "材质过敏或排斥、穿着场景、风格偏好，以及用户明确要求你记住的事项。"
                "不要保存商品知识、客服回答、模型推测、一次性临时问题。"
                "不要保存密码、支付信息、身份证号、手机号、详细地址等敏感信息。"
                "输出必须是JSON对象，格式为："
                '{"memories":[{"content":"...","memory_type":"preference",'
                '"importance":0.7,"expires_at":null}]}。'
                "memory_type 只能是 profile、preference、constraint、instruction、episode、procedure。"
                "profile/preference/constraint/instruction 用于稳定用户画像和偏好；"
                "episode 用于有复用价值的历史事件摘要；"
                "procedure 用于用户明确要求长期遵循的可复用流程。"
                "importance 为0到1之间的数字。没有可保存内容时返回 {\"memories\":[]}。"
            )
        )

    def extract(
        self,
        *,
        user_message: str,
        assistant_message: str,
        existing_memories: list[dict] | None = None,
    ) -> list[ExtractedMemory]:
        existing_block = self._format_existing_memories(existing_memories or [])
        messages = [
            self.system_prompt,
            HumanMessage(
                content=(
                    "请从下面这轮电商客服对话中抽取需要长期记住的用户信息。\n\n"
                    f"已有相关记忆：\n{existing_block}\n\n"
                    f"用户消息：{user_message}\n\n"
                    f"客服回复：{assistant_message}"
                )
            ),
        ]
        response = invoke_with_retry(
            lambda: self.model.invoke(messages),
            retry_policy=self.retry_policy,
            operation="memory_extractor.invoke",
            on_failure=self._trace_retry_failure,
        )
        payload = self._parse_payload(self._coerce_content(response.content))
        return self._normalize_memories(payload.get("memories"))

    def _trace_retry_failure(self, event: dict[str, Any]) -> None:
        if self.trace_recorder is None:
            return
        self.trace_recorder.event(
            "model",
            "memory_extractor.model_retry",
            {"model_name": self.model_name, **event},
            level="warning" if event.get("will_retry") else "error",
        )

    def _format_existing_memories(self, memories: list[dict]) -> str:
        if not memories:
            return "无"
        return "\n".join(f"- {item['content']}" for item in memories[:8])

    def _coerce_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    parts.append(str(item["text"]))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(content)

    def _parse_payload(self, raw_response: str) -> dict[str, Any]:
        candidates = [raw_response]
        start = raw_response.find("{")
        end = raw_response.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.insert(0, raw_response[start : end + 1])

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        return {}

    def _normalize_memories(self, value: Any) -> list[ExtractedMemory]:
        if not isinstance(value, list):
            return []

        memories: list[ExtractedMemory] = []
        for item in value:
            if not isinstance(item, dict):
                continue

            content = str(item.get("content") or "").strip()
            if not content:
                continue

            memory_type = str(item.get("memory_type") or item.get("type") or "")
            if memory_type not in MEMORY_TYPES:
                memory_type = "preference"

            expires_at = item.get("expires_at")
            if expires_at is not None:
                expires_at = str(expires_at).strip() or None

            memories.append(
                ExtractedMemory(
                    content=content,
                    memory_type=memory_type,
                    importance=_clamp_importance(item.get("importance", 0.5)),
                    expires_at=expires_at,
                )
            )

        return memories[:5]
