from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def coerce_message_content(content: Any) -> str:
    """Normalize LangChain message content to a plain string.
    将 LangChain 消息内容规范化为纯字符串格式。

    Handles str, list-of-dicts (multi-part messages), and arbitrary values.
    处理字符串、字典列表（多部分消息）和任意值类型。
    """
    if isinstance(content, str):
        # 如果已经是字符串，直接返回
        return content
    if isinstance(content, list):
        # 如果是列表（LangChain多部分消息格式），逐个提取文本
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                # 字典类型且有 text 字段，提取其文本值
                parts.append(str(item["text"]))
            else:
                # 其他类型直接转字符串
                parts.append(str(item))
        # 用换行符合并所有部分
        return "\n".join(parts)
    # 其他类型直接转字符串作为兜底
    return str(content)


def parse_json_object(raw_response: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from an LLM response string.
    从 LLM 返回的字符串中尽力提取 JSON 对象。
    LLM 经常在 JSON 前后附加说明文字，本函数会尝试从响应中定位并解析 JSON。
    """
    # 候选解析目标列表，优先尝试提取后的内容
    candidates = [raw_response]
    # 查找第一个 { 和最后一个 }，提取中间的 JSON 片段
    start = raw_response.find("{")
    end = raw_response.rfind("}")
    if start >= 0 and end > start:
        # 将提取到的 JSON 片段放在候选列表首位，优先尝试
        candidates.insert(0, raw_response[start : end + 1])

    # 逐个尝试解析候选字符串
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue  # 解析失败则尝试下一个候选
        if isinstance(payload, dict):
            return payload  # 成功解析为字典对象则返回
    # 所有尝试都失败时返回空字典作为兜底
    return {}


def coerce_bool(value: Any, *, default: bool = False) -> bool:
    """Coerce a value to bool, with support for common string representations.
    将任意值强制转为布尔值，支持常见的中英文字符串表示。
    例如 "是"、"true"、"yes"、"1" 等都视为 True。
    """
    if isinstance(value, bool):
        # 已经是布尔类型，直接返回
        return value
    if isinstance(value, str):
        # 字符串类型：去除空白、转为小写后匹配常见真值/假值表示
        normalized = value.strip().lower()
        # 匹配中英文真值表示
        if normalized in {"true", "yes", "on", "1", "是", "需要"}:
            return True
        # 匹配假值表示
        if normalized in {"false", "no", "off", "0"}:
            return False
    if value is None:
        # None 时使用默认值
        return default
    # 其他类型使用 Python 内置 bool() 转换
    return bool(value)


def utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def cache_key_or_none(cache: Any, category: str, payload: dict[str, Any]) -> str | None:
    """Generate a cache key, or return None if no cache is configured."""
    if cache is None:
        return None
    return cache.make_key(category, payload)


_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name: str) -> str:
    """Load a prompt template from the prompts/ directory.

    Args:
        name: The prompt filename (e.g. "system_prompt.txt").

    Returns:
        The prompt content as a string, with leading/trailing whitespace stripped.
    """
    file_path = _PROMPTS_DIR / name
    return file_path.read_text(encoding="utf-8").strip()
