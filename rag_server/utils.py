from __future__ import annotations

import json
from typing import Any


def coerce_message_content(content: Any) -> str:
    """Normalize LangChain message content to a plain string.

    Handles str, list-of-dicts (multi-part messages), and arbitrary values.
    """
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


def parse_json_object(raw_response: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from an LLM response string."""
    candidates = [raw_response]
    start = raw_response.find("{")
    end = raw_response.rfind("}")
    if start >= 0 and end > start:
        candidates.insert(0, raw_response[start : end + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def coerce_bool(value: Any, *, default: bool = False) -> bool:
    """Coerce a value to bool, with support for common string representations."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1", "是", "需要"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
    if value is None:
        return default
    return bool(value)
