from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from .trace_service import preview_text

ANTHROPIC_SKILL_FILENAME = "SKILL.md"
DEFAULT_PROJECT_SKILLS_DIR = ".claude/skills"
MAX_SKILL_FILE_BYTES = 30_000
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
XML_TAG_PATTERN = re.compile(r"<[A-Za-z/][^>\n]*>")
SLASH_SKILL_PATTERN = re.compile(r"^/([a-z0-9][a-z0-9-]{0,63})(?:\s+|$)")
RESERVED_SKILL_NAME_PARTS = {"anthropic", "claude"}


@dataclass(frozen=True)
class SkillDefinition:
    """Anthropic-style skill loaded from one SKILL.md file."""

    name: str
    description: str
    body: str
    path: Path
    frontmatter: dict[str, Any]

    @property
    def directory(self) -> Path:
        return self.path.parent

    @property
    def when_to_use(self) -> str:
        return str(self.frontmatter.get("when_to_use") or "").strip()

    @property
    def allowed_tools(self) -> list[str]:
        value = self.frontmatter.get("allowed-tools")
        if value is None:
            value = self.frontmatter.get("allowed_tools")
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return []

    @property
    def user_invocable(self) -> bool:
        value = self.frontmatter.get("user-invocable")
        if value is None:
            value = self.frontmatter.get("user_invocable")
        if value is None:
            return True
        return _coerce_bool(value, default=True)

    @property
    def disable_model_invocation(self) -> bool:
        value = self.frontmatter.get("disable-model-invocation")
        if value is None:
            value = self.frontmatter.get("disable_model_invocation")
        return _coerce_bool(value, default=False)

    def discovery_line(self) -> str:
        suffix = f" When to use: {self.when_to_use}" if self.when_to_use else ""
        return f"- /{self.name}: {self.description}{suffix}"

    def render(self, supporting_files: list[str]) -> str:
        lines = [
            f"Skill loaded: {self.name}",
            f"Description: {self.description}",
            f"Path: {self.path}",
        ]
        if self.when_to_use:
            lines.append(f"When to use: {self.when_to_use}")
        if self.allowed_tools:
            lines.append(f"Allowed tools: {', '.join(self.allowed_tools)}")

        lines.extend(["", "SKILL.md instructions:", self.body.strip() or "(empty)"])
        if supporting_files:
            lines.extend(
                [
                    "",
                    "Supporting files available through read_skill_file:",
                    *[f"- {item}" for item in supporting_files],
                ]
            )
        return "\n".join(lines)


class SkillRegistry:
    """Discover and load Anthropic-style skills from project skill directories."""

    def __init__(self, skill_dirs: list[str | Path] | None = None) -> None:
        self.skill_dirs = [Path(item) for item in (skill_dirs or [])]
        self.errors: list[str] = []

    @classmethod
    def from_project_root(
        cls,
        project_root: str | Path = ".",
        extra_skill_dirs: list[str | Path] | None = None,
    ) -> SkillRegistry:
        root = Path(project_root)
        skill_dirs: list[str | Path] = [root / DEFAULT_PROJECT_SKILLS_DIR]
        if extra_skill_dirs:
            skill_dirs.extend(extra_skill_dirs)
        return cls(skill_dirs)

    def list_skills(self) -> list[SkillDefinition]:
        self.errors = []
        skills: dict[str, SkillDefinition] = {}

        for skill_dir in self.skill_dirs:
            base_dir = skill_dir.expanduser()
            if not base_dir.exists():
                continue
            if not base_dir.is_dir():
                self.errors.append(f"{base_dir} is not a directory")
                continue

            for path in sorted(base_dir.glob(f"*/{ANTHROPIC_SKILL_FILENAME}")):
                try:
                    skill = self._load_skill_file(path)
                except ValueError as error:
                    self.errors.append(f"{path}: {error}")
                    continue

                if skill.name in skills:
                    self.errors.append(
                        f"{path}: duplicate skill name '{skill.name}', ignored"
                    )
                    continue
                skills[skill.name] = skill

        return sorted(skills.values(), key=lambda item: item.name)

    def get_skill(self, name: str) -> SkillDefinition | None:
        normalized_name = name.strip().lower()
        for skill in self.list_skills():
            if skill.name == normalized_name:
                return skill
        return None

    def discovery_prompt(self) -> str:
        skills = [
            skill
            for skill in self.list_skills()
            if not skill.disable_model_invocation
        ]
        if not skills:
            return ""

        return "\n".join(
            [
                "可用 Anthropic-style Skills 如下。这里仅列出 frontmatter 元数据；"
                "当用户问题匹配某个 skill，或用户使用 /skill-name 显式调用时，"
                "先调用 load_skill(name) 读取完整 SKILL.md，再按其中说明执行。",
                "如果已加载的 SKILL.md 提到额外文件，再调用 read_skill_file(name, relative_path)。",
                *[skill.discovery_line() for skill in skills],
            ]
        )

    def explicit_invocation_name(self, text: str) -> str | None:
        match = SLASH_SKILL_PATTERN.match(text.strip())
        if match is None:
            return None
        name = match.group(1).lower()
        skill = self.get_skill(name)
        if skill is None or not skill.user_invocable:
            return None
        return name

    def render_explicit_skill_context(self, text: str) -> str:
        name = self.explicit_invocation_name(text)
        if name is None:
            return ""

        skill = self.get_skill(name)
        if skill is None:
            return ""

        return "\n".join(
            [
                f"用户显式调用了 /{name} skill。请忽略消息开头的 /{name} 命令前缀，"
                "并优先遵循下面的 skill 内容。",
                "",
                skill.render(self.list_supporting_files(name)),
            ]
        )

    def load_skill(self, name: str) -> str:
        skill = self.get_skill(name)
        if skill is None:
            return self._missing_skill_message(name)
        return skill.render(self.list_supporting_files(skill.name))

    def list_supporting_files(self, name: str, max_files: int = 80) -> list[str]:
        skill = self.get_skill(name)
        if skill is None:
            return []

        files: list[str] = []
        for path in sorted(skill.directory.rglob("*")):
            if not path.is_file() or path.name == ANTHROPIC_SKILL_FILENAME:
                continue
            if any(part.startswith(".") for part in path.relative_to(skill.directory).parts):
                continue
            files.append(path.relative_to(skill.directory).as_posix())
            if len(files) >= max_files:
                break
        return files

    def read_supporting_file(self, name: str, relative_path: str) -> str:
        skill = self.get_skill(name)
        if skill is None:
            return self._missing_skill_message(name)

        try:
            path = _safe_child_path(skill.directory, relative_path)
        except ValueError as error:
            return f"无法读取 skill 文件：{error}"

        if path.name == ANTHROPIC_SKILL_FILENAME:
            return "请使用 load_skill(name) 读取 SKILL.md。"
        if not path.exists() or not path.is_file():
            return f"skill 文件不存在：{relative_path}"

        raw = path.read_bytes()
        truncated = len(raw) > MAX_SKILL_FILE_BYTES
        if truncated:
            raw = raw[:MAX_SKILL_FILE_BYTES]

        content = raw.decode("utf-8", errors="replace")
        if truncated:
            content += "\n\n[内容已截断]"
        return f"Skill file: {skill.name}/{relative_path}\n\n{content}"

    def _missing_skill_message(self, name: str) -> str:
        available = ", ".join(skill.name for skill in self.list_skills()) or "无"
        return f"未找到 skill: {name}。可用 skills: {available}"

    def _load_skill_file(self, path: Path) -> SkillDefinition:
        text = path.read_text(encoding="utf-8")
        frontmatter, body = _parse_frontmatter(text)

        raw_name = str(frontmatter.get("name") or "").strip().lower()
        raw_description = str(frontmatter.get("description") or "").strip()
        _validate_skill_metadata(raw_name, raw_description)

        directory_name = path.parent.name
        if raw_name != directory_name:
            raise ValueError(
                f"frontmatter name '{raw_name}' must match directory '{directory_name}'"
            )

        return SkillDefinition(
            name=raw_name,
            description=raw_description,
            body=body,
            path=path,
            frontmatter=frontmatter,
        )


def build_skill_tools(
    skill_registry: SkillRegistry,
    *,
    trace_recorder: Any | None = None,
):
    """Expose Anthropic-style progressive skill loading as LangChain tools."""

    @tool(description="按名称加载一个 Anthropic-style skill 的完整 SKILL.md 指令。")
    def load_skill(name: str) -> str:
        """Load the full SKILL.md instructions for an available skill."""
        result = skill_registry.load_skill(name)
        if trace_recorder is not None:
            trace_recorder.event(
                "skill",
                "skill.load_skill",
                {
                    "name": name,
                    "skill_name": str(name).strip().lower(),
                    "result_preview": preview_text(result),
                },
            )
        return result

    @tool(description="读取已加载 skill 目录下的支撑文件；relative_path 是支撑文件相对路径。")
    def read_skill_file(name: str, relative_path: str = "") -> str:
        """Read a supporting file from a skill directory."""
        if not relative_path.strip():
            available_files = skill_registry.list_supporting_files(name)
            if available_files:
                files = ", ".join(available_files)
                result = (
                    "read_skill_file 缺少 relative_path。"
                    f"可读取的支撑文件: {files}。"
                    "如果只是要读取完整 SKILL.md，请调用 load_skill(name)。"
                )
            else:
                result = (
                    "read_skill_file 缺少 relative_path。"
                    "当前 skill 没有可读取的支撑文件；"
                    "如果只是要读取完整 SKILL.md，请调用 load_skill(name)。"
                )
            if trace_recorder is not None:
                trace_recorder.event(
                    "skill",
                    "skill.read_skill_file_missing_path",
                    {
                        "name": name,
                        "skill_name": str(name).strip().lower(),
                        "available_files": available_files,
                        "result_preview": preview_text(result),
                    },
                    level="warning",
                )
            return result
        result = skill_registry.read_supporting_file(name, relative_path)
        if trace_recorder is not None:
            trace_recorder.event(
                "skill",
                "skill.read_skill_file",
                {
                    "name": name,
                    "skill_name": str(name).strip().lower(),
                    "relative_path": relative_path,
                    "result_preview": preview_text(result),
                },
            )
        return result

    return [load_skill, read_skill_file]


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md must start with YAML frontmatter")

    end_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = index
            break
    if end_index is None:
        raise ValueError("SKILL.md frontmatter must end with ---")

    frontmatter = _parse_simple_yaml(lines[1:end_index])
    body = "\n".join(lines[end_index + 1 :]).strip()
    return frontmatter, body


def _parse_simple_yaml(lines: list[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    current_key: str | None = None

    for raw_line in lines:
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        if raw_line.startswith((" ", "\t")) and current_key:
            item = raw_line.strip()
            if item.startswith("- "):
                existing = payload.setdefault(current_key, [])
                if not isinstance(existing, list):
                    existing = []
                    payload[current_key] = existing
                existing.append(_coerce_scalar(item[2:].strip()))
            elif isinstance(payload.get(current_key), str):
                payload[current_key] = f"{payload[current_key]}\n{item}"
            continue

        if ":" not in raw_line:
            continue

        key, raw_value = raw_line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        current_key = key

        if not value:
            payload[key] = []
            continue
        if value in {"|", ">"}:
            payload[key] = ""
            continue
        payload[key] = _coerce_scalar(value)

    return payload


def _coerce_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""

    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_coerce_scalar(item.strip()) for item in inner.split(",")]
    return value


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
    return default


def _validate_skill_metadata(name: str, description: str) -> None:
    if not name:
        raise ValueError("frontmatter field 'name' is required")
    if not SKILL_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "skill name must be lowercase letters, numbers, and hyphens only"
        )
    if any(part in name for part in RESERVED_SKILL_NAME_PARTS):
        raise ValueError("skill name must not contain reserved words")
    if XML_TAG_PATTERN.search(name):
        raise ValueError("skill name must not contain XML tags")

    if not description:
        raise ValueError("frontmatter field 'description' is required")
    if len(description) > 1024:
        raise ValueError("skill description is too long")
    if XML_TAG_PATTERN.search(description):
        raise ValueError("skill description must not contain XML tags")


def _safe_child_path(base_dir: Path, relative_path: str) -> Path:
    if not relative_path.strip():
        raise ValueError("relative_path must not be empty")

    base = base_dir.resolve()
    target = (base / relative_path).resolve()
    try:
        target.relative_to(base)
    except ValueError as error:
        raise ValueError("relative_path escapes the skill directory") from error
    return target
