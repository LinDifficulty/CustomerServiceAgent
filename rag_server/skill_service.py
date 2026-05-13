from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from .trace_service import TraceRecorder, preview_text
from .utils import coerce_bool

# ---- 常量定义 ----

# Anthropic Skill 的定义文件名，位于每个 skill 目录下
ANTHROPIC_SKILL_FILENAME = "SKILL.md"
# 默认的项目 skills 目录路径（相对于项目根目录）
DEFAULT_PROJECT_SKILLS_DIR = ".claude/skills"
# 单个 skill 支持文件的最大读取字节数（约 30KB），防止超大文件撑爆上下文
MAX_SKILL_FILE_BYTES = 30_000
# Skill 名称正则：只允许小写字母、数字和连字符，长度 1-64
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
# XML 标签正则：用于检测名称和描述中是否包含 XML 标签，防止注入攻击
XML_TAG_PATTERN = re.compile(r"<[A-Za-z/][^>\n]*>")
# 斜杠命令正则：匹配 "/skill-name" 格式，用于检测用户显式调用 skill
SLASH_SKILL_PATTERN = re.compile(r"^/([a-z0-9][a-z0-9-]{0,63})(?:\s+|$)")
# 保留名称片段：skill 名称中不能包含这些关键词，防止冒充系统 skill
RESERVED_SKILL_NAME_PARTS = {"anthropic", "claude"}


@dataclass(frozen=True)
class SkillDefinition:
    """从单个 SKILL.md 文件加载的 Anthropic 风格 Skill 定义。

    采用 frozen=True 确保数据不可变，支持 progressive disclosure 模式：
    先只暴露 frontmatter 元数据，用户按需加载完整内容。
    """

    name: str  # Skill 名称，必须与目录名一致
    description: str  # Skill 简要描述
    body: str  # SKILL.md 正文（frontmatter 之后的部分）
    path: Path  # SKILL.md 文件的绝对路径
    frontmatter: dict[str, Any]  # 解析后的 YAML frontmatter 元数据

    @property
    def directory(self) -> Path:
        """Skill 所在目录的路径。"""
        return self.path.parent

    @property
    def when_to_use(self) -> str:
        """何时应该使用该 skill 的条件说明，来自 frontmatter。"""
        return str(self.frontmatter.get("when_to_use") or "").strip()

    @property
    def allowed_tools(self) -> list[str]:
        """该 skill 允许使用的工具列表，支持中横线和下划线两种 key 名。

        值可以是列表或逗号分隔的字符串，兼容不同的 YAML 写法。
        """
        # 兼容两种 key 名：allowed-tools（kebab）和 allowed_tools（snake）
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
        """用户是否可以通过 '/skill-name' 命令显式调用该 skill。

        兼容两种 frontmatter key，默认为 True。
        """
        value = self.frontmatter.get("user-invocable")
        if value is None:
            value = self.frontmatter.get("user_invocable")
        if value is None:
            return True  # 默认允许用户调用
        return coerce_bool(value, default=True)

    @property
    def disable_model_invocation(self) -> bool:
        """是否禁止模型自动调用该 skill（仅允许用户显式调用）。

        兼容两种 frontmatter key，默认为 False。
        """
        value = self.frontmatter.get("disable-model-invocation")
        if value is None:
            value = self.frontmatter.get("disable_model_invocation")
        return coerce_bool(value, default=False)

    def discovery_line(self) -> str:
        """生成 skill 发现阶段的单行摘要信息。

        用于 discovery_prompt 中展示可用 skills 列表，
        包含名称、描述和可选的触发条件。
        """
        suffix = f" When to use: {self.when_to_use}" if self.when_to_use else ""
        return f"- /{self.name}: {self.description}{suffix}"

    def render(self, supporting_files: list[str]) -> str:
        """渲染 skill 的完整上下文文本，用于提供给 LLM。

        包含元数据、正文和支持文件列表。
        """
        lines = [
            f"Skill loaded: {self.name}",
            f"Description: {self.description}",
            f"Path: {self.path}",
        ]
        if self.when_to_use:
            lines.append(f"When to use: {self.when_to_use}")
        if self.allowed_tools:
            lines.append(f"Allowed tools: {', '.join(self.allowed_tools)}")

        # 正文部分：SKILL.md 的核心指令内容
        lines.extend(["", "SKILL.md instructions:", self.body.strip() or "(empty)"])
        # 支撑文件列表：告知 LLM 有哪些额外文件可读取
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
    """Skill 注册表：负责发现、加载和管理项目中的 Anthropic 风格 Skills。

    核心功能：
    - 从多个 skill 目录中扫描并加载所有 SKILL.md 文件
    - 支持 progressive disclosure（渐进披露）：先暴露元数据，按需加载完整内容
    - 支持用户显式调用（/skill-name）和模型自动发现
    - 提供 load_skill 和 read_skill_file 工具，用于按需读取 skill 内容
    """

    def __init__(self, skill_dirs: list[str | Path] | None = None) -> None:
        # 将传入的路径列表统一转换为 Path 对象
        self.skill_dirs = [Path(item) for item in (skill_dirs or [])]
        # 错误收集列表：记录发现/加载过程中的错误，但不中断整体流程
        self.errors: list[str] = []
        # 文件系统缓存：避免每次请求都扫描磁盘和解析 SKILL.md
        self._skills_cache: list[SkillDefinition] | None = None
        self._skills_cache_mtime: float = 0.0

    @classmethod
    def from_project_root(
        cls,
        project_root: str | Path = ".",
        extra_skill_dirs: list[str | Path] | None = None,
    ) -> SkillRegistry:
        """从项目根目录创建 SkillRegistry。

        默认会包含项目根目录下的 .claude/skills 目录，
        可通过 extra_skill_dirs 添加额外的 skill 搜索路径。
        """
        root = Path(project_root)
        # 默认搜索路径：项目根目录下的 .claude/skills
        skill_dirs: list[str | Path] = [root / DEFAULT_PROJECT_SKILLS_DIR]
        if extra_skill_dirs:
            skill_dirs.extend(extra_skill_dirs)
        return cls(skill_dirs)

    def _skills_files_mtime(self) -> float:
        """返回所有 skill 目录下 SKILL.md 文件的最大 mtime，用于缓存失效判断。"""
        max_mtime = 0.0
        for skill_dir in self.skill_dirs:
            base_dir = skill_dir.expanduser()
            if not base_dir.is_dir():
                continue
            for path in base_dir.glob(f"*/{ANTHROPIC_SKILL_FILENAME}"):
                mtime = path.stat().st_mtime
                if mtime > max_mtime:
                    max_mtime = mtime
        return max_mtime

    def list_skills(self) -> list[SkillDefinition]:
        """扫描所有 skill 目录，返回所有有效的 Skill 定义列表。

        首次调用扫描磁盘并缓存结果；后续调用仅在 SKILL.md 文件 mtime 变更时重新扫描。
        按 skill 名称的字母顺序排列。
        """
        # 检查缓存是否有效（文件未被修改）
        current_mtime = self._skills_files_mtime()
        if self._skills_cache is not None and current_mtime <= self._skills_cache_mtime:
            return self._skills_cache

        self.errors = []  # 每次扫描前清空错误列表
        skills: dict[str, SkillDefinition] = {}

        for skill_dir in self.skill_dirs:
            base_dir = skill_dir.expanduser()  # 展开 ~ 用户目录
            # 跳过不存在的目录
            if not base_dir.exists():
                continue
            # 非目录的路径记录错误后跳过
            if not base_dir.is_dir():
                self.errors.append(f"{base_dir} is not a directory")
                continue

            # 遍历所有子目录下的 SKILL.md 文件（按文件名排序以保证确定性）
            for path in sorted(base_dir.glob(f"*/{ANTHROPIC_SKILL_FILENAME}")):
                try:
                    # 解析并加载单个 SKILL.md 文件
                    skill = self._load_skill_file(path)
                except ValueError as error:
                    # 加载失败不影响其他 skill，记录错误后继续
                    self.errors.append(f"{path}: {error}")
                    continue

                # 同名 skill 去重：先到先得，后续记录错误并跳过
                if skill.name in skills:
                    self.errors.append(f"{path}: duplicate skill name '{skill.name}', ignored")
                    continue
                skills[skill.name] = skill

        # 更新缓存
        self._skills_cache = sorted(skills.values(), key=lambda item: item.name)
        self._skills_cache_mtime = current_mtime
        return self._skills_cache

    def get_skill(self, name: str) -> SkillDefinition | None:
        """根据名称查找一个 Skill 定义。

        名称匹配忽略大小写和前后空格。未找到时返回 None。
        """
        normalized_name = name.strip().lower()
        for skill in self.list_skills():
            if skill.name == normalized_name:
                return skill
        return None

    def discovery_prompt(self) -> str:
        """生成 Skill 发现的提示词片段，用于注入到 Agent 系统提示词中。

        只列出允许模型自动调用的 skill（disable_model_invocation=False 的），
        通过 progressive disclosure 模式：先只暴露元数据，
        Agent 需要时再调用 load_skill 加载完整内容。
        """
        # 过滤掉禁止模型自动调用的 skill
        skills = [skill for skill in self.list_skills() if not skill.disable_model_invocation]
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
        """检测用户消息是否以 '/skill-name' 格式显式调用某项 skill。

        如果匹配成功且该 skill 允许用户调用（user_invocable=True），
        返回 skill 名称；否则返回 None。
        """
        match = SLASH_SKILL_PATTERN.match(text.strip())
        if match is None:
            return None
        name = match.group(1).lower()
        skill = self.get_skill(name)
        if skill is None or not skill.user_invocable:
            return None
        return name

    def render_explicit_skill_context(self, text: str) -> str:
        """为显式 skill 调用生成完整的上下文提示词。

        当用户使用 '/skill-name' 命令时，生成包含完整 SKILL.md 内容
        和支撑文件列表的上下文文本，供 Agent 在处理该消息时使用。
        """
        name = self.explicit_invocation_name(text)
        if name is None:
            return ""

        skill = self.get_skill(name)
        if skill is None:
            return ""

        # 生成上下文，提示 Agent 忽略命令前缀并遵循 skill 指令
        return "\n".join(
            [
                f"用户显式调用了 /{name} skill。请忽略消息开头的 /{name} 命令前缀，并优先遵循下面的 skill 内容。",
                "",
                skill.render(self.list_supporting_files(name)),
            ]
        )

    def load_skill(self, name: str) -> str:
        """加载指定 skill 的完整内容（渲染后的文本）。

        如果 skill 不存在，返回友好的错误消息提示可用 skills。
        """
        skill = self.get_skill(name)
        if skill is None:
            return self._missing_skill_message(name)
        return skill.render(self.list_supporting_files(skill.name))

    def list_supporting_files(self, name: str, max_files: int = 80) -> list[str]:
        """列出 skill 目录下的所有支撑文件（排除 SKILL.md 本身）。

        限定最多返回 max_files 个文件，并且跳过隐藏文件/目录（以 . 开头）。
        返回相对于 skill 目录的文件路径列表。
        """
        skill = self.get_skill(name)
        if skill is None:
            return []

        files: list[str] = []
        # 递归遍历 skill 目录下的所有文件
        for path in sorted(skill.directory.rglob("*")):
            # 跳过 SKILL.md 本身和非文件条目
            if not path.is_file() or path.name == ANTHROPIC_SKILL_FILENAME:
                continue
            # 跳过隐藏文件/目录（路径中任何部分以 . 开头）
            if any(part.startswith(".") for part in path.relative_to(skill.directory).parts):
                continue
            # 转为 POSIX 风格的相对路径
            files.append(path.relative_to(skill.directory).as_posix())
            if len(files) >= max_files:
                break
        return files

    def read_supporting_file(self, name: str, relative_path: str) -> str:
        """读取 skill 目录下的支撑文件内容。

        包含多重安全检查：路径穿越检查、文件大小截断、编码容错。
        如果文件不存在或为 SKILL.md，返回相应的错误/提示消息。
        """
        skill = self.get_skill(name)
        if skill is None:
            return self._missing_skill_message(name)

        # 安全检查：防止 relative_path 穿越到 skill 目录之外
        try:
            path = _safe_child_path(skill.directory, relative_path)
        except ValueError as error:
            return f"无法读取 skill 文件：{error}"

        # 如果试图读取 SKILL.md，引导用户使用正确的工具
        if path.name == ANTHROPIC_SKILL_FILENAME:
            return "请使用 load_skill(name) 读取 SKILL.md。"
        if not path.exists() or not path.is_file():
            return f"skill 文件不存在：{relative_path}"

        # 读取文件并截断超长内容
        raw = path.read_bytes()
        truncated = len(raw) > MAX_SKILL_FILE_BYTES
        if truncated:
            raw = raw[:MAX_SKILL_FILE_BYTES]

        # 解码为 UTF-8，无法解码时用替换字符占位
        content = raw.decode("utf-8", errors="replace")
        if truncated:
            content += "\n\n[内容已截断]"
        return f"Skill file: {skill.name}/{relative_path}\n\n{content}"

    def _missing_skill_message(self, name: str) -> str:
        """生成 skill 不存在时的友好错误消息，附带所有可用 skills 列表。"""
        available = ", ".join(skill.name for skill in self.list_skills()) or "无"
        return f"未找到 skill: {name}。可用 skills: {available}"

    def _load_skill_file(self, path: Path) -> SkillDefinition:
        """从 SKILL.md 文件路径加载并解析一个 Skill 定义。

        步骤：
        1. 读取文件文本
        2. 分离 YAML frontmatter 和正文
        3. 校验 frontmatter 元数据的有效性
        4. 确保 frontmatter name 与目录名一致
        5. 构建并返回 SkillDefinition
        """
        text = path.read_text(encoding="utf-8")
        # 分离 frontmatter 和 body
        frontmatter, body = _parse_frontmatter(text)

        # 提取并清理元数据字段
        raw_name = str(frontmatter.get("name") or "").strip().lower()
        raw_description = str(frontmatter.get("description") or "").strip()
        # 校验名称和描述的合法性（格式、长度、禁止内容）
        _validate_skill_metadata(raw_name, raw_description)

        # frontmatter 中的 name 必须与 skill 目录名一致
        directory_name = path.parent.name
        if raw_name != directory_name:
            raise ValueError(f"frontmatter name '{raw_name}' must match directory '{directory_name}'")

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
    trace_recorder: TraceRecorder | None = None,
):
    """将 Skill 按需加载能力暴露为 LangChain 工具函数。

    返回两个工具函数：
    - load_skill(name)：按名称加载完整 SKILL.md 指令
    - read_skill_file(name, relative_path)：读取 skill 目录下的支撑文件

    这两个工具遵循 progressive disclosure 模式：
    先由 Agent 发现有哪些 skill（通过 discovery_prompt），
    再按需调用工具加载具体内容。
    """

    @tool(description="按名称加载一个 Anthropic-style skill 的完整 SKILL.md 指令。")
    def load_skill(name: str) -> str:
        """加载指定 skill 的完整 SKILL.md 指令内容。"""
        result = skill_registry.load_skill(name)
        # 记录工具调用追踪事件
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
        """读取 skill 目录下的支撑文件。

        如果未提供 relative_path，会列出所有可读取的支撑文件供选择。
        """
        if not relative_path.strip():
            # 未提供路径时，列出可用的支撑文件
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
            # 记录缺少路径的告警追踪事件
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
        # 有路径时读取具体文件
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


# ---- 内部辅助函数 ----


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """从 SKILL.md 文件中分离 YAML frontmatter 和正文。

    YAML frontmatter 由前后各一个 '---' 标记包裹。
    返回 (frontmatter_dict, body_text)。
    """
    lines = text.splitlines()
    # SKILL.md 必须以 YAML frontmatter 开头（第一行是 '---'）
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md must start with YAML frontmatter")

    # 找到关闭的 '---' 标记
    end_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = index
            break
    if end_index is None:
        raise ValueError("SKILL.md frontmatter must end with ---")

    # 解析 frontmatter 行（第一个 --- 之后、第二个 --- 之前）
    frontmatter = _parse_simple_yaml(lines[1:end_index])
    # 第二个 --- 之后的部分为正文
    body = "\n".join(lines[end_index + 1 :]).strip()
    return frontmatter, body


def _parse_simple_yaml(lines: list[str]) -> dict[str, Any]:
    """简单 YAML 解析器，支持标量、布尔、列表和内联数组。

    这是一个轻量级解析器，仅处理 SKILL.md frontmatter 中出现的
    YAML 子集，不依赖第三方 YAML 库以减少依赖。

    支持的语法：
    - key: value       (标量值)
    - key: true/false  (布尔值)
    - key:             (空值创建空列表)
    -   - item         (列表项，以 - 开头)
    - key: | / key: >  (多行文本标记)
    - key: [a, b]      (内联数组)
    """
    payload: dict[str, Any] = {}
    current_key: str | None = None  # 当前正在处理的键，用于多行值拼接

    for raw_line in lines:
        # 跳过空行和注释行（以 # 开头）
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        # 缩进的行：属于上一个键的多行值或列表项
        if raw_line.startswith((" ", "\t")) and current_key:
            item = raw_line.strip()
            if item.startswith("- "):
                # 列表项：追加到当前键的列表中
                existing = payload.setdefault(current_key, [])
                if not isinstance(existing, list):
                    existing = []
                    payload[current_key] = existing
                existing.append(_coerce_scalar(item[2:].strip()))
            elif isinstance(payload.get(current_key), str):
                # 多行字符串：拼接到当前键的值后面（避免首行出现前导换行）
                if payload[current_key]:
                    payload[current_key] = f"{payload[current_key]}\n{item}"
                else:
                    payload[current_key] = item
            continue

        # 新键值对行：必须包含 ':' 分隔符
        if ":" not in raw_line:
            continue

        # 按第一个 ':' 分割键和值
        key, raw_value = raw_line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        current_key = key  # 更新当前键，后续缩进行会归属于它

        if not value:
            # 值为空：创建一个空列表，后续缩进行可添加列表项
            payload[key] = []
            continue
        if value in {"|", ">"}:
            # YAML 多行文本标记：初始化为空字符串，后续缩进行拼接
            payload[key] = ""
            continue
        payload[key] = _coerce_scalar(value)

    return payload


def _coerce_scalar(value: str) -> Any:
    """将字符串标量转换为相应的 Python 类型。

    处理规则：
    - 空字符串保持不变
    - 引号包裹的字符串去引号
    - "true"/"false" 转换为布尔值
    - [a, b] 格式转为列表
    - 其他保持字符串原样
    """
    value = value.strip()
    if not value:
        return ""

    # 处理引号包裹的字符串（单引号或双引号）
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    # 处理布尔值
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    # 处理内联数组 [item1, item2, ...]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_coerce_scalar(item.strip()) for item in inner.split(",")]
    return value


def _validate_skill_metadata(name: str, description: str) -> None:
    """校验 skill 的 frontmatter 元数据是否合法。

    校验规则：
    - name 不能为空
    - name 必须匹配 SKILL_NAME_PATTERN（小写字母/数字/连字符，1-64 位）
    - name 不能包含保留词（anthropic, claude）
    - name 和 description 不能包含 XML 标签（防止注入攻击）
    - description 长度不超过 1024 字符
    """
    if not name:
        raise ValueError("frontmatter field 'name' is required")
    if not SKILL_NAME_PATTERN.fullmatch(name):
        raise ValueError("skill name must be lowercase letters, numbers, and hyphens only")
    # 检查是否包含保留名称片段，防止冒充系统 skill
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
    """安全解析相对路径，防止路径穿越攻击。

    确保解析后的绝对路径在 base_dir 的子树内，
    如果 relative_path 试图通过 '..' 等手法逃逸到
    skill 目录之外，则抛出 ValueError。
    """
    if not relative_path.strip():
        raise ValueError("relative_path must not be empty")

    base = base_dir.resolve()  # 解析为绝对路径
    target = (base / relative_path).resolve()  # 拼接并解析目标路径
    try:
        # 验证目标路径确实在 base 目录下
        target.relative_to(base)
    except ValueError as error:
        raise ValueError("relative_path escapes the skill directory") from error
    return target
