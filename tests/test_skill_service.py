"""Skill 服务单元测试。

测试技能注册表的发现（list_skills）、加载（load_skill/get_skill）、
frontmatter 解析（_parse_frontmatter）、元数据校验（_validate_skill_metadata）、
以及安全子路径解析（_safe_child_path）等核心功能。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag_server.skill_service import (
    SkillRegistry,
    build_skill_tools,
    _parse_frontmatter,
    _safe_child_path,
    _validate_skill_metadata,
)


# 便捷工具函数：在指定目录下创建一个 Skill（写入 SKILL.md）
def _write_skill(skills_dir: Path, name: str, description: str, body: str = "") -> Path:
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}",
        encoding="utf-8",
    )
    return skill_path


class SkillRegistryTests(unittest.TestCase):
    # 验证技能发现：从技能目录中扫描并列出所有 SKILL.md 定义的技能
    def test_discover_skills(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir) / "skills"
            _write_skill(skills_dir, "sizing-advice", "尺码建议")
            _write_skill(skills_dir, "care-guidance", "洗涤养护")
            registry = SkillRegistry(skill_dirs=[str(skills_dir)])
            skills = registry.list_skills()

        self.assertEqual(len(skills), 2)
        names = [s.name for s in skills]
        self.assertIn("sizing-advice", names)
        self.assertIn("care-guidance", names)

    # 验证获取技能：通过名称获取完整的技能对象（含 name, description, body）
    def test_get_skill(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir) / "skills"
            _write_skill(skills_dir, "test-skill", "A test skill", "Do something.")
            registry = SkillRegistry(skill_dirs=[str(skills_dir)])
            skill = registry.get_skill("test-skill")

        self.assertIsNotNone(skill)
        self.assertEqual(skill.name, "test-skill")
        self.assertEqual(skill.description, "A test skill")
        self.assertIn("Do something", skill.body)

    # 验证获取不存在的技能返回 None
    def test_get_nonexistent_skill_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = SkillRegistry(skill_dirs=[temp_dir])
            skill = registry.get_skill("no-such-skill")

        self.assertIsNone(skill)

    # 验证加载技能返回渲染后的内容（含 "Skill loaded" 标记和执行步骤）
    def test_load_skill_returns_rendered_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir) / "skills"
            _write_skill(skills_dir, "my-skill", "Test skill", "Step 1: do X.")
            registry = SkillRegistry(skill_dirs=[str(skills_dir)])
            rendered = registry.load_skill("my-skill")

        self.assertIn("Skill loaded: my-skill", rendered)
        self.assertIn("Step 1: do X.", rendered)

    # 验证加载不存在的技能时返回可用技能列表
    def test_load_missing_skill_lists_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir) / "skills"
            _write_skill(skills_dir, "real-skill", "存在的skill")
            registry = SkillRegistry(skill_dirs=[str(skills_dir)])
            result = registry.load_skill("missing")

        self.assertIn("未找到 skill: missing", result)
        self.assertIn("real-skill", result)

    # 验证发现提示词生成：包含技能名称和描述，用于注入 System Prompt
    def test_discovery_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir) / "skills"
            _write_skill(skills_dir, "sizing-advice", "尺码建议")
            registry = SkillRegistry(skill_dirs=[str(skills_dir)])
            prompt = registry.discovery_prompt()

        self.assertIn("sizing-advice", prompt)
        self.assertIn("尺码建议", prompt)

    # 验证显式调用匹配：如用户输入 "/sizing-advice 我穿什么码" 时能识别出技能名
    def test_explicit_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir) / "skills"
            _write_skill(skills_dir, "sizing-advice", "尺码建议")
            registry = SkillRegistry(skill_dirs=[str(skills_dir)])

            self.assertEqual(
                registry.explicit_invocation_name("/sizing-advice 我穿什么码"),
                "sizing-advice",
            )
            self.assertIsNone(registry.explicit_invocation_name("普通问题"))

    # 验证重名技能去重：多个目录中存在同名技能时只保留第一个，并记录错误
    def test_duplicate_skill_name_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dir1 = Path(temp_dir) / "d1"
            dir2 = Path(temp_dir) / "d2"
            _write_skill(dir1, "same-name", "first")
            _write_skill(dir2, "same-name", "second")
            registry = SkillRegistry(skill_dirs=[str(dir1), str(dir2)])
            skills = registry.list_skills()

        self.assertEqual(len(skills), 1)
        self.assertTrue(len(registry.errors) > 0)

    # 验证列出技能的附属文件（如 data.txt 等非 SKILL.md 文件）
    def test_supporting_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir) / "skills"
            _write_skill(skills_dir, "my-skill", "Test skill")
            (skills_dir / "my-skill" / "data.txt").write_text("extra data")
            registry = SkillRegistry(skill_dirs=[str(skills_dir)])
            files = registry.list_supporting_files("my-skill")

        self.assertEqual(files, ["data.txt"])

    # 验证读取技能的附属文件内容
    def test_read_supporting_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir) / "skills"
            _write_skill(skills_dir, "my-skill", "Test skill")
            (skills_dir / "my-skill" / "info.txt").write_text("hello world")
            registry = SkillRegistry(skill_dirs=[str(skills_dir)])
            content = registry.read_supporting_file("my-skill", "info.txt")

        self.assertIn("hello world", content)

    # 验证安全防护：通过 read_supporting_file 读取 SKILL.md 会被拦截，提示使用 load_skill
    def test_read_skill_md_via_read_supporting_file_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir) / "skills"
            _write_skill(skills_dir, "my-skill", "Test skill")
            registry = SkillRegistry(skill_dirs=[str(skills_dir)])
            result = registry.read_supporting_file("my-skill", "SKILL.md")

        self.assertIn("load_skill", result)

    # 验证 read_skill_file 工具：缺少 relative_path 参数时给出提示和可用文件列表
    def test_read_skill_file_tool_handles_missing_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir) / "skills"
            _write_skill(skills_dir, "my-skill", "Test skill")
            (skills_dir / "my-skill" / "info.txt").write_text("hello world")
            registry = SkillRegistry(skill_dirs=[str(skills_dir)])
            tools = {tool.name: tool for tool in build_skill_tools(registry)}
            result = tools["read_skill_file"].invoke({"name": "my-skill"})

        self.assertIn("缺少 relative_path", result)
        self.assertIn("info.txt", result)
        self.assertIn("load_skill", result)


class FrontmatterTests(unittest.TestCase):
    # 验证正确解析 YAML frontmatter，提取 name、description 和 body
    def test_parse_valid_frontmatter(self) -> None:
        text = "---\nname: test-skill\ndescription: A test\n---\nBody text."
        frontmatter, body = _parse_frontmatter(text)

        self.assertEqual(frontmatter["name"], "test-skill")
        self.assertEqual(frontmatter["description"], "A test")
        self.assertEqual(body, "Body text.")

    # 验证缺少 frontmatter 时抛出 ValueError
    def test_missing_frontmatter_raises(self) -> None:
        with self.assertRaises(ValueError):
            _parse_frontmatter("No frontmatter here")

    # 验证 frontmatter 未闭合（只有开头 `---` 无结尾 `---`）时抛出 ValueError
    def test_unclosed_frontmatter_raises(self) -> None:
        with self.assertRaises(ValueError):
            _parse_frontmatter("---\nname: test\n")


class ValidationTests(unittest.TestCase):
    # 验证合法的技能元数据不抛出异常
    def test_valid_metadata(self) -> None:
        _validate_skill_metadata("sizing-advice", "A description")

    # 验证空技能名抛出 ValueError
    def test_empty_name_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_skill_metadata("", "desc")

    # 验证保留名称（如 anthropic-tool）抛出 ValueError
    def test_reserved_name_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_skill_metadata("anthropic-tool", "desc")

    # 验证非法命名模式（如包含大写）抛出 ValueError
    def test_invalid_name_pattern_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_skill_metadata("Bad_Name", "desc")

    # 验证描述超长（>1024 字符）抛出 ValueError
    def test_long_description_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_skill_metadata("my-skill", "x" * 1025)

    # 验证名称中包含 XML/HTML 标签时抛出 ValueError（防注入）
    def test_xml_in_name_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_skill_metadata("<script>alert</script>", "desc")


class SafeChildPathTests(unittest.TestCase):
    # 验证合法的子路径能正确解析
    def test_valid_child_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            (base / "file.txt").touch()
            result = _safe_child_path(base, "file.txt")
            self.assertEqual(result, (base / "file.txt").resolve())

    # 验证路径穿越攻击（如 ../../etc/passwd）抛出 ValueError
    def test_traversal_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                _safe_child_path(Path(temp_dir), "../../etc/passwd")

    # 验证空路径参数抛出 ValueError
    def test_empty_path_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                _safe_child_path(Path(temp_dir), "")


if __name__ == "__main__":
    unittest.main()
