from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag_server.skill_service import (
    SkillDefinition,
    SkillRegistry,
    _parse_frontmatter,
    _safe_child_path,
    _validate_skill_metadata,
)


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

    def test_get_nonexistent_skill_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = SkillRegistry(skill_dirs=[temp_dir])
            skill = registry.get_skill("no-such-skill")

        self.assertIsNone(skill)

    def test_load_skill_returns_rendered_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir) / "skills"
            _write_skill(skills_dir, "my-skill", "Test skill", "Step 1: do X.")
            registry = SkillRegistry(skill_dirs=[str(skills_dir)])
            rendered = registry.load_skill("my-skill")

        self.assertIn("Skill loaded: my-skill", rendered)
        self.assertIn("Step 1: do X.", rendered)

    def test_load_missing_skill_lists_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir) / "skills"
            _write_skill(skills_dir, "real-skill", "存在的skill")
            registry = SkillRegistry(skill_dirs=[str(skills_dir)])
            result = registry.load_skill("missing")

        self.assertIn("未找到 skill: missing", result)
        self.assertIn("real-skill", result)

    def test_discovery_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir) / "skills"
            _write_skill(skills_dir, "sizing-advice", "尺码建议")
            registry = SkillRegistry(skill_dirs=[str(skills_dir)])
            prompt = registry.discovery_prompt()

        self.assertIn("sizing-advice", prompt)
        self.assertIn("尺码建议", prompt)

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

    def test_supporting_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir) / "skills"
            _write_skill(skills_dir, "my-skill", "Test skill")
            (skills_dir / "my-skill" / "data.txt").write_text("extra data")
            registry = SkillRegistry(skill_dirs=[str(skills_dir)])
            files = registry.list_supporting_files("my-skill")

        self.assertEqual(files, ["data.txt"])

    def test_read_supporting_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir) / "skills"
            _write_skill(skills_dir, "my-skill", "Test skill")
            (skills_dir / "my-skill" / "info.txt").write_text("hello world")
            registry = SkillRegistry(skill_dirs=[str(skills_dir)])
            content = registry.read_supporting_file("my-skill", "info.txt")

        self.assertIn("hello world", content)

    def test_read_skill_md_via_read_supporting_file_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir) / "skills"
            _write_skill(skills_dir, "my-skill", "Test skill")
            registry = SkillRegistry(skill_dirs=[str(skills_dir)])
            result = registry.read_supporting_file("my-skill", "SKILL.md")

        self.assertIn("load_skill", result)


class FrontmatterTests(unittest.TestCase):
    def test_parse_valid_frontmatter(self) -> None:
        text = "---\nname: test-skill\ndescription: A test\n---\nBody text."
        frontmatter, body = _parse_frontmatter(text)

        self.assertEqual(frontmatter["name"], "test-skill")
        self.assertEqual(frontmatter["description"], "A test")
        self.assertEqual(body, "Body text.")

    def test_missing_frontmatter_raises(self) -> None:
        with self.assertRaises(ValueError):
            _parse_frontmatter("No frontmatter here")

    def test_unclosed_frontmatter_raises(self) -> None:
        with self.assertRaises(ValueError):
            _parse_frontmatter("---\nname: test\n")


class ValidationTests(unittest.TestCase):
    def test_valid_metadata(self) -> None:
        _validate_skill_metadata("sizing-advice", "A description")

    def test_empty_name_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_skill_metadata("", "desc")

    def test_reserved_name_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_skill_metadata("anthropic-tool", "desc")

    def test_invalid_name_pattern_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_skill_metadata("Bad_Name", "desc")

    def test_long_description_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_skill_metadata("my-skill", "x" * 1025)

    def test_xml_in_name_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_skill_metadata("<script>alert</script>", "desc")


class SafeChildPathTests(unittest.TestCase):
    def test_valid_child_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            (base / "file.txt").touch()
            result = _safe_child_path(base, "file.txt")
            self.assertEqual(result, (base / "file.txt").resolve())

    def test_traversal_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                _safe_child_path(Path(temp_dir), "../../etc/passwd")

    def test_empty_path_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                _safe_child_path(Path(temp_dir), "")


if __name__ == "__main__":
    unittest.main()
