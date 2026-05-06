from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag_server.skill_service import SkillRegistry


class SkillRegistryTest(unittest.TestCase):
    def test_discovers_valid_anthropic_style_skill(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir) / ".claude" / "skills"
            skill_dir = skills_dir / "sample-skill"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "\n".join(
                    [
                        "---",
                        "name: sample-skill",
                        "description: Use this skill for sample tasks.",
                        "when_to_use: 用户明确请求 sample 时使用。",
                        "allowed-tools:",
                        "  - search_product_knowledge",
                        "---",
                        "",
                        "# Sample Skill",
                        "",
                        "Follow the sample workflow.",
                    ]
                ),
                encoding="utf-8",
            )
            (skill_dir / "notes.md").write_text("extra notes", encoding="utf-8")

            registry = SkillRegistry.from_project_root(temp_dir)
            skills = registry.list_skills()

            self.assertEqual([item.name for item in skills], ["sample-skill"])
            self.assertEqual(
                skills[0].allowed_tools,
                ["search_product_knowledge"],
            )
            self.assertIn("sample-skill", registry.discovery_prompt())
            self.assertIn("notes.md", registry.load_skill("sample-skill"))
            self.assertIn(
                "extra notes",
                registry.read_supporting_file("sample-skill", "notes.md"),
            )

    def test_rejects_invalid_skill_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir) / ".claude" / "skills"
            skill_dir = skills_dir / "bad-skill"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "\n".join(
                    [
                        "---",
                        "name: Bad Skill",
                        "description: Invalid name.",
                        "---",
                        "body",
                    ]
                ),
                encoding="utf-8",
            )

            registry = SkillRegistry.from_project_root(temp_dir)

            self.assertEqual(registry.list_skills(), [])
            self.assertTrue(registry.errors)

    def test_refuses_supporting_file_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir) / ".claude" / "skills"
            skill_dir = skills_dir / "safe-skill"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "\n".join(
                    [
                        "---",
                        "name: safe-skill",
                        "description: Use this skill for safe tasks.",
                        "---",
                        "body",
                    ]
                ),
                encoding="utf-8",
            )

            registry = SkillRegistry.from_project_root(temp_dir)

            self.assertIn(
                "escapes the skill directory",
                registry.read_supporting_file("safe-skill", "../secret.txt"),
            )


if __name__ == "__main__":
    unittest.main()
