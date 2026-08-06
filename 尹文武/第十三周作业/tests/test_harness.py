from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from skill_harness.config import Settings
from skill_harness.executor import SkillExecutor
from skill_harness.llm import AIService
from skill_harness.memory import SemanticMemory
from skill_harness.models import ExecuteRequest, LoadLayer
from skill_harness.registry import SkillRegistry


def make_settings(root: Path) -> Settings:
    return Settings(
        base_dir=root,
        skills_dir=root / "skills",
        data_dir=root / "data",
        openai_api_key=None,
        openai_base_url=None,
        chat_model="unused",
        embedding_model="unused",
        embedding_dimension=64,
        heartbeat_seconds=30,
        skill_timeout_seconds=3,
        top_k_skills=3,
    )


def write_test_skill(root: Path) -> None:
    skill = root / "skills" / "echo"
    skill.mkdir(parents=True)
    (skill / "skill.json").write_text(
        json.dumps(
            {
                "name": "echo",
                "description": "echo text",
                "keywords": ["repeat"],
                "handler": "handler.py:run",
                "resources": ["prefix.txt"],
            }
        ),
        encoding="utf-8",
    )
    (skill / "SKILL.md").write_text("Echo the input.", encoding="utf-8")
    (skill / "prefix.txt").write_text("result:", encoding="utf-8")
    (skill / "handler.py").write_text(
        "def run(text, arguments, context):\n"
        "    return context.resources['prefix.txt'] + text\n",
        encoding="utf-8",
    )


class HarnessTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        write_test_skill(self.root)
        self.settings = make_settings(self.root)
        self.ai = AIService(self.settings)
        self.registry = SkillRegistry(self.settings.skills_dir, self.ai)
        self.memory = SemanticMemory(
            self.settings.data_dir, self.settings.embedding_dimension, self.ai
        )
        self.executor = SkillExecutor(
            self.settings, self.registry, self.memory, self.ai
        )

    async def asyncTearDown(self) -> None:
        self.temporary.cleanup()

    async def test_progressive_loading(self) -> None:
        self.assertEqual(self.registry.discover(), [])
        skill = self.registry.get("echo")
        self.assertEqual(skill.layer, LoadLayer.METADATA)
        self.assertIsNone(skill.instructions)
        self.assertIsNone(skill.handler)

        await self.registry.ensure_layer("echo", LoadLayer.INSTRUCTIONS)
        self.assertEqual(skill.layer, LoadLayer.INSTRUCTIONS)
        self.assertIsNotNone(skill.instructions)
        self.assertIsNone(skill.handler)

        await self.registry.ensure_layer("echo", LoadLayer.RUNTIME)
        self.assertEqual(skill.layer, LoadLayer.RUNTIME)
        self.assertTrue(callable(skill.handler))
        self.assertEqual(skill.resources["prefix.txt"], "result:")

    async def test_execution_adds_searchable_memory(self) -> None:
        self.registry.discover()
        response = await self.executor.execute(
            ExecuteRequest(input="hello", skill="echo")
        )
        self.assertEqual(response.output, "result:hello")
        self.assertEqual(response.layer, LoadLayer.MEMORY)
        self.assertEqual(self.memory.size, 1)
        hits = await self.memory.search("hello", skill="echo")
        self.assertEqual(len(hits), 1)

    async def test_memory_persists_under_unicode_path(self) -> None:
        unicode_directory = self.root / "中文数据"
        memory = SemanticMemory(
            unicode_directory, self.settings.embedding_dimension, self.ai
        )
        await memory.add("可持久化的记忆", {"skill": "echo"})
        memory.save()

        restored = SemanticMemory(
            unicode_directory, self.settings.embedding_dimension, self.ai
        )
        self.assertEqual(restored.size, 1)
        hits = await restored.search("可持久化的记忆", skill="echo")
        self.assertEqual(len(hits), 1)

    async def test_selector_does_not_promote_skills(self) -> None:
        self.registry.discover()
        ranked = await self.registry.select("please repeat this")
        self.assertEqual(ranked[0][0], "echo")
        self.assertEqual(self.registry.get("echo").layer, LoadLayer.METADATA)

    async def test_manifest_change_returns_skill_to_cold_state(self) -> None:
        self.registry.discover()
        await self.registry.ensure_layer("echo", LoadLayer.RUNTIME)
        manifest_path = self.root / "skills" / "echo" / "skill.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["version"] = "2.0.0"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        await self.registry.refresh()
        skill = self.registry.get("echo")
        self.assertEqual(skill.manifest.version, "2.0.0")
        self.assertEqual(skill.layer, LoadLayer.METADATA)
        self.assertIsNone(skill.handler)


if __name__ == "__main__":
    unittest.main()
