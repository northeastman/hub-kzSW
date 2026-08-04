from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from typing import Any

from .config import Settings
from .llm import AIService
from .memory import SemanticMemory
from .models import ExecuteRequest, ExecuteResponse, LoadLayer, MemoryHit
from .registry import LoadedSkill, SkillRegistry


@dataclass(slots=True)
class SkillContext:
    skill_name: str
    instructions: str
    resources: dict[str, str]
    memories: list[MemoryHit]
    ai: AIService


class SkillExecutor:
    def __init__(
        self,
        settings: Settings,
        registry: SkillRegistry,
        memory: SemanticMemory,
        ai: AIService,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.memory = memory
        self.ai = ai

    async def execute(self, request: ExecuteRequest) -> ExecuteResponse:
        started = time.perf_counter()
        skill_name = request.skill or await self._choose_skill(request.input)
        skill = await self.registry.ensure_layer(skill_name, LoadLayer.RUNTIME)
        memories = await self.memory.search(
            request.input, top_k=request.memory_top_k, skill=skill_name
        )
        # Memory is a per-execution context, so promotion happens only after retrieval.
        skill = await self.registry.ensure_layer(skill_name, LoadLayer.MEMORY)
        output = await asyncio.wait_for(
            self._invoke(skill, request, memories),
            timeout=self.settings.skill_timeout_seconds,
        )
        await self.memory.add(
            self._memory_text(request.input, output),
            {"skill": skill_name, "arguments": request.arguments},
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return ExecuteResponse(
            skill=skill_name,
            output=output,
            layer=skill.layer,
            memory_used=len(memories),
            elapsed_ms=round(elapsed_ms, 3),
        )

    async def _choose_skill(self, query: str) -> str:
        ranked = await self.registry.select(query, self.settings.top_k_skills)
        if not ranked:
            raise LookupError("No enabled skills were discovered")
        return ranked[0][0]

    async def _invoke(
        self, skill: LoadedSkill, request: ExecuteRequest, memories: list[MemoryHit]
    ) -> Any:
        assert skill.handler is not None
        context = SkillContext(
            skill_name=skill.manifest.name,
            instructions=skill.instructions or "",
            resources=skill.resources,
            memories=memories,
            ai=self.ai,
        )
        if inspect.iscoroutinefunction(skill.handler):
            return await skill.handler(request.input, request.arguments, context)
        return await asyncio.to_thread(
            skill.handler, request.input, request.arguments, context
        )

    @staticmethod
    def _memory_text(user_input: str, output: Any) -> str:
        rendered = str(output)
        if len(rendered) > 4000:
            rendered = rendered[:4000] + "…"
        return f"input: {user_input}\noutput: {rendered}"

