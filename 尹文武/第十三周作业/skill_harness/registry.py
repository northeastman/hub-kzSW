from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np
from pydantic import ValidationError

from .llm import AIService
from .models import LoadLayer, SkillManifest, SkillSnapshot


@dataclass(slots=True)
class LoadedSkill:
    manifest: SkillManifest
    root: Path
    layer: LoadLayer = LoadLayer.METADATA
    instructions: str | None = None
    resources: dict[str, str] = field(default_factory=dict)
    handler: Callable[..., Any] | None = None
    module: ModuleType | None = None

    def snapshot(self) -> SkillSnapshot:
        return SkillSnapshot(
            name=self.manifest.name,
            version=self.manifest.version,
            description=self.manifest.description,
            keywords=self.manifest.keywords,
            layer=self.layer,
            loaded_resources=sorted(self.resources),
        )


class SkillRegistry:
    """Discovers metadata cheaply and promotes only selected skills."""

    def __init__(self, skills_dir: Path, ai: AIService) -> None:
        self.skills_dir = skills_dir
        self.ai = ai
        self._skills: dict[str, LoadedSkill] = {}
        self._lock = asyncio.Lock()
        self._catalog_vectors: np.ndarray | None = None
        self._catalog_names: list[str] = []

    def discover(self) -> list[str]:
        """Layer 1: parse manifests only; never import skill code."""
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        found: dict[str, LoadedSkill] = {}
        errors: list[str] = []
        for path in sorted(self.skills_dir.glob("*/skill.json")):
            try:
                manifest = SkillManifest.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
                if not manifest.enabled:
                    continue
                existing = self._skills.get(manifest.name)
                found[manifest.name] = (
                    existing
                    if (
                        existing
                        and existing.root == path.parent.resolve()
                        and existing.manifest == manifest
                    )
                    else LoadedSkill(manifest=manifest, root=path.parent.resolve())
                )
            except (OSError, ValidationError, json.JSONDecodeError) as exc:
                errors.append(f"{path}: {exc}")
        self._skills = found
        self._catalog_vectors = None
        self._catalog_names = []
        return errors

    async def refresh(self) -> list[str]:
        """Serialize a live catalog refresh with layer promotion."""
        async with self._lock:
            return self.discover()

    def list(self) -> list[SkillSnapshot]:
        return [self._skills[name].snapshot() for name in sorted(self._skills)]

    def get(self, name: str) -> LoadedSkill:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise KeyError(f"Unknown skill: {name}") from exc

    async def ensure_layer(self, name: str, target: LoadLayer) -> LoadedSkill:
        async with self._lock:
            skill = self.get(name)
            if target >= LoadLayer.INSTRUCTIONS and skill.layer < LoadLayer.INSTRUCTIONS:
                self._load_instructions(skill)
                skill.layer = LoadLayer.INSTRUCTIONS
            if target >= LoadLayer.RUNTIME and skill.layer < LoadLayer.RUNTIME:
                self._load_runtime(skill)
                skill.layer = LoadLayer.RUNTIME
            # MEMORY is prepared by the executor after retrieving relevant memories.
            if target >= LoadLayer.MEMORY and skill.layer < LoadLayer.MEMORY:
                if skill.layer < LoadLayer.RUNTIME:
                    self._load_runtime(skill)
                skill.layer = LoadLayer.MEMORY
            return skill

    async def select(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Rank Layer-1 catalog entries without loading their instructions/code."""
        if not self._skills:
            return []
        if self._catalog_vectors is None:
            self._catalog_names = sorted(self._skills)
            texts = [
                " ".join(
                    [
                        self._skills[name].manifest.name,
                        self._skills[name].manifest.description,
                        *self._skills[name].manifest.keywords,
                    ]
                )
                for name in self._catalog_names
            ]
            self._catalog_vectors = await self.ai.embed(texts)
        query_vector = (await self.ai.embed([query]))[0]
        scores = self._catalog_vectors @ query_vector
        order = np.argsort(-scores)[: min(top_k, len(scores))]
        return [(self._catalog_names[int(i)], float(scores[int(i)])) for i in order]

    def _load_instructions(self, skill: LoadedSkill) -> None:
        path = self._safe_child(skill.root, skill.manifest.instructions)
        skill.instructions = path.read_text(encoding="utf-8")

    def _load_runtime(self, skill: LoadedSkill) -> None:
        if skill.instructions is None:
            self._load_instructions(skill)
        skill.resources = {
            resource: self._safe_child(skill.root, resource).read_text(encoding="utf-8")
            for resource in skill.manifest.resources
        }
        module_path_text, separator, function_name = skill.manifest.handler.partition(":")
        if not separator or not function_name:
            raise ValueError(
                f"{skill.manifest.name}: handler must be '<python-file>:<function>'"
            )
        module_path = self._safe_child(skill.root, module_path_text)
        module_name = f"_progressive_skill_{skill.manifest.name}_{abs(hash(module_path))}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load handler module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            handler = getattr(module, function_name)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        if not callable(handler):
            raise TypeError(f"{skill.manifest.name}: handler is not callable")
        skill.module = module
        skill.handler = handler

    @staticmethod
    def _safe_child(root: Path, relative: str) -> Path:
        candidate = (root / relative).resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError(f"Skill path escapes its directory: {relative}")
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate
