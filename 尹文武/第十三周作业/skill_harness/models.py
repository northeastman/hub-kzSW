from __future__ import annotations

from enum import IntEnum
from typing import Any

from pydantic import BaseModel, Field


class LoadLayer(IntEnum):
    """A skill is promoted through these layers only when needed."""

    METADATA = 1
    INSTRUCTIONS = 2
    RUNTIME = 3
    MEMORY = 4


class SkillManifest(BaseModel):
    name: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")
    version: str = "0.1.0"
    description: str
    keywords: list[str] = Field(default_factory=list)
    instructions: str = "SKILL.md"
    handler: str
    resources: list[str] = Field(default_factory=list)
    enabled: bool = True


class SkillSnapshot(BaseModel):
    name: str
    version: str
    description: str
    keywords: list[str]
    layer: LoadLayer
    loaded_resources: list[str] = Field(default_factory=list)


class ExecuteRequest(BaseModel):
    input: str = Field(min_length=1)
    skill: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    memory_top_k: int = Field(default=3, ge=0, le=20)


class ExecuteResponse(BaseModel):
    skill: str
    output: Any
    layer: LoadLayer
    memory_used: int
    elapsed_ms: float


class MemorySearchRequest(BaseModel):
    query: str = Field(min_length=1)
    skill: str | None = None
    top_k: int = Field(default=5, ge=1, le=50)


class MemoryHit(BaseModel):
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)

