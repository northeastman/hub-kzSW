from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _optional_env(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime configuration loaded from environment variables."""

    base_dir: Path
    skills_dir: Path
    data_dir: Path
    openai_api_key: str | None
    openai_base_url: str | None
    chat_model: str
    embedding_model: str
    embedding_dimension: int
    heartbeat_seconds: int
    skill_timeout_seconds: float
    top_k_skills: int

    @classmethod
    def from_env(cls, base_dir: Path | None = None) -> "Settings":
        root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

        def resolve_path(name: str, default: str) -> Path:
            path = Path(os.getenv(name, default))
            return (root / path).resolve() if not path.is_absolute() else path.resolve()

        return cls(
            base_dir=root,
            skills_dir=resolve_path("SKILLS_DIR", "skills"),
            data_dir=resolve_path("DATA_DIR", "data"),
            openai_api_key=_optional_env("OPENAI_API_KEY"),
            openai_base_url=_optional_env("OPENAI_BASE_URL"),
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "1536")),
            heartbeat_seconds=max(5, int(os.getenv("HEARTBEAT_SECONDS", "30"))),
            skill_timeout_seconds=max(
                0.1, float(os.getenv("SKILL_TIMEOUT_SECONDS", "30"))
            ),
            top_k_skills=max(1, int(os.getenv("TOP_K_SKILLS", "3"))),
        )

