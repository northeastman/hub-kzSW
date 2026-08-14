from __future__ import annotations

import os
from dataclasses import dataclass


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数") from exc
    if value < 1:
        raise ValueError(f"{name} 必须大于或等于 1")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    agent_max_iterations: int = 20
    child_max_iterations: int = 12
    max_concurrent_children: int = 3

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise ValueError("必须设置 DEEPSEEK_API_KEY")
        return cls(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            agent_max_iterations=_positive_int("AGENT_MAX_ITERATIONS", 20),
            child_max_iterations=_positive_int("DELEGATION_MAX_ITERATIONS", 12),
            max_concurrent_children=_positive_int(
                "DELEGATION_MAX_CONCURRENT_CHILDREN", 3
            ),
        )
