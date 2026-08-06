"""OpenAI 兼容模型配置。

API Key 只从环境变量读取，不写入代码或配置文件。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    display_name: str
    api_key_env: str
    base_url: str
    model: str


PROVIDERS: dict[str, dict[str, str]] = {
    "deepseek": {
        "display_name": "DeepSeek",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "qwen": {
        "display_name": "Qwen (DashScope)",
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
}


def get_llm_config() -> LLMConfig:
    provider = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
    if provider not in PROVIDERS:
        choices = ", ".join(sorted(PROVIDERS))
        raise ValueError(f"未知 LLM_PROVIDER={provider!r}，可选：{choices}")

    raw = PROVIDERS[provider]
    model = os.getenv("AGENT_MODEL", raw["model"]).strip()
    return LLMConfig(
        provider=provider,
        display_name=raw["display_name"],
        api_key_env=raw["api_key_env"],
        base_url=raw["base_url"],
        model=model,
    )


def create_client(config: LLMConfig | None = None) -> tuple[Any, LLMConfig]:
    config = config or get_llm_config()
    api_key = os.getenv(config.api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"使用 {config.display_name} 需要设置环境变量 {config.api_key_env}"
        )
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=config.base_url), config
