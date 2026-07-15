from __future__ import annotations

from collections.abc import Iterator
from logging import Logger

from openai import OpenAI

from knowledge_assistant.core.settings import LlmSettings


class DashScopeChatClient:
    """OpenAI-compatible DashScope chat client with sync and streaming methods."""

    def __init__(self, settings: LlmSettings, logger: Logger):
        if not settings.api_key:
            raise ValueError("LLM API key is empty. Set KA_LLM_API_KEY or llm.api_key.")
        self.settings = settings
        self.logger = logger
        self.client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)

    def generate_text(self, prompt: str, system_prompt: str = "你是一个有用的助手。") -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.settings.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                timeout=self.settings.timeout_seconds,
                temperature=0.1,
            )
            if completion.choices and completion.choices[0].message:
                return completion.choices[0].message.content or ""
            return ""
        except Exception as exc:
            self.logger.error(f"LLM text generation failed: {exc}")
            raise

    def stream_text(self, prompt: str, system_prompt: str = "你是一个有用的助手。") -> Iterator[str]:
        try:
            completion = self.client.chat.completions.create(
                model=self.settings.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                timeout=self.settings.timeout_seconds,
                stream=True,
            )
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as exc:
            self.logger.error(f"LLM stream generation failed: {exc}")
            raise

