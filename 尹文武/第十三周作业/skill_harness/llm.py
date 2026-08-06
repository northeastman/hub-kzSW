from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np
from openai import AsyncOpenAI

from .config import Settings


class AIService:
    """OpenAI-compatible chat and embedding client with an offline embedding fallback."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = (
            AsyncOpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )
            if settings.openai_api_key
            else None
        )

    @property
    def embedding_backend(self) -> str:
        return "openai" if self.client else "local-hash"

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.settings.embedding_dimension), dtype=np.float32)
        if self.client:
            response = await self.client.embeddings.create(
                model=self.settings.embedding_model,
                input=texts,
            )
            vectors = np.asarray([item.embedding for item in response.data], dtype=np.float32)
            if vectors.shape[1] != self.settings.embedding_dimension:
                raise ValueError(
                    "Embedding dimension mismatch: API returned "
                    f"{vectors.shape[1]}, configured {self.settings.embedding_dimension}. "
                    "Set EMBEDDING_DIMENSION to the model's actual dimension."
                )
        else:
            vectors = np.vstack([self._hash_embedding(text) for text in texts])
        return self._normalize(vectors)

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        if not self.client:
            raise RuntimeError("Chat requires OPENAI_API_KEY (and optionally OPENAI_BASE_URL)")
        kwargs: dict[str, Any] = {
            "model": self.settings.chat_model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format
        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def _hash_embedding(self, text: str) -> np.ndarray:
        """Deterministic feature hashing keeps development/tests usable without a key."""
        vector = np.zeros(self.settings.embedding_dimension, dtype=np.float32)
        lowered = text.lower()
        tokens = re.findall(r"[a-z0-9_]+", lowered)
        chinese = re.findall(r"[\u4e00-\u9fff]", lowered)
        tokens.extend(chinese)
        tokens.extend(a + b for a, b in zip(chinese, chinese[1:]))
        for token in tokens or [text]:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "little") % vector.size
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign
        return vector

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return (vectors / norms).astype(np.float32)
