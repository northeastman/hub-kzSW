from __future__ import annotations

import json
from logging import Logger
from typing import Any

import redis

from knowledge_assistant.core.settings import RedisSettings


class RedisAnswerCache:
    def __init__(self, settings: RedisSettings, logger: Logger):
        self.logger = logger
        self.client = redis.StrictRedis(
            host=settings.host,
            port=settings.port,
            password=settings.password,
            db=settings.db,
            decode_responses=True,
        )
        self.client.ping()
        self.logger.info("Redis connected")

    def set_json(self, key: str, value: Any) -> None:
        self.client.set(key, json.dumps(value, ensure_ascii=False))

    def get_json(self, key: str) -> Any | None:
        value = self.client.get(key)
        return json.loads(value) if value else None

    def set_answer(self, query: str, answer: str) -> None:
        self.client.set(f"answer:{query}", answer)

    def get_answer(self, query: str) -> str | None:
        return self.client.get(f"answer:{query}")

