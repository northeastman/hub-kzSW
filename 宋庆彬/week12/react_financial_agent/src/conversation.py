"""多轮对话的历史消息和内存会话存储。"""

from __future__ import annotations

from typing import Iterable, Mapping


MAX_MESSAGE_CHARS = 20_000


def normalize_history(history: Iterable[Mapping[str, object]] | None) -> list[dict[str, str]]:
    """只保留可以安全放回 Chat Completions 的 user/assistant 消息。"""
    normalized: list[dict[str, str]] = []
    for message in history or []:
        role = message.get("role")
        content = message.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        content = content.strip()
        if content:
            normalized.append({"role": role, "content": content[:MAX_MESSAGE_CHARS]})
    return normalized


class ConversationStore:
    """进程内会话存储。"""

    def __init__(self):
        self._items: dict[str, list[dict[str, str]]] = {}

    def get(self, conversation_id: str) -> list[dict[str, str]]:
        return [message.copy() for message in self._items.get(conversation_id, [])]

    def append_turn(self, conversation_id: str, question: str, answer: str) -> None:
        self._items.setdefault(conversation_id, []).extend(
            [
                {"role": "user", "content": question.strip()[:MAX_MESSAGE_CHARS]},
                {"role": "assistant", "content": answer.strip()[:MAX_MESSAGE_CHARS]},
            ]
        )

    def clear(self, conversation_id: str) -> bool:
        return self._items.pop(conversation_id, None) is not None
