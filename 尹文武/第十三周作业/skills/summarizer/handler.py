from __future__ import annotations

from typing import Any


async def run(
    user_input: str, arguments: dict[str, Any], context: Any
) -> dict[str, Any]:
    max_chars = min(max(int(arguments.get("max_chars", 240)), 40), 2000)
    if context.ai.client:
        memory_context = "\n".join(hit.text for hit in context.memories[:2])
        content = await context.ai.chat(
            [
                {
                    "role": "system",
                    "content": context.instructions
                    + "\n风格要求："
                    + context.resources.get("style.txt", ""),
                },
                {
                    "role": "user",
                    "content": f"可参考的历史结果：\n{memory_context}\n\n待总结文本：\n{user_input}",
                },
            ]
        )
        backend = "openai"
    else:
        compact = " ".join(user_input.split())
        content = compact if len(compact) <= max_chars else compact[:max_chars].rstrip() + "…"
        backend = "offline-fallback"
    return {"summary": content, "backend": backend}

