from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    async def execute(self, **arguments: Any) -> str:
        raise NotImplementedError


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "在公共互联网中搜索最新信息，返回标题、URL 和摘要。"
        "遇到可能随时间变化的事实时使用此工具，并在回答中引用结果 URL。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "明确、聚焦的搜索关键词"},
            "max_results": {
                "type": "integer",
                "description": "返回结果数量，取值范围为 1 到 10",
                "minimum": 1,
                "maximum": 10,
                "default": 5,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    async def execute(self, **arguments: Any) -> str:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return json.dumps({"error": "搜索关键词不能为空"}, ensure_ascii=False)
        limit = max(1, min(int(arguments.get("max_results", 5)), 10))

        def search() -> list[dict[str, str]]:
            from ddgs import DDGS

            rows = DDGS().text(query, max_results=limit)
            return [
                {
                    "title": str(row.get("title", "")),
                    "url": str(row.get("href", row.get("url", ""))),
                    "snippet": str(row.get("body", row.get("snippet", ""))),
                }
                for row in rows
            ]

        try:
            results = await asyncio.to_thread(search)
            return json.dumps(
                {"query": query, "results": results}, ensure_ascii=False
            )
        except Exception as exc:  # 将搜索服务或网络错误返回给模型处理
            return json.dumps(
                {"query": query, "error": f"联网搜索失败：{exc}"},
                ensure_ascii=False,
            )
