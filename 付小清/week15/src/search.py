"""联网搜索：Tavily API + Mock 离线模式。"""
import json
import logging
import os
import time
import urllib.request

from llm_client import is_mock_mode

logger = logging.getLogger(__name__)
TAVILY_URL = "https://api.tavily.com/search"


def tavily_search(query: str, max_results: int = 5) -> dict:
    if is_mock_mode():
        time.sleep(1.2)
        return {
            "answer": f"【Mock】关于「{query}」的摘要：市场持续扩张，增速约 15%。",
            "results": [
                {
                    "title": f"Mock 来源 - {query[:20]}",
                    "url": "https://example.com/mock",
                    "content": f"模拟搜索结果：{query} 相关数据与趋势分析。",
                }
            ],
            "response_time": 1.2,
        }

    key = os.getenv("TAVILY_API_KEY")
    if not key:
        logger.warning("未设置 TAVILY_API_KEY，使用 Mock 搜索")
        time.sleep(0.8)
        return {
            "answer": f"【Mock】关于「{query}」的摘要：市场持续扩张，增速约 15%。",
            "results": [
                {
                    "title": f"Mock 来源 - {query[:20]}",
                    "url": "https://example.com/mock",
                    "content": f"模拟搜索结果：{query} 相关数据与趋势分析。",
                }
            ],
            "response_time": 0.8,
        }
    payload = {
        "api_key": key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": True,
    }
    try:
        req = urllib.request.Request(
            TAVILY_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": (r.get("content") or "")[:600],
            }
            for r in data.get("results", [])
        ]
        return {
            "answer": data.get("answer") or "",
            "results": results,
            "response_time": data.get("response_time"),
        }
    except Exception as e:
        logger.warning("Tavily 搜索失败 '%s': %s", query, e)
        return {"error": f"{type(e).__name__}: {str(e)[:100]}"}


def format_search_result(result: dict) -> str:
    if "error" in result:
        return f"搜索失败: {result['error']}"
    parts = []
    if result.get("answer"):
        parts.append(f"摘要: {result['answer']}")
    for i, res in enumerate(result.get("results", []), 1):
        parts.append(f"[{i}] {res['title']}\n    {res['content'][:300]}")
    return "\n".join(parts) if parts else "无结果"
