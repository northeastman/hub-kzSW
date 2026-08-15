"""Tavily 联网搜索封装（零额外依赖，用标准库 urllib）

历史人物文章需要联网素材（生平、轶事、争议评价），Tavily 是为 LLM 优化的搜索 API。
用 urllib 而非 requests，避免引入新依赖。

使用方式：
  from tavily_search import tavily_search, format_search_result
  r = tavily_search("苏轼 乌台诗案")
  # r = {"answer": "...", "results": [{"title","url","content"}], "response_time": ...}

依赖：环境变量 TAVILY_API_KEY
"""
import os, json, urllib.request, logging

logger = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"


# 进程内查询缓存：同一 query 不重复打 Tavily（重跑/同人物省 token、省时间）
_SEARCH_CACHE: dict[str, dict] = {}


def tavily_search(query: str, max_results: int = 5) -> dict:
    """调用 Tavily 搜索。返回 {answer, results, response_time}。
    失败返回 {"error": ...}，不抛异常（ReAct loop 兜底）。"""
    cache_key = f"{query}::{max_results}"
    if cache_key in _SEARCH_CACHE:
        return _SEARCH_CACHE[cache_key]
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return {"error": "未设置 TAVILY_API_KEY"}
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
        results = [{"title": r.get("title", ""), "url": r.get("url", ""),
                     "content": (r.get("content") or "")[:600]}
                   for r in data.get("results", [])]
        out = {"answer": data.get("answer") or "",
               "results": results,
               "response_time": data.get("response_time")}
        _SEARCH_CACHE[cache_key] = out  # 仅缓存成功结果
        return out
    except Exception as e:
        logger.warning(f"Tavily 搜索失败 '{query}': {e}")
        return {"error": f"{type(e).__name__}: {str(e)[:100]}"}


def format_search_result(r: dict) -> str:
    """把 Tavily 返回格式化成喂给 LLM 的文本。"""
    if "error" in r:
        return f"搜索失败: {r['error']}"
    parts = []
    if r.get("answer"):
        parts.append(f"摘要: {r['answer']}")
    for i, res in enumerate(r.get("results", []), 1):
        parts.append(f"[{i}] {res['title']}\n    URL: {res['url']}\n    {res['content'][:300]}")
    return "\n".join(parts) if parts else "无结果"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    r = tavily_search("苏轼 乌台诗案 经过")
    print(format_search_result(r)[:400])
