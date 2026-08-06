"""
intent_matcher.py — LLM 驱动的 Skill 意图匹配器

工作流程：
  1. 确保 registry 已加载所有 skill 的 Level 1 元数据
  2. 把每个 skill 的 name + description 拼成候选清单
  3. 调 LLM 输出 JSON：{matched: [{name, score, reason}]}
  4. 按 score 降序返回 Top-K

为什么不直接用关键词匹配？
  Skill 的 description 是自然语言（"画个图""diagram""闪卡"），用户输入更口语化
  （"给我整一张 crazy 单词卡"），关键词匹配会漏。LLM 能跨语义对齐。
"""

from __future__ import annotations
import json
import re
import logging
from typing import Callable, Awaitable

from .skill_registry import SkillRegistry, SkillEntry
from .llm_config import get_chat_client

logger = logging.getLogger(__name__)

# 用于广播事件的回调类型（供 serve.py 注入 SSE 推送）
Broadcaster = Callable[[str, dict], Awaitable[None]]


_MATCH_SYSTEM_PROMPT = """你是一个 Skill 路由器。给定用户输入和一批候选 Skill（含 name 和 description），
判断哪些 Skill 适合处理该输入，并给出 0~1 的置信度和一句话理由。

输出严格的 JSON（不要代码块包裹，不要解释）：
{
  "matched": [
    {"name": "<skill name>", "score": 0.95, "reason": "<一句话理由>"}
  ]
}

规则：
- score >= 0.6 才算命中
- 没有任何 skill 匹配时返回 {"matched": []}
- 只能从候选清单里选，不要编造
- 按 score 从高到低排序"""


class IntentMatcher:
    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    async def match(
        self,
        user_input: str,
        top_k: int = 3,
        broadcaster: Broadcaster | None = None,
    ) -> list[dict]:
        """
        返回 [{"name", "score", "reason", "display_name", "description"}]，
        按 score 降序，最多 top_k 条。
        """
        # Step 1: 确保所有 skill 至少加载到 Level 1
        if broadcaster:
            await broadcaster("intent_start", {"user_input": user_input})
        candidates = self.registry.load_all_metadata()
        if not candidates:
            if broadcaster:
                await broadcaster("intent_empty", {})
            return []

        # Step 2: 拼候选清单
        candidate_lines = []
        for e in candidates:
            candidate_lines.append(
                f"- name: {e.name}\n  description: {e.description}"
            )
        candidate_block = "\n".join(candidate_lines)
        if broadcaster:
            await broadcaster(
                "intent_candidates",
                {"count": len(candidates), "names": [e.name for e in candidates]},
            )

        # Step 3: 调 LLM
        user_prompt = (
            f"用户输入：\n{user_input}\n\n"
            f"候选 Skill：\n{candidate_block}\n\n"
            f"请输出 JSON。"
        )
        try:
            client, model = get_chat_client()
            # 同步 SDK 包到线程池里跑，避免阻塞 event loop
            import asyncio
            resp_text = await asyncio.to_thread(self._call_llm, client, model, user_prompt)
        except Exception as e:
            logger.error(f"[Intent] LLM 调用失败：{e}")
            if broadcaster:
                await broadcaster("intent_error", {"error": str(e)})
            return []

        # Step 4: 解析 JSON
        parsed = self._extract_json(resp_text)
        if not parsed:
            logger.warning(f"[Intent] JSON 解析失败，原文：{resp_text[:200]}")
            if broadcaster:
                await broadcaster("intent_parse_fail", {"raw": resp_text[:200]})
            return []

        matched = parsed.get("matched", [])
        # 补全 display_name / description，并过滤低分
        result = []
        for item in matched:
            name = item.get("name", "")
            entry = self.registry.get(name)
            if entry is None:
                continue
            score = float(item.get("score", 0))
            if score < 0.6:
                continue
            result.append({
                "name":         name,
                "score":        score,
                "reason":       item.get("reason", ""),
                "display_name": entry.display_name,
                "description":  entry.description,
            })
        result.sort(key=lambda x: x["score"], reverse=True)
        result = result[:top_k]

        if broadcaster:
            await broadcaster("intent_done", {"matched": result})
        return result

    def _call_llm(self, client, model: str, user_prompt: str) -> str:
        """同步 LLM 调用（由 asyncio.to_thread 包裹执行）。"""
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _MATCH_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        return resp.choices[0].message.content or ""

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """从可能带杂质的 LLM 输出中提取第一个 JSON 对象。"""
        # 先去掉 ```json ... ``` 代码块包裹
        text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
        text = re.sub(r"\n?```\s*$", "", text)
        # 抓第一个 {...}
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
