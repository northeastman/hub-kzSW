"""
Stage 1 — Skill 匹配（仅使用索引元数据）

在尚未加载 SKILL.md 正文的情况下，根据用户 query 选择最相关的 skill。
支持规则匹配（离线 demo）与 LLM 匹配（完整 harness）。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from skill_registry import SkillMeta

# 规则：关键词 → skill name
RULES: list[tuple[str, list[str]]] = [
    (
        "flash-card",
        [
            r"闪卡",
            r"flash\s*card",
            r"flashcard",
            r"单词卡",
            r"flash-card",
        ],
    ),
    (
        "baoyu-diagram",
        [
            r"画个?图",
            r"画一?个.*图",
            r"diagram",
            r"flowchart",
            r"架构图",
            r"流程图",
            r"时序图",
            r"结构图",
            r"思维导图",
            r"svg",
            r"可视化",
        ],
    ),
]


@dataclass
class MatchResult:
    skill_name: str
    confidence: float
    method: str  # "rule" | "llm" | "none"
    reason: str = ""


def _rules_for_skill(name: str) -> list[str]:
    for skill_name, patterns in RULES:
        if skill_name == name:
            return patterns
    return []


def match_by_rules(query: str, skills: list[SkillMeta]) -> MatchResult | None:
    q = query.strip()
    scores: list[tuple[str, float, str]] = []

    for meta in skills:
        patterns = _rules_for_skill(meta.name)
        hits = [p for p in patterns if re.search(p, q, re.I)]
        if hits:
            scores.append((meta.name, min(0.5 + 0.1 * len(hits), 0.95), f"规则命中: {hits[0]}"))

        # description 子串（弱匹配）
        desc_snippets = re.findall(r"[\u4e00-\u9fff]{2,}|flash|diagram|card", meta.description, re.I)
        for snip in desc_snippets[:5]:
            if len(snip) >= 2 and snip.lower() in q.lower():
                scores.append((meta.name, 0.55, f"描述关键词: {snip}"))

    if not scores:
        return None

    scores.sort(key=lambda x: x[1], reverse=True)
    best_name, conf, reason = scores[0]
    return MatchResult(skill_name=best_name, confidence=conf, method="rule", reason=reason)


def match_by_llm(query: str, skills: list[SkillMeta]) -> MatchResult | None:
    """把索引元数据发给 LLM，让其选择 skill（模拟 Agent 读 available_skills 后决策）。"""
    try:
        from openai import OpenAI
    except ImportError:
        return None

    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    base_url = os.environ.get(
        "AGENT_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    model = os.environ.get("AGENT_MODEL", "qwen-plus")

    index_json = [
        {"name": s.name, "description": s.description[:500]}
        for s in skills
    ]

    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt = f"""你是 skill 路由器。根据用户请求，从下列 skill 索引（仅 name+description，无正文）中选择最合适的一个。

Skills 索引:
{json.dumps(index_json, ensure_ascii=False, indent=2)}

用户请求: {query}

若无合适 skill，返回 {{"skill": null, "confidence": 0, "reason": "..."}}。
否则返回 JSON: {{"skill": "<name>", "confidence": 0.0-1.0, "reason": "..."}}
只输出 JSON。"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = resp.choices[0].message.content or ""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    data = json.loads(m.group())
    name = data.get("skill")
    if not name:
        return MatchResult(skill_name="", confidence=0, method="llm", reason=data.get("reason", "无匹配"))
    return MatchResult(
        skill_name=name,
        confidence=float(data.get("confidence", 0.8)),
        method="llm",
        reason=data.get("reason", ""),
    )


def match_skill(
    query: str,
    skills: list[SkillMeta],
    *,
    prefer_llm: bool = False,
) -> MatchResult | None:
    if prefer_llm:
        llm = match_by_llm(query, skills)
        if llm and llm.skill_name:
            return llm

    rule = match_by_rules(query, skills)
    if rule:
        return rule

    if not prefer_llm:
        return match_by_llm(query, skills)
    return None


def get_skill_by_name(skills: list[SkillMeta], name: str) -> SkillMeta | None:
    for s in skills:
        if s.name == name:
            return s
    return None
