"""LLM 客户端：支持真实 API 与 Mock 离线模式。"""
import os
import time
import logging
import re

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
_client = None


def is_mock_mode() -> bool:
    return os.getenv("MOCK_MODE", "").lower() in ("1", "true", "yes")


def get_api_key() -> str:
    return os.getenv("DASHSCOPE_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or ""


def get_base_url() -> str:
    return os.getenv("AGENT_BASE_URL", DEFAULT_BASE_URL)


def get_model() -> str:
    return os.getenv("AGENT_MODEL", DEFAULT_MODEL)


def get_client():
    global _client
    if _client is None:
        key = get_api_key()
        if not key:
            raise EnvironmentError(
                "请设置 DASHSCOPE_API_KEY 或 DEEPSEEK_API_KEY，或 MOCK_MODE=1 使用离线演示"
            )
        from openai import OpenAI

        _client = OpenAI(api_key=key, base_url=get_base_url())
    return _client


def _mock_llm(system: str, user: str) -> str:
    """离线 Mock：根据 agent 角色与对话轮次返回固定 ReAct 步骤。"""
    is_main = "主分析师" in system or "dispatch_subagents" in system
    has_observation = "Observation:" in user
    question_match = re.search(r"Question:\s*(.+?)(?:\n\n|$)", user, re.S)
    question = (question_match.group(1).strip() if question_match else "调研课题")

    if is_main:
        if not has_observation:
            parts = [p.strip() for p in re.split(r"[：:，,、]", question) if p.strip()]
            if len(parts) >= 2:
                subtopics = parts[-3:] if len(parts) >= 3 else parts[-2:]
            else:
                subtopics = [f"{question} 市场规模", f"{question} 竞争格局", f"{question} 趋势分析"]
            sub_input = " | ".join(subtopics[:3])
            return (
                f"Thought: 这是多维度调研问题，需要派发子 agent 并行收集各侧面信息\n"
                f"Action: dispatch_subagents\n"
                f"Action Input: {sub_input}"
            )
        return (
            "Thought: 已收齐各子 agent 的并行调研结果，综合成结构化报告\n"
            "Final Answer: 【综合调研报告】\n"
            "1. 市场规模：子 agent 已收集相关数据与来源\n"
            "2. 竞争格局：主要参与者与份额信息已汇总\n"
            "3. 趋势分析：政策与消费趋势要点已整合\n"
            "结论：多侧面并行调研完成，报告分维度组织并标注来源。"
        )

    if not has_observation:
        search_q = question[:60]
        return (
            f"Thought: 需要联网搜索「{search_q}」的相关资料\n"
            f"Action: web_search\n"
            f"Action Input: {search_q}"
        )
    return (
        f"Thought: 搜索完成，可以给出该子课题结论\n"
        f"Final Answer: 关于「{question[:40]}」的调研要点："
        f"市场规模持续增长，头部品牌集中度提升，政策方向明确。（Mock 数据）"
    )


def llm_chat(system, user, *, temperature=0.0, max_tokens=1024, stop=None, retries=3):
    if is_mock_mode():
        return _mock_llm(system, user)

    for attempt in range(retries):
        try:
            resp = get_client().chat.completions.create(
                model=get_model(),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
            logger.warning("LLM 重试(%s): %s", attempt + 1, str(e)[:80])
