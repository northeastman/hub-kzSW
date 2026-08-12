from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from .agent import Agent
from .config import Settings
from .delegation import DelegateTaskTool
from .tools import WebSearchTool


CHILD_SYSTEM_PROMPT = """你是上下文隔离的工作 Agent，负责完成一项委派任务。
除收到的目标与上下文外，你无法访问父 Agent 的任何对话。请独立工作；需要最新信息或需要
外部核实时，应调用 web_search，并引用结果 URL。你不能继续委派任务，也不能向用户提问。
只需向父 Agent 返回有用且自包含的最终总结。"""


def build_main_agent(settings: Settings, *, client: Any | None = None) -> Agent:
    shared_client = client or AsyncOpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
    )
    web_search = WebSearchTool()

    def child_factory(goal: str, context: str, max_iterations: int) -> Agent:
        # 新建 Agent 即可获得全新的消息历史；它只继承安全的联网搜索能力。
        return Agent(
            client=shared_client,
            model=settings.model,
            tools=[web_search],
            system_prompt=CHILD_SYSTEM_PROMPT,
            max_iterations=max_iterations,
            name=f"子 Agent（{goal[:16]}）",
            leave_progress=False,
        )

    delegation = DelegateTaskTool(
        child_factory,
        max_concurrent_children=settings.max_concurrent_children,
        default_max_iterations=settings.child_max_iterations,
    )
    return Agent(
        client=shared_client,
        model=settings.model,
        tools=[web_search, delegation],
        max_iterations=settings.agent_max_iterations,
        name="主 Agent",
    )
