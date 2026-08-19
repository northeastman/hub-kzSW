# agents/sub_agent.py
from .base_agent import BaseAgent
from layers.memory import Memory
from typing import List, Optional

class SubAgent(BaseAgent):
    """子Agent，轻量级ReAct循环，不包含create_subagent等高级工具"""
    def __init__(
        self,
        memory: Memory,
        allowed_tools: List[str] = None,
        max_steps: int = 5,
        depth: int = 0,
        name: str = "SubAgent"
    ):
        system_prompt = (
            "你是一个子任务执行助手。你的目标是根据用户描述完成一个具体子任务，"
            "并返回最终结果。你可以使用提供的工具来获取信息或进行计算。"
            "请尽可能简洁、准确地完成任务。"
        )
        # 子Agent默认允许使用基础工具（search, calculator），但不包括create_subagent等
        if allowed_tools is None:
            allowed_tools = ["search", "calculator"]
        super().__init__(
            memory=memory,
            system_prompt=system_prompt,
            allowed_tools=allowed_tools,
            max_steps=max_steps,
            depth=depth,
            name=name
        )