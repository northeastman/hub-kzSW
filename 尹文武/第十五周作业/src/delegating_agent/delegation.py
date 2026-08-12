from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any, Protocol

from .tools import Tool


class RunnableAgent(Protocol):
    async def run(self, prompt: str) -> str: ...


ChildFactory = Callable[[str, str, int], RunnableAgent]


class DelegateTaskTool(Tool):
    """将上下文全新且相互隔离的子 Agent 封装为模型可调用的工具。"""

    name = "delegate_task"
    description = """把单个任务或一批并行任务委派给上下文隔离的子 Agent。
单任务传入 goal/context，批任务传入 tasks，两种形式不能同时使用。每个子 Agent 只能看到
传给它的目标与上下文，可以联网搜索，但不能继续委派。批任务结果保持输入顺序。请确保各项
任务相互独立、信息完备，并提供完整上下文。"""
    parameters = {
        "type": "object",
        "properties": {
            "goal": {"type": "string", "description": "单个子 Agent 要达成的明确目标"},
            "context": {
                "type": "string",
                "description": "隔离的子 Agent 所需的全部背景信息与约束",
                "default": "",
            },
            "tasks": {
                "type": "array",
                "description": "需要并行执行且相互独立的任务列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string"},
                        "context": {"type": "string", "default": ""},
                    },
                    "required": ["goal"],
                    "additionalProperties": False,
                },
                "minItems": 1,
            },
            "max_iterations": {
                "type": "integer",
                "description": "每个子 Agent 的最大迭代轮数",
                "minimum": 1,
            },
        },
        "additionalProperties": False,
    }

    def __init__(
        self,
        child_factory: ChildFactory,
        *,
        max_concurrent_children: int = 3,
        default_max_iterations: int = 12,
    ) -> None:
        self.child_factory = child_factory
        self.max_concurrent_children = max_concurrent_children
        self.default_max_iterations = default_max_iterations

    async def execute(self, **arguments: Any) -> str:
        goal = arguments.get("goal")
        tasks = arguments.get("tasks")
        if bool(goal) == bool(tasks):
            return json.dumps(
                {"error": "必须且只能提供 goal 或 tasks 其中之一"}, ensure_ascii=False
            )

        raw_tasks = tasks or [{"goal": goal, "context": arguments.get("context", "")}]
        if len(raw_tasks) > self.max_concurrent_children:
            return json.dumps(
                {
                    "error": (
                        f"批任务包含 {len(raw_tasks)} 项，最多允许 "
                        f"{self.max_concurrent_children} 项"
                    )
                },
                ensure_ascii=False,
            )
        iterations = int(arguments.get("max_iterations", self.default_max_iterations))
        if iterations < 1:
            return json.dumps(
                {"error": "max_iterations 必须大于或等于 1"},
                ensure_ascii=False,
            )

        async def run_one(index: int, item: dict[str, Any]) -> dict[str, Any]:
            child_goal = str(item.get("goal", "")).strip()
            context = str(item.get("context", "")).strip()
            if not child_goal:
                return {"index": index, "status": "error", "error": "目标不能为空"}
            try:
                child = self.child_factory(child_goal, context, iterations)
                summary = await child.run(_child_prompt(child_goal, context))
                return {
                    "index": index,
                    "goal": child_goal,
                    "status": "completed",
                    "summary": summary,
                }
            except Exception as exc:
                return {
                    "index": index,
                    "goal": child_goal,
                    "status": "error",
                    "error": str(exc),
                }

        results = await asyncio.gather(
            *(run_one(index, item) for index, item in enumerate(raw_tasks))
        )
        return json.dumps({"results": results}, ensure_ascii=False)


def _child_prompt(goal: str, context: str) -> str:
    return f"""请完成以下委派任务。

目标：
{goal}

上下文：
{context or '（未提供额外上下文。）'}

请返回简洁且自包含的总结，其中应包括：任务结果、支持证据、使用联网搜索时的来源 URL、
重要的不确定因素以及所有未完成工作。不要向用户提问；请自行作出合理假设并明确说明。"""
