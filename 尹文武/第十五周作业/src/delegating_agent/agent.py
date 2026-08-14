from __future__ import annotations

import json
import sys
from typing import Any, Iterable

from tqdm import tqdm

from .tools import Tool


DEFAULT_SYSTEM_PROMPT = """你是主 Agent，请准确解决用户的请求。
你可以联网搜索，也可以把相互独立、范围明确的工作委派给上下文隔离的子 Agent。
仅在确有帮助时进行委派。子 Agent 完全不了解当前对话，因此必须在目标和上下文中提供
它完成任务所需的全部信息。请综合子 Agent 的结果，不要只做简单转发。使用联网搜索后，
必须在最终回答中引用来源 URL，绝不能编造来源。"""


class Agent:
    """基于 OpenAI 兼容客户端的轻量异步工具调用 Agent。"""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        tools: Iterable[Tool] = (),
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_iterations: int = 20,
        name: str = "主 Agent",
        show_progress: bool = True,
        leave_progress: bool = True,
    ) -> None:
        self.client = client
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.name = name
        self.show_progress = show_progress
        self.leave_progress = leave_progress

    async def run(self, prompt: str) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        definitions = [tool.definition for tool in self.tools.values()]

        # 重定向输出或运行测试时自动关闭进度条，避免污染结构化输出。
        progress = tqdm(
            total=self.max_iterations,
            desc=f"{self.name}：准备中",
            unit="轮",
            dynamic_ncols=True,
            leave=self.leave_progress,
            disable=not self.show_progress or not sys.stderr.isatty(),
        )
        completed = False
        try:
            for _ in range(self.max_iterations):
                progress.set_description_str(f"{self.name}：正在请求模型")
                request: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                }
                if definitions:
                    request.update({"tools": definitions, "tool_choice": "auto"})
                response = await self.client.chat.completions.create(**request)
                progress.update(1)
                message = response.choices[0].message
                assistant_message = message.model_dump(exclude_none=True)
                messages.append(assistant_message)

                tool_calls = message.tool_calls or []
                if not tool_calls:
                    completed = True
                    progress.set_description_str(f"{self.name}：已完成")
                    progress.total = progress.n
                    progress.refresh()
                    return message.content or ""

                for call in tool_calls:
                    progress.set_description_str(
                        f"{self.name}：正在执行 {call.function.name}"
                    )
                    tool = self.tools.get(call.function.name)
                    try:
                        arguments = json.loads(call.function.arguments or "{}")
                        if not isinstance(arguments, dict):
                            raise ValueError("工具参数必须是 JSON 对象")
                        if tool is None:
                            result = json.dumps(
                                {"error": f"未知工具：{call.function.name}"},
                                ensure_ascii=False,
                            )
                        else:
                            result = await tool.execute(**arguments)
                    except Exception as exc:
                        result = json.dumps({"error": str(exc)}, ensure_ascii=False)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result,
                        }
                    )
        finally:
            if not completed:
                progress.set_description_str(f"{self.name}：已停止")
            progress.close()

        raise RuntimeError(
            f"Agent 达到 {self.max_iterations} 轮迭代上限后停止"
        )
