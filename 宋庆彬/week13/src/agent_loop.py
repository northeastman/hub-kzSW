"""最小 Function Calling Agent Loop。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.skill_registry import SkillRegistry
from src.tool_registry import ToolRegistry, TurnState


BASE_SYSTEM_PROMPT = """你是一个支持渐进式 Skills 的助手。

工作规则：
1. 下方只提供 Skill 摘要索引，并没有加载完整 Skill。
2. 当用户请求明显匹配某个 Skill 时，必须先调用 load_skill，读取完整指令后再执行。
3. 不匹配任何 Skill 的普通问题可以直接回答，不要为了调用工具而调用工具。
4. Skill 加载后，严格遵循其流程、约束和输出要求。
5. 只能调用当前 tools 列表中出现的工具，不能声称执行了没有实际调用的工具。
6. Skill、资源和工具轨迹只在当前用户轮次有效，下一轮需要重新判断。
7. 工具失败时先阅读错误信息；能够修正则重试，否则如实向用户说明。

## 可用 Skill 摘要索引
{catalog}
"""


@dataclass
class TurnMetrics:
    catalog_chars: int
    loaded_skills: list[str] = field(default_factory=list)
    loaded_resources: list[str] = field(default_factory=list)
    written_files: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    steps: int = 0
    finish_reason: str = ""


@dataclass
class TurnResult:
    answer: str
    metrics: TurnMetrics


class AgentLoop:
    """每个 run_turn 都创建独立的临时 Context，以便轮次结束后释放 Skill。"""

    def __init__(
        self,
        client: Any,
        model: str,
        skills: SkillRegistry,
        workspace: Path,
        *,
        max_steps: int = 8,
        temperature: float = 0.0,
        max_history_messages: int = 20,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps 必须大于 0")
        self.client = client
        self.model = model
        self.skills = skills
        self.workspace = workspace.resolve()
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_history_messages = max_history_messages
        self.tools = ToolRegistry(skills, self.workspace)

    def run_turn(
        self,
        user_input: str,
        history: list[dict[str, str]] | None = None,
    ) -> TurnResult:
        user_input = user_input.strip()
        if not user_input:
            raise ValueError("user_input 不能为空")

        catalog = self.skills.catalog()
        system_prompt = BASE_SYSTEM_PROMPT.format(catalog=catalog)
        state = TurnState(workspace=self.workspace)

        # history 只包含最终问答；Skill 正文和工具轨迹只进入本地 turn_messages。
        durable_history = self._sanitize_history(history or [])
        turn_messages: list[Any] = [
            {"role": "system", "content": system_prompt},
            *durable_history,
            {"role": "user", "content": user_input},
        ]

        metrics = TurnMetrics(catalog_chars=len(catalog))

        for step in range(1, self.max_steps + 1):
            metrics.steps = step
            response = self.client.chat.completions.create(
                model=self.model,
                messages=turn_messages,
                tools=self.tools.schemas_for(state),
                tool_choice="auto",
                temperature=self.temperature,
            )
            choice = response.choices[0]
            message = choice.message
            tool_calls = list(getattr(message, "tool_calls", None) or [])

            if not tool_calls:
                metrics.finish_reason = getattr(choice, "finish_reason", "") or "stop"
                self._copy_state_to_metrics(state, metrics)
                answer = getattr(message, "content", None) or "（模型返回空内容）"
                return TurnResult(answer=answer, metrics=metrics)

            turn_messages.append(self._assistant_message_dict(message, tool_calls))

            for tool_call in tool_calls:
                function = tool_call.function
                tool_name = function.name
                try:
                    arguments = json.loads(function.arguments or "{}")
                    if not isinstance(arguments, dict):
                        raise ValueError("工具参数必须是 JSON 对象")
                except (json.JSONDecodeError, ValueError) as exc:
                    observation = f"工具参数解析失败：{exc}"
                else:
                    observation = self.tools.execute(
                        tool_name,
                        arguments,
                        state,
                    )

                turn_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": observation,
                    }
                )

        metrics.finish_reason = "max_steps"
        self._copy_state_to_metrics(state, metrics)
        return TurnResult(
            answer=f"已达到最大执行步数 {self.max_steps}，任务未能正常结束。",
            metrics=metrics,
        )

    @staticmethod
    def _assistant_message_dict(message: Any, tool_calls: list[Any]) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": getattr(message, "content", None),
            "tool_calls": [
                {
                    "id": call.id,
                    "type": getattr(call, "type", "function"),
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in tool_calls
            ],
        }

    @staticmethod
    def _copy_state_to_metrics(
        state: TurnState,
        metrics: TurnMetrics,
    ) -> None:
        metrics.loaded_skills = list(state.loaded_skills)
        metrics.loaded_resources = list(state.loaded_resources)
        metrics.written_files = list(state.written_files)
        metrics.tool_calls = list(state.tool_calls)

    def _sanitize_history(
        self,
        history: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """历史只接受最终 user/assistant 文本，主动丢弃内部工具消息。"""
        durable: list[dict[str, str]] = []
        for message in history:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                continue
            durable.append({"role": role, "content": content})
        return durable[-self.max_history_messages :]
