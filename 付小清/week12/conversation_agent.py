"""
多轮对话 Agent 核心模块

教学重点：
  1. 单轮 Agent：每问独立，messages 每次重建 → 无法理解「那五粮液呢？」这类追问
  2. 多轮 Agent：跨轮保留 user/assistant 摘要历史 → 追问可引用上文
  3. ReAct 内部循环（Thought→Action→Observation）仍在一轮内完成；
     跨轮只持久化「用户问题 + Final Answer」，避免 token 爆炸

复用 ../react_financial_agent/src 中的 ReAct 实现与工具集。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Literal

# 复用教学项目
AGENT_ROOT = Path(__file__).parent.parent / "react_financial_agent" / "src"
sys.path.insert(0, str(AGENT_ROOT))

Mode = Literal["manual", "fc"]


@dataclass
class TurnResult:
    """一轮对话的完整结果"""
    question: str
    answer: str
    steps: list[dict] = field(default_factory=list)
    turn_index: int = 0


@dataclass
class ConversationSession:
    """
    多轮对话会话：跨轮保留精简历史，每轮内部仍走 ReAct 工具循环。

    history 格式（OpenAI messages 子集，仅 user/assistant）：
      [{"role": "user", "content": "茅台2023毛利率？"},
       {"role": "assistant", "content": "91.96%..."}, ...]
    """
    mode: Mode = "manual"
    max_steps: int = 10
    history: list[dict] = field(default_factory=list)

    def reset(self) -> None:
        self.history.clear()

    def _build_messages(self, question: str) -> list[dict]:
        """构造带历史上下文的 messages（供扩展/调试）"""
        if self.mode == "manual":
            from react_manual import SYSTEM_PROMPT
            system = SYSTEM_PROMPT
        else:
            from react_function_calling import FC_SYSTEM_PROMPT
            system = FC_SYSTEM_PROMPT + "\n\n你可以参考之前的对话记录理解用户的追问（如「那五粮液呢？」）。"

        return [
            {"role": "system", "content": system},
            *self.history,
            {"role": "user", "content": question},
        ]

    def ask(self, question: str) -> TurnResult:
        """
        处理用户一轮输入：ReAct 推理 → 将 (question, answer) 写入 history
        """
        steps: list[dict] = []
        answer = ""

        for step_data in self._run_react_with_history(question):
            steps.append(step_data)
            if step_data.get("type") == "final":
                answer = step_data.get("answer", "")
            elif step_data.get("type") in ("error", "max_steps"):
                answer = step_data.get("answer") or step_data.get("observation", "")

        turn_index = len(self.history) // 2 + 1
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        return TurnResult(
            question=question,
            answer=answer,
            steps=steps,
            turn_index=turn_index,
        )

    def ask_stream(self, question: str) -> Generator[dict, None, TurnResult]:
        """流式版本：逐步 yield ReAct 步骤，最后 yield turn_done"""
        steps: list[dict] = []
        answer = ""

        for step_data in self._run_react_with_history(question):
            steps.append(step_data)
            yield step_data
            if step_data.get("type") == "final":
                answer = step_data.get("answer", "")
            elif step_data.get("type") in ("error", "max_steps"):
                answer = step_data.get("answer") or step_data.get("observation", "")

        turn_index = len(self.history) // 2 + 1
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        result = TurnResult(
            question=question,
            answer=answer,
            steps=steps,
            turn_index=turn_index,
        )
        yield {"type": "turn_done", "turn_index": turn_index, "answer": answer}
        return result

    def _run_react_with_history(self, question: str) -> Generator[dict, None, None]:
        """
        带历史上下文的 ReAct 循环（从 react_manual / react_function_calling 抽取并扩展）
        """
        if self.mode == "manual":
            yield from self._react_manual_loop(question)
        else:
            yield from self._react_fc_loop(question)

    def _react_manual_loop(self, question: str) -> Generator[dict, None, None]:
        import json
        from react_manual import (
            SYSTEM_PROMPT,
            _parse_step,
            client,
            MODEL,
        )
        from tools import TOOLS_MAP

        multi_turn_hint = (
            "\n\n【多轮对话】用户可能基于上文追问（如「那五粮液呢？」「差多少？」），"
            "请结合历史对话理解意图，必要时重新调用工具获取新数据。"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + multi_turn_hint},
            *self.history,
            {"role": "user", "content": question},
        ]

        for step in range(1, self.max_steps + 1):
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0,
                stop=["Observation:"],
            )
            llm_output = response.choices[0].message.content.strip()
            parsed = _parse_step(llm_output)

            if parsed["type"] == "final":
                yield {
                    "step": step,
                    "type": "final",
                    "thought": parsed["thought"],
                    "answer": parsed["answer"],
                }
                return

            if parsed["type"] == "unparseable":
                yield {
                    "step": step,
                    "type": "error",
                    "observation": f"格式解析失败，原始输出：{llm_output[:200]}",
                }
                return

            tool_name = parsed["action"]
            tool_args = parsed["action_input"]
            tool_fn = TOOLS_MAP.get(tool_name)

            if tool_fn is None:
                observation = f"未知工具 '{tool_name}'，可用工具：{list(TOOLS_MAP.keys())}"
            else:
                try:
                    observation = tool_fn(**tool_args)
                except TypeError as e:
                    observation = f"工具参数错误: {e}"

            yield {
                "step": step,
                "type": "action",
                "thought": parsed["thought"],
                "action": tool_name,
                "action_input": tool_args,
                "observation": str(observation),
            }

            messages.append({"role": "assistant", "content": llm_output})
            messages.append({"role": "user", "content": f"Observation: {observation}\n"})

        yield {
            "step": self.max_steps + 1,
            "type": "max_steps",
            "answer": f"已达最大步数 {self.max_steps}，未能得出最终答案",
        }

    def _react_fc_loop(self, question: str) -> Generator[dict, None, None]:
        import json
        from react_function_calling import FC_SYSTEM_PROMPT, client, MODEL
        from tools import TOOLS_MAP, TOOLS_SCHEMA

        system = (
            FC_SYSTEM_PROMPT
            + "\n\n【多轮对话】用户可能基于上文追问，请结合历史对话理解意图。"
        )
        messages = [
            {"role": "system", "content": system},
            *self.history,
            {"role": "user", "content": question},
        ]

        for step in range(1, self.max_steps + 1):
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0,
            )
            msg = response.choices[0].message
            reason = response.choices[0].finish_reason

            if reason == "stop" or not msg.tool_calls:
                yield {
                    "step": step,
                    "type": "final",
                    "thought": "",
                    "answer": msg.content or "（模型返回空内容）",
                }
                return

            messages.append(msg)

            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                tool_fn = TOOLS_MAP.get(tool_name)
                if tool_fn is None:
                    observation = f"未知工具 '{tool_name}'"
                else:
                    try:
                        observation = tool_fn(**tool_args)
                    except TypeError as e:
                        observation = f"工具参数错误: {e}"

                yield {
                    "step": step,
                    "type": "action",
                    "thought": "",
                    "action": tool_name,
                    "action_input": tool_args,
                    "observation": str(observation),
                }

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(observation),
                })

        yield {
            "step": self.max_steps + 1,
            "type": "max_steps",
            "answer": f"已达最大步数 {self.max_steps}，未能得出最终答案",
        }


def run_single_turn(question: str, mode: Mode = "manual", max_steps: int = 10) -> TurnResult:
    """单轮模式：每问独立，history 始终为空（对照组）"""
    session = ConversationSession(mode=mode, max_steps=max_steps)
    steps: list[dict] = []
    answer = ""
    for step_data in session._run_react_with_history(question):
        steps.append(step_data)
        if step_data.get("type") == "final":
            answer = step_data.get("answer", "")
        elif step_data.get("type") in ("error", "max_steps"):
            answer = step_data.get("answer") or step_data.get("observation", "")
    return TurnResult(question=question, answer=answer, steps=steps, turn_index=1)
