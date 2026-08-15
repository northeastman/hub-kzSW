"""通用 ReAct 循环引擎：主 agent 与 subagent 共用。"""
import re
import time
import logging
from typing import Callable, Optional

from llm_client import llm_chat

logger = logging.getLogger(__name__)

REACT_SYSTEM = """你是调研助手，能用以下工具联网搜索。

可用工具：
{tools_desc}

按如下格式严格输出（每轮一次 Thought/Action/Action Input）：
Thought: 你的推理
Action: 工具名
Action Input: 工具参数（字符串）

工具执行后会得到 Observation。多轮调用直到能给出完整答案，最后用：
Thought: 我已收集足够信息
Final Answer: 综合答案

规则：
- Action 必须是上面列出的工具名之一
- 每轮只调一次工具，等 Observation 再决定下一步"""


def build_tools_desc(tools: dict) -> str:
    return "\n".join(f"- {name}: {desc}" for name, (_, desc) in tools.items())


class ReActLoop:
    def __init__(
        self,
        agent_name: str,
        tools: dict,
        max_steps: int = 6,
        model_tag: str = "deepseek-chat",
        system_prompt: Optional[str] = None,
    ):
        self.agent_name = agent_name
        self.tools = tools
        self.max_steps = max_steps
        self.model_tag = model_tag
        self._system_template = system_prompt or REACT_SYSTEM
        self.trace: list[dict] = []

    def run(
        self,
        question: str,
        on_step: Callable = None,
        shared_state: dict = None,
    ) -> dict:
        self.trace = []
        t0 = time.time()
        system = self._system_template.format(tools_desc=build_tools_desc(self.tools))
        history = f"Question: {question}\n\n"
        final_answer = ""

        for step_idx in range(self.max_steps):
            llm_out = llm_chat(
                system, history, temperature=0.0, max_tokens=768, stop=["Observation:"]
            )
            thought, action, action_input = self._parse(llm_out)
            step = {
                "idx": step_idx,
                "agent": self.agent_name,
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "observation": None,
            }

            if action == "Final Answer":
                step["final"] = True
                final_answer = action_input
                self.trace.append(step)
                if on_step:
                    on_step(step)
                break

            step["final"] = False
            if on_step:
                on_step(step)

            observation = self._exec_tool(action, action_input, shared_state)
            step["observation"] = observation
            step["done"] = True
            self.trace.append(step)
            if on_step:
                on_step(step)

            history += llm_out + f"Observation: {observation[:1200]}\n"
        else:
            final_answer = "（已达最大步数）" + (self.trace[-1].get("observation", "") or "")
            step = {
                "idx": self.max_steps,
                "agent": self.agent_name,
                "thought": "达到步数上限",
                "action": "Final Answer",
                "action_input": final_answer,
                "observation": None,
                "final": True,
            }
            self.trace.append(step)
            if on_step:
                on_step(step)

        return {
            "final_answer": final_answer,
            "trace": self.trace,
            "duration": round(time.time() - t0, 2),
        }

    def _parse(self, text: str) -> tuple[str, str, str]:
        thought = ""
        m = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.S)
        if m:
            thought = m.group(1).strip()[:400]

        mfa = re.search(r"Final Answer:\s*(.*)", text, re.S)
        if mfa:
            ans = mfa.group(1).strip()
            if "<tool_call>" in ans:
                tc = self._parse_tool_call(ans)
                if tc:
                    return thought, tc[0], tc[1]
            return thought, "Final Answer", ans

        tc = self._parse_tool_call(text)
        if tc:
            return thought or "调用工具", tc[0], tc[1]

        ma = re.search(r"Action:\s*(.*)", text)
        mi = re.search(r"Action Input:\s*(.*)", text)
        if ma:
            action = ma.group(1).strip()
            action_input = mi.group(1).strip() if mi else ""
            return thought, action, action_input

        if text.strip():
            tc = self._parse_tool_call(text)
            if tc:
                return thought or "调用工具", tc[0], tc[1]
            return thought or "综合结果", "Final Answer", text.strip()
        return thought, "", ""

    def _parse_tool_call(self, text: str) -> tuple[str, str] | None:
        """解析 qwen 等模型的 <tool_call> XML 格式。"""
        m = re.search(
            r"<tool_call>\s*(\w+)\s*.*?<arg_value>(.*?)</arg_value>",
            text,
            re.S,
        )
        if m:
            return m.group(1).strip(), m.group(2).strip()
        if "dispatch_subagents" in text and "|" in text:
            m2 = re.search(r"dispatch_subagents[^\|]*(\|.+\|.+)", text)
            if m2:
                return "dispatch_subagents", m2.group(1).strip().lstrip(":")
        return None

    def _exec_tool(self, action: str, action_input: str, shared_state: dict) -> str:
        if action not in self.tools:
            return f"工具 '{action}' 不存在，可选: {list(self.tools.keys())}"
        fn, _ = self.tools[action]
        try:
            if shared_state is not None:
                return str(fn(action_input, shared_state=shared_state))
            return str(fn(action_input))
        except Exception as e:
            return f"工具执行出错: {type(e).__name__}: {str(e)[:120]}"
