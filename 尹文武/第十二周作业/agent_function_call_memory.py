"""
Function Calling API 版 ReAct Agent

教学重点：
  1. 与手写版对比：框架帮你处理格式解析，但 Thought 过程在内部不可见
  2. tool_choice="auto" 让模型自己决定调用哪个工具或直接回答
  3. finish_reason 判断：tool_calls 表示继续调用，stop 表示给出最终答案
  4. 相同工具集，相同问题，对比两种实现的稳定性和步骤数
  5. 使用 ShortTermMemory 保存最近若干轮完整对话，实现短期记忆

使用方式：
  # 原有单次问答方式
  python react_function_calling.py
  python react_function_calling.py --question "茅台近一年股价涨跌幅如何？"
  python react_function_calling.py --question "..." --max_steps 8

  # 连续对话模式，自动复用短期记忆
  python react_function_calling.py --chat

  # 设置最多记住最近 10 轮对话
  python react_function_calling.py --chat --memory-turns 10

连续对话命令：
  /clear   清空短期记忆
  /memory  查看当前记忆状态
  /exit    退出程序

依赖：
  pip install openai faiss-cpu sentence-transformers akshare

环境变量：
  Windows PowerShell:
    $env:DEEPSEEK_API_KEY="sk-xxx"

  Linux / macOS:
    export DEEPSEEK_API_KEY="sk-xxx"
"""

import os
import json
import time
import logging
import argparse

from collections import deque
from copy import deepcopy
from threading import RLock
from typing import Deque, Generator, Optional, Any

from openai import OpenAI


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ── 模型配置 ────────────────────────────────────────────────────────────────

# client = OpenAI(
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )
# MODEL = os.getenv("AGENT_MODEL", "qwen-max")

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

MODEL = os.getenv("AGENT_MODEL", "deepseek-v4-flash")


FC_SYSTEM_PROMPT = """你是一个专业的A股金融分析助手。
规则：
- 调用 financial_indicator 或 stock_price 之前，必须先用 company_lookup 获取股票代码
- 数字计算必须使用 calculator 工具，不能心算
- Final Answer 必须引用具体数据来源
- 如果没有合适工具能回答，直接说明原因
"""


# ── 短期记忆 ────────────────────────────────────────────────────────────────

class ShortTermMemory:
    """
    Agent 短期记忆。

    保存最近 max_turns 轮完整对话。每一轮可以包含：

        user
        assistant（tool_calls）
        tool
        assistant（final answer）

    与只保存“用户问题 + 最终答案”相比，保存完整工具调用轨迹可以让模型在
    后续问题中继续使用之前查到的股票代码、财务指标、股价数据等信息。

    注意：
        1. 这是进程内记忆，程序退出后会消失。
        2. deque(maxlen=max_turns) 会自动删除最早的对话。
        3. 每个用户会话应使用独立的 ShortTermMemory 实例。
    """

    def __init__(self, max_turns: int = 6):
        if max_turns <= 0:
            raise ValueError("max_turns 必须大于 0")

        self.max_turns = max_turns

        # 每个元素是一轮完整对话，对话内部是若干 OpenAI message
        self._turns: Deque[list[dict[str, Any]]] = deque(
            maxlen=max_turns
        )

        # 避免在多线程环境下同时读写记忆时出现数据竞争
        self._lock = RLock()

    def add_turn(self, messages: list[dict[str, Any]]) -> None:
        """
        保存一轮已经完成的对话。

        Args:
            messages:
                当前轮的完整消息，例如：

                [
                    {"role": "user", "content": "..."},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [...]
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "...",
                        "content": "..."
                    },
                    {"role": "assistant", "content": "..."}
                ]
        """
        if not messages:
            return

        with self._lock:
            # 使用深拷贝，避免外部 messages 后续修改影响记忆内容
            self._turns.append(deepcopy(messages))

    def get_messages(self) -> list[dict[str, Any]]:
        """
        将最近若干轮对话展开为 OpenAI API 所需的 messages 列表。

        Returns:
            [
                {"role": "user", ...},
                {"role": "assistant", ...},
                {"role": "tool", ...},
                ...
            ]
        """
        with self._lock:
            result: list[dict[str, Any]] = []

            for turn in self._turns:
                result.extend(deepcopy(turn))

            return result

    def clear(self) -> None:
        """清空全部短期记忆。"""
        with self._lock:
            self._turns.clear()

    @property
    def turn_count(self) -> int:
        """当前已经保存的完整对话轮数。"""
        with self._lock:
            return len(self._turns)

    def __len__(self) -> int:
        return self.turn_count


def _assistant_tool_message_to_dict(msg: Any) -> dict[str, Any]:
    """
    将 OpenAI SDK 返回的 ChatCompletionMessage 转换成可长期放入
    messages 列表的普通字典。

    不直接使用 msg.model_dump()，是为了避免保存 reasoning_content、
    annotations、audio 等不同模型厂商可能返回的额外字段，导致下一次
   请求时出现兼容性问题。
    """
    result: dict[str, Any] = {
        "role": "assistant",
        "content": msg.content,
    }

    if msg.tool_calls:
        result["tool_calls"] = []

        for tool_call in msg.tool_calls:
            result["tool_calls"].append({
                "id": tool_call.id,
                "type": tool_call.type or "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            })

    return result


# ── Agent 主循环 ────────────────────────────────────────────────────────────

def run(
    question: str,
    max_steps: int = 10,
    memory: Optional[ShortTermMemory] = None,
) -> Generator[dict, None, None]:
    """
    执行 Function Calling 版 ReAct 循环，yield 每一步结构化结果。

    格式与 react_manual.run() 保持一致，便于 evaluate.py 统一对比。

    Args:
        question:
            当前用户问题。

        max_steps:
            最大模型调用步数。

        memory:
            短期记忆对象。

            如果需要多轮对话，连续调用 run() 时必须传入同一个
            ShortTermMemory 对象。

            如果传入 None，则本轮不读取也不保存跨轮记忆，
            保持原有单轮运行行为。

    Examples:
        memory = ShortTermMemory(max_turns=6)

        list(run(
            "贵州茅台2023年的毛利率是多少？",
            memory=memory,
        ))

        list(run(
            "那五粮液呢？",
            memory=memory,
        ))
    """
    from tools import TOOLS_MAP, TOOLS_SCHEMA

    # system prompt 永远放在第一位
    messages: list[Any] = [
        {
            "role": "system",
            "content": FC_SYSTEM_PROMPT,
        }
    ]

    # 加载之前已经完成的若干轮对话
    if memory is not None:
        messages.extend(memory.get_messages())

    user_message = {
        "role": "user",
        "content": question,
    }

    messages.append(user_message)

    # 单独保存“当前轮”的完整消息轨迹。
    # 只有当前轮正常获得 Final Answer 后，才写入短期记忆。
    current_turn_messages: list[dict[str, Any]] = [
        deepcopy(user_message)
    ]

    for step in range(1, max_steps + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0,
        )

        msg = response.choices[0].message
        reason = response.choices[0].finish_reason

        # 模型决定直接回答，不再调用工具
        if reason == "stop" or not msg.tool_calls:
            answer = msg.content or "（模型返回空内容）"

            # 保存当前轮最终回答
            current_turn_messages.append({
                "role": "assistant",
                "content": answer,
            })

            # 只有完成一轮问答后才写入记忆
            if memory is not None:
                memory.add_turn(current_turn_messages)

            yield {
                "step": step,
                "type": "final",
                "thought": "",
                "answer": answer,
            }
            return

        # 模型请求调用工具
        #
        # messages.append(msg) 保留原有实现，不改变当前轮 Function Calling 逻辑。
        messages.append(msg)

        # 同时保存一份普通字典，用于跨轮短期记忆
        current_turn_messages.append(
            _assistant_tool_message_to_dict(msg)
        )

        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name

            try:
                tool_args = json.loads(
                    tool_call.function.arguments
                )
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

            observation_text = str(observation)

            step_result = {
                "step": step,
                "type": "action",
                "thought": "",
                "action": tool_name,
                "action_input": tool_args,
                "observation": observation_text,
            }

            yield step_result

            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": observation_text,
            }

            # 给当前轮模型继续推理使用
            messages.append(tool_message)

            # 给后续轮次作为短期记忆使用
            current_turn_messages.append(
                deepcopy(tool_message)
            )

    # 达到最大步数时，不把这一轮写入记忆。
    #
    # 原因是这轮对话没有形成完整 Final Answer，把不完整轨迹写入记忆
    # 可能让后续问题误以为上一轮已经完成。
    yield {
        "step": max_steps + 1,
        "type": "max_steps",
        "answer": f"已达最大步数 {max_steps}，未能得出最终答案",
    }


# ── CLI 打印 ────────────────────────────────────────────────────────────────

COLORS = {
    "thought": "\033[36m",
    "action": "\033[33m",
    "obs": "\033[32m",
    "final": "\033[35m",
    "error": "\033[31m",
    "reset": "\033[0m",
}


def _c(color: str, text: str) -> str:
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def run_and_print(
    question: str,
    max_steps: int = 10,
    memory: Optional[ShortTermMemory] = None,
) -> None:
    """
    执行 Agent 并打印每一步。

    memory 参数新增在最后，不影响原有调用方式：

        run_and_print(question)
        run_and_print(question, max_steps=8)

    多轮对话时：

        memory = ShortTermMemory()
        run_and_print(question1, memory=memory)
        run_and_print(question2, memory=memory)
    """
    print(f"\n{'=' * 60}")
    print(f"问题: {question}")
    print(f"模型: {MODEL}  实现: Function Calling")

    if memory is not None:
        print(
            f"短期记忆: {memory.turn_count}/"
            f"{memory.max_turns} 轮"
        )

    print("=" * 60)

    start = time.time()

    for step_data in run(
        question,
        max_steps=max_steps,
        memory=memory,
    ):
        stype = step_data["type"]

        if stype == "action":
            print(f"\n[Step {step_data['step']}]")

            # Thought 在 FC 版不可见，显示提示
            print(_c(
                "thought",
                "🧠 Thought: （模型内部推理，Function Calling 版不可见）",
            ))

            print(_c(
                "action",
                f"🔧 Action:  {step_data['action']}",
            ))

            print(_c(
                "action",
                "   Input:   "
                + json.dumps(
                    step_data["action_input"],
                    ensure_ascii=False,
                ),
            ))

            print(_c(
                "obs",
                f"👁  Obs:     "
                f"{step_data['observation'][:300]}",
            ))

        elif stype == "final":
            elapsed = time.time() - start

            print(f"\n{'─' * 60}")
            print(_c(
                "final",
                f"\n✅ Final Answer:\n"
                f"{step_data['answer']}",
            ))

            print(
                f"\n共 {step_data['step']} 步，"
                f"耗时 {elapsed:.1f}s"
            )

        elif stype in ("error", "max_steps"):
            print(_c(
                "error",
                f"\n⚠️  {step_data.get('answer', '')}",
            ))


# ── 连续对话模式 ────────────────────────────────────────────────────────────

def chat_loop(
    max_steps: int,
    memory: ShortTermMemory,
) -> None:
    """
    连续对话模式。

    整个循环复用同一个 ShortTermMemory 实例，因此后一个问题可以
    读取前面问题的完整上下文和工具调用结果。
    """
    print("\n" + "=" * 60)
    print("Function Calling ReAct Agent 连续对话模式")
    print(f"模型: {MODEL}")
    print(f"短期记忆上限: 最近 {memory.max_turns} 轮")
    print("=" * 60)

    print("\n可用命令：")
    print("  /clear   清空短期记忆")
    print("  /memory  查看记忆状态")
    print("  /exit    退出程序")

    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if not question:
            continue

        command = question.lower()

        if command in {
            "/exit",
            "exit",
            "quit",
            "q",
            "退出",
        }:
            print("已退出。")
            break

        if command in {
            "/clear",
            "clear",
            "清空",
        }:
            memory.clear()
            print("短期记忆已清空。")
            continue

        if command in {
            "/memory",
            "memory",
            "记忆",
        }:
            print(
                f"当前保存 {memory.turn_count} 轮对话，"
                f"最大保留 {memory.max_turns} 轮。"
            )
            continue

        run_and_print(
            question=question,
            max_steps=max_steps,
            memory=memory,
        )


# ── 程序入口 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--question",
        default="贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？",
        help="单次问答的问题",
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=10,
        help="每轮问答允许的最大 Agent 步数",
    )

    parser.add_argument(
        "--chat",
        action="store_true",
        help="启动连续对话模式",
    )

    parser.add_argument(
        "--memory-turns",
        type=int,
        default=6,
        help="短期记忆最多保留的完整对话轮数，默认 6",
    )

    args = parser.parse_args()

    if args.memory_turns <= 0:
        parser.error("--memory-turns 必须大于 0")

    # 一个 ShortTermMemory 实例对应一个用户会话
    session_memory = ShortTermMemory(
        max_turns=args.memory_turns
    )

    if args.chat:
        chat_loop(
            max_steps=args.max_steps,
            memory=session_memory,
        )
    else:
        # 保持原有单次问答逻辑，只是额外传入了记忆对象
        run_and_print(
            question=args.question,
            max_steps=args.max_steps,
            memory=session_memory,
        )
