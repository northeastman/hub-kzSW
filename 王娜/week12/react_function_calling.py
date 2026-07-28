"""
Function Calling API 版 ReAct Agent

教学重点：
  1. 与手写版对比：框架帮你处理格式解析，但 Thought 过程在内部不可见
  2. tool_choice="auto" 让模型自己决定调用哪个工具或直接回答
  3. finish_reason 判断：tool_calls 表示继续调用，stop 表示给出最终答案
  4. 相同工具集，相同问题，对比两种实现的稳定性和步骤数

使用方式：
  python react_function_calling.py                       # 进入多轮对话模式
  python react_function_calling.py --question "五粮液2023年净利润是多少？"
  python react_function_calling.py --question "..." --max_steps 8

依赖：
  pip install openai faiss-cpu sentence-transformers akshare
  # 默认使用 Kimi（Moonshot）
  export KIMI_API_KEY="sk-xxx"
  # 或切换到 DeepSeek
  export LLM_PROVIDER="deepseek"
  export DEEPSEEK_API_KEY="sk-xxx"
"""

import os
import json
import time
import logging
import argparse
from typing import Generator

from openai import OpenAI

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── LLM 客户端 ────────────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "kimi")
if LLM_PROVIDER == "deepseek":
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    MODEL = os.getenv("AGENT_MODEL", "deepseek-chat")
else:
    client = OpenAI(
        api_key=os.getenv("KIMI_API_KEY"),
        base_url="https://api.moonshot.cn/v1",
    )
    MODEL = os.getenv("AGENT_MODEL", "moonshot-v1-8k")


# ── 短期会话记忆 ───────────────────────────────────────────────────────────────
class SessionMemory:
    """保存同一 session 内的多轮问答，用于省略/指代理解"""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.history = []

    def add(self, user_query: str, final_answer: str):
        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "assistant", "content": final_answer})
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-self.max_turns * 2:]

    def reset(self):
        self.history = []


FC_SYSTEM_PROMPT = """你是一个专业的A股金融分析助手。
规则：
- 调用 financial_indicator 或 stock_price 之前，必须先用 company_lookup 获取股票代码
- 数字计算必须使用 calculator 工具，不能心算
- Final Answer 必须引用具体数据来源
- 如果没有合适工具能回答，直接说明原因
- 如果用户问题涉及历史对话中的指标/公司/时间，请结合上下文补全省略表述（如"茅台呢"指"贵州茅台的同类指标"）
"""


def run(question: str, max_steps: int = 10, memory: SessionMemory | None = None) -> Generator[dict, None, None]:
    """
    执行 Function Calling 版 ReAct 循环，yield 每一步结构化结果

    格式与 react_manual.run() 保持一致，便于 evaluate.py 统一对比
    """
    from tools import TOOLS_MAP, TOOLS_SCHEMA

    messages = [{"role": "system", "content": FC_SYSTEM_PROMPT}]
    if memory:
        messages.extend(memory.history)
    messages.append({"role": "user", "content": question})

    for step in range(1, max_steps + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0,
        )
        msg    = response.choices[0].message
        reason = response.choices[0].finish_reason

        # 模型决定直接回答（无工具调用）
        if reason == "stop" or not msg.tool_calls:
            yield {
                "step":   step,
                "type":   "final",
                "thought": "",
                "answer": msg.content or "（模型返回空内容）",
            }
            return

        # 模型请求调用工具
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

            step_result = {
                "step":         step,
                "type":         "action",
                "thought":      "",   # Function Calling 版 Thought 在模型内部，不可见
                "action":       tool_name,
                "action_input": tool_args,
                "observation":  str(observation),
            }
            yield step_result

            messages.append({
                "role":         "tool",
                "tool_call_id": tool_call.id,
                "content":      str(observation),
            })

    yield {
        "step":   max_steps + 1,
        "type":   "max_steps",
        "answer": f"已达最大步数 {max_steps}，未能得出最终答案",
    }


# ── CLI 打印（复用 react_manual 的彩色输出） ───────────────────────────────────

COLORS = {
    "thought": "\033[36m",
    "action":  "\033[33m",
    "obs":     "\033[32m",
    "final":   "\033[35m",
    "error":   "\033[31m",
    "reset":   "\033[0m",
}

def _c(color: str, text: str) -> str:
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def run_and_print(question: str, max_steps: int = 10, memory: SessionMemory | None = None) -> SessionMemory:
    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print(f"模型: {MODEL}  实现: Function Calling")
    print('='*60)

    start = time.time()
    final_answer = ""

    for step_data in run(question, max_steps=max_steps, memory=memory):
        stype = step_data["type"]

        if stype == "action":
            print(f"\n[Step {step_data['step']}]")
            # Thought 在 FC 版不可见，显示提示
            print(_c("thought", "🧠 Thought: （模型内部推理，Function Calling 版不可见）"))
            print(_c("action",  f"🔧 Action:  {step_data['action']}"))
            print(_c("action",  f"   Input:   {json.dumps(step_data['action_input'], ensure_ascii=False)}"))
            print(_c("obs",     f"👁  Obs:     {step_data['observation'][:300]}"))

        elif stype == "final":
            elapsed = time.time() - start
            final_answer = step_data["answer"]
            print(f"\n{'─'*60}")
            print(_c("final", f"\n✅ Final Answer:\n{final_answer}"))
            print(f"\n共 {step_data['step']} 步，耗时 {elapsed:.1f}s")

        elif stype in ("error", "max_steps"):
            print(_c("error", f"\n⚠️  {step_data.get('answer', '')}"))

    # 将本轮问答存入记忆
    if memory is not None:
        memory.add(question, final_answer)
    return memory


def interactive_mode(max_steps: int = 10):
    """多轮对话交互模式，支持同一 session 内的省略/指代"""
    print(_c("final", "🧠 多轮对话模式已启动（输入 'exit' 退出，'reset' 重置记忆，'!clear' 清屏）"))
    print("="*60)

    memory = SessionMemory()
    history_count = 0

    while True:
        try:
            question = input(f"\n{_c('thought', '💬 你: ')}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print(_c("final", "👋 再见！"))
            break
        if question.lower() == "reset":
            memory.reset()
            history_count = 0
            print(_c("obs", "🔄 记忆已重置"))
            continue
        if question.lower() == "!clear":
            os.system("cls" if os.name == "nt" else "clear")
            continue

        history_count += 1
        memory = run_and_print(question, max_steps=max_steps, memory=memory)

    print(f"\n本次会话共 {history_count} 轮问答")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question",  default=None, help="单轮问题（不指定则进入多轮对话模式）")
    parser.add_argument("--max_steps", type=int, default=10)
    args = parser.parse_args()

    if args.question:
        run_and_print(args.question, max_steps=args.max_steps)
    else:
        interactive_mode(max_steps=args.max_steps)
