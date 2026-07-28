"""
Function Calling API 版 ReAct Agent —— 多轮对话增强版

相比原版 react_function_calling.py 的改动：
  1. run() 新增 history 参数
  2. System Prompt 增加多轮对话指引
  3. 历史消息以标准 user/assistant 角色追加到 messages
  4. Function Calling 天然支持多轮，OpenAI Messages API 就是为此设计的

使用方式：
  from react_function_calling_mt import run
  history = [
      {"role": "user", "content": "茅台2023年毛利率？"},
      {"role": "assistant", "content": "茅台2023年毛利率为91.96%..."},
  ]
  for step in run("那五粮液呢？", history=history):
      print(step)
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import argparse
from typing import Generator

# 复用原项目的 tools.py
_HERE = os.path.dirname(os.path.abspath(__file__))
_ORIG_SRC = os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "note", "week12 agent", "react_financial_agent", "src")
)
if _ORIG_SRC not in sys.path:
    sys.path.insert(0, _ORIG_SRC)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

from openai import OpenAI

# ── LLM 客户端（默认用 DashScope qwen-max；可切换 DeepSeek）──────────────────
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

多轮对话规则：
- 对话历史会以 user/assistant 消息形式提供，回答新问题时可参考之前已查询过的数据
- 如果新问题需要之前查询过的数据，可直接引用，避免重复调用相同工具
- 如果新问题需要新数据（如不同公司、不同年份），仍需重新调用工具获取
- 如果用户使用指代词（如"它""那家""去年"等），需结合历史判断具体指代对象
"""


def run(
    question: str,
    max_steps: int = 10,
    history: list[dict] | None = None,
) -> Generator[dict, None, None]:
    """
    执行多轮 Function Calling 版 ReAct 循环

    Args:
        question: 当前用户问题
        max_steps: 最大工具调用步数
        history: 之前的对话历史，格式 [{"role": "user"|"assistant", "content": "..."}]

    Yields:
        每个 dict 表示一步（与 react_manual_mt.run 格式一致，便于统一对比）
    """
    from tools import TOOLS_MAP, TOOLS_SCHEMA

    # ── 拼接 messages：system + 历史 + 当前问题 ──────────────────────────────
    messages = [{"role": "system", "content": FC_SYSTEM_PROMPT}]

    if history:
        messages.extend(history)
        logger.info(f"多轮对话：携带 {len(history)//2} 轮历史")

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


# ── CLI 测试入口 ──────────────────────────────────────────────────────────────

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


def run_and_print(question: str, max_steps: int = 10, history: list[dict] | None = None) -> str:
    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print(f"模型: {MODEL}  实现: Function Calling（多轮版）")
    if history:
        print(f"历史: 携带 {len(history)//2} 轮对话")
    print('='*60)

    start = time.time()
    final_answer = ""

    for step_data in run(question, max_steps=max_steps, history=history):
        stype = step_data["type"]

        if stype == "action":
            print(f"\n[Step {step_data['step']}]")
            print(_c("thought", "🧠 Thought: （模型内部推理，Function Calling 版不可见）"))
            print(_c("action",  f"🔧 Action:  {step_data['action']}"))
            print(_c("action",  f"   Input:   {json.dumps(step_data['action_input'], ensure_ascii=False)}"))
            print(_c("obs",     f"👁  Obs:     {step_data['observation'][:300]}"))

        elif stype == "final":
            elapsed = time.time() - start
            print(f"\n{'─'*60}")
            print(_c("final", f"\n✅ Final Answer:\n{step_data['answer']}"))
            print(f"\n共 {step_data['step']} 步，耗时 {elapsed:.1f}s")
            final_answer = step_data["answer"]

        elif stype in ("error", "max_steps"):
            print(_c("error", f"\n⚠️  {step_data.get('answer', '')}"))
            final_answer = step_data.get("answer", "")

    return final_answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多轮对话版 ReAct Agent（Function Calling）")
    parser.add_argument("--q1", default="茅台2023年毛利率是多少？")
    parser.add_argument("--q2", default="那五粮液呢？")
    parser.add_argument("--q3", default="两家差多少个百分点？")
    parser.add_argument("--max_steps", type=int, default=10)
    args = parser.parse_args()

    # 模拟三轮对话
    history = []
    for i, q in enumerate([args.q1, args.q2, args.q3], 1):
        print(f"\n{'#'*60}")
        print(f"# 第 {i} 轮对话")
        print(f"{'#'*60}")
        answer = run_and_print(q, max_steps=args.max_steps, history=history)
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer})
