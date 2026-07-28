"""
手写 Prompt 解析版 ReAct Agent —— 多轮对话增强版

相比原版 react_manual.py 的改动：
  1. run() 新增 history 参数，接受之前的对话历史
  2. System Prompt 增加多轮对话指引
  3. 历史消息以标准 user/assistant 角色追加到 messages，模型自然延续上下文
  4. 不存储中间 ReAct 步骤，只把最终 Final Answer 存入历史（由 serve.py 负责）

使用方式：
  from react_manual_mt import run
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
import re
import json
import time
import logging
import argparse
from typing import Generator

# 复用原项目的 tools.py（含 FAISS 索引、AkShare 工具等）
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

# ── LLM 客户端 ────────────────────────────────────────────────────────────────
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
MODEL = os.getenv("AGENT_MODEL", "qwen-max")


# ── System Prompt（增加多轮对话指引）─────────────────────────────────────────
SYSTEM_PROMPT = """你是一个专业的A股金融分析助手，可以使用以下工具来回答问题：

工具列表：
1. rag_search(query) - 在年报中语义检索文本内容（战略/财务数据/风险因素等）
2. company_lookup(name) - 将公司名称转换为股票代码
3. calculator(expr) - 计算数学表达式（支持四则运算和math函数）
4. financial_indicator(symbol) - 获取实时财务指标（PE/PB/ROE等）
5. stock_price(symbol, start_date, end_date) - 获取历史股价，日期格式YYYYMMDD

你必须严格按照以下格式交替输出，每次只能调用一个工具：

Thought: 分析当前状态，决定下一步做什么
Action: 工具名称
Action Input: {"参数名": "参数值"}

收到工具结果后继续推理，直到可以给出最终答案：

Thought: 已有足够信息
Final Answer: 完整的回答（含数据来源）

规则：
- 必须先用 company_lookup 获取股票代码，再调用 financial_indicator 或 stock_price
- 数字计算必须用 calculator，不能心算
- Final Answer 必须引用具体数据来源（哪份年报哪一页，或AkShare实时数据）
- 如果没有合适工具能回答，直接输出 Final Answer 说明原因

多轮对话规则：
- 对话历史会以 user/assistant 消息形式提供，回答新问题时可参考之前已查询过的数据
- 如果新问题需要之前查询过的数据，可直接引用，避免重复调用相同工具
- 如果新问题需要新数据（如不同公司、不同年份），仍需重新调用工具获取
- 如果用户使用指代词（如"它""那家""去年"等），需结合历史判断具体指代对象
"""

# ── 格式解析（与原版一致）─────────────────────────────────────────────────────
_THOUGHT_RE      = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)", re.DOTALL)
_ACTION_RE       = re.compile(r"Action:\s*(\w+)")
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(\{.+?\})", re.DOTALL)
_FINAL_RE        = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)


def _parse_step(text: str) -> dict:
    """从 LLM 输出中解析一步的结构化内容"""
    final = _FINAL_RE.search(text)
    if final:
        thought_m = _THOUGHT_RE.search(text)
        return {
            "type":    "final",
            "thought": thought_m.group(1).strip() if thought_m else "",
            "answer":  final.group(1).strip(),
        }

    thought_m = _THOUGHT_RE.search(text)
    action_m  = _ACTION_RE.search(text)
    input_m   = _ACTION_INPUT_RE.search(text)

    if not action_m:
        return {"type": "unparseable", "raw": text}

    try:
        action_input = json.loads(input_m.group(1)) if input_m else {}
    except json.JSONDecodeError:
        action_input = {}

    return {
        "type":         "action",
        "thought":      thought_m.group(1).strip() if thought_m else "",
        "action":       action_m.group(1).strip(),
        "action_input": action_input,
    }


# ── ReAct 核心循环（多轮版）────────────────────────────────────────────────────

def run(
    question: str,
    max_steps: int = 10,
    history: list[dict] | None = None,
    verbose: bool = True,
) -> Generator[dict, None, None]:
    """
    执行多轮 ReAct 循环

    Args:
        question: 当前用户问题
        max_steps: 最大工具调用步数
        history: 之前的对话历史，格式 [{"role": "user"|"assistant", "content": "..."}]
                 只含最终问答，不含中间 ReAct 步骤

    Yields:
        每个 dict 表示一步：
          {"step": int, "type": "action", "thought": str, "action": str,
           "action_input": dict, "observation": str}
        最后一步：
          {"step": int, "type": "final", "thought": str, "answer": str}
    """
    from tools import TOOLS_MAP

    # ── 拼接 messages：system + 历史 + 当前问题 ──────────────────────────────
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        # 历史已经是标准 [{"role":"user","content":"..."},{"role":"assistant","content":"..."}] 格式
        messages.extend(history)
        if verbose:
            logger.info(f"多轮对话：携带 {len(history)//2} 轮历史")

    messages.append({"role": "user", "content": question})

    for step in range(1, max_steps + 1):
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
                "step":    step,
                "type":    "final",
                "thought": parsed["thought"],
                "answer":  parsed["answer"],
            }
            return

        if parsed["type"] == "unparseable":
            yield {
                "step":        step,
                "type":        "error",
                "observation": f"格式解析失败，原始输出：{llm_output[:200]}",
            }
            return

        # 执行工具
        tool_name  = parsed["action"]
        tool_args  = parsed["action_input"]
        tool_fn    = TOOLS_MAP.get(tool_name)

        if tool_fn is None:
            observation = f"未知工具 '{tool_name}'，可用工具：{list(TOOLS_MAP.keys())}"
        else:
            try:
                observation = tool_fn(**tool_args)
            except TypeError as e:
                observation = f"工具参数错误: {e}"

        step_result = {
            "step":         step,
            "type":         "action",
            "thought":      parsed["thought"],
            "action":       tool_name,
            "action_input": tool_args,
            "observation":  str(observation),
        }
        yield step_result

        # 将本步结果追加到对话历史（这是单轮内的 ReAct 记忆）
        messages.append({"role": "assistant", "content": llm_output})
        messages.append({
            "role":    "user",
            "content": f"Observation: {observation}\n",
        })

    yield {
        "step":   max_steps + 1,
        "type":   "max_steps",
        "answer": f"已达最大步数 {max_steps}，未能得出最终答案",
    }


# ── CLI 测试入口 ──────────────────────────────────────────────────────────────

COLORS = {
    "thought":  "\033[36m",
    "action":   "\033[33m",
    "obs":      "\033[32m",
    "final":    "\033[35m",
    "error":    "\033[31m",
    "reset":    "\033[0m",
}

def _c(color: str, text: str) -> str:
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def run_and_print(question: str, max_steps: int = 10, history: list[dict] | None = None):
    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print(f"模型: {MODEL}  实现: 手写Prompt解析（多轮版）")
    if history:
        print(f"历史: 携带 {len(history)//2} 轮对话")
    print('='*60)

    start = time.time()

    for step_data in run(question, max_steps=max_steps, history=history):
        stype = step_data["type"]

        if stype == "action":
            print(f"\n[Step {step_data['step']}]")
            print(_c("thought", f"🧠 Thought: {step_data['thought']}"))
            print(_c("action",  f"🔧 Action:  {step_data['action']}"))
            print(_c("action",  f"   Input:   {json.dumps(step_data['action_input'], ensure_ascii=False)}"))
            print(_c("obs",     f"👁  Obs:     {step_data['observation'][:300]}"))

        elif stype == "final":
            elapsed = time.time() - start
            print(f"\n{'─'*60}")
            if step_data.get("thought"):
                print(_c("thought", f"🧠 Thought: {step_data['thought']}"))
            print(_c("final",  f"\n✅ Final Answer:\n{step_data['answer']}"))
            print(f"\n共 {step_data['step']} 步，耗时 {elapsed:.1f}s")

        elif stype in ("error", "max_steps"):
            print(_c("error", f"\n⚠️  {step_data.get('answer', step_data.get('observation', ''))}"))

    return step_data.get("answer", "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多轮对话版 ReAct Agent（手写解析）")
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
        # 把本轮结果存入历史，供下一轮使用
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer})
