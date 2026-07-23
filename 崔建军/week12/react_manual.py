"""
手写 Prompt 解析版 ReAct Agent

教学重点：
  1. ReAct 核心循环：Thought → Action → Observation，逐步推理
  2. System Prompt 约束输出格式，Python 正则解析每一步
  3. 对话历史拼接方式：每轮结果追加到 prompt，形成上下文记忆
  4. 停止条件：模型输出 Final Answer 或达到最大步数

使用方式：
  python react_manual.py
  python react_manual.py --question "茅台和五粮液2023年毛利率差多少？"
  python react_manual.py --question "..." --max_steps 8 --verbose

依赖：
  pip install openai faiss-cpu sentence-transformers akshare
  export DASHSCOPE_API_KEY="sk-xxx"
"""

import os
import re
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
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
MODEL = os.getenv("AGENT_MODEL", "qwen-max")
# client = OpenAI(
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://api.deepseek.com",
# )
# MODEL = os.getenv("AGENT_MODEL", "deepseek-v4-flash")


# ── System Prompt ─────────────────────────────────────────────────────────────
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
"""

# ── 格式解析 ──────────────────────────────────────────────────────────────────
_THOUGHT_RE      = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)", re.DOTALL)
_ACTION_RE       = re.compile(r"Action:\s*(\w+)")
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(\{.+?\})", re.DOTALL)
_FINAL_RE        = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)


def _count_tokens(content: str) -> int:
    """统一的 token 估算函数，兼顾中英文混合文本"""
    chinese_chars = len([c for c in content if '\u4e00' <= c <= '\u9fff'])
    english_chars = len(content) - chinese_chars
    return int(chinese_chars * 1.5 + english_chars * 0.75)


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


# ── ReAct 核心循环 ─────────────────────────────────────────────────────────────

def run(question: str, max_steps: int = 10, verbose: bool = True, 
        messages: list = None) -> Generator[dict, None, None]:
    """
    执行 ReAct 循环，yield 每一步的结构化结果

    参数:
      question: 当前问题
      max_steps: 最大迭代步数
      verbose: 是否打印详细信息
      messages: 外部传入的对话历史列表（用于短期记忆），为 None 时创建新列表

    每个 yield 的 dict 格式：
      {"step": int, "thought": str, "action": str, "action_input": dict, "observation": str}
    最后一个 yield：
      {"step": int, "thought": str, "type": "final", "answer": str}
    """
    from tools import TOOLS_MAP

    if messages is None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ]
    else:
        messages.append({"role": "user", "content": question})

    for step in range(1, max_steps + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
            stop=["Observation:"],  # 让模型停在调用工具前
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

        # 将本步结果追加到对话历史
        messages.append({"role": "assistant", "content": llm_output})
        messages.append({
            "role":    "user",
            "content": f"Observation: {observation}\n",
        })

    # 超出最大步数，强制终止
    yield {
        "step":   max_steps + 1,
        "type":   "max_steps",
        "answer": f"已达最大步数 {max_steps}，未能得出最终答案",
    }


# ── 对话会话管理 ──────────────────────────────────────────────────────────────

class ConversationSession:
    """
    对话会话管理类，支持短期记忆功能

    功能：
      1. 维护跨轮次的对话历史（短期记忆）
      2. 上下文长度管理，避免 token 超限（截断+摘要）
      3. 清理中间步骤，只保留最终答案
      4. 提供连续对话接口
    """
    def __init__(self, max_context_tokens: int = 15000):
        """
        参数:
          max_context_tokens: 上下文最大 token 数，超过时触发截断+摘要
        """
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.max_context_tokens = max_context_tokens
        self._summary_prompt = "请用一句话简要总结以下对话的关键信息：\n{conversation}"

    def _estimate_tokens(self) -> int:
        """估算整个消息列表的 token 数"""
        return sum(_count_tokens(msg.get("content", "")) for msg in self.messages)

    def _generate_summary(self, conversation_text: str) -> str:
        """调用 LLM 生成对话摘要"""
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "你是一个对话摘要助手，用一句话总结对话要点。"},
                    {"role": "user", "content": self._summary_prompt.format(conversation=conversation_text)},
                ],
                temperature=0.3,
                max_tokens=100,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"生成摘要失败: {e}")
            return ""

    def _trim_context(self):
        """上下文过长时，对早期对话生成摘要并保留最近几轮完整对话"""
        current_tokens = self._estimate_tokens()
        if current_tokens < self.max_context_tokens * 0.8:
            return

        system_msg = self.messages[0]
        non_system_messages = self.messages[1:]

        truncated_messages = []
        summary_messages = []
        token_count = _count_tokens(system_msg.get("content", ""))
        target_tokens = int(self.max_context_tokens * 0.7)

        for msg in reversed(non_system_messages):
            msg_tokens = _count_tokens(msg.get("content", ""))

            if token_count + msg_tokens < target_tokens:
                truncated_messages.insert(0, msg)
                token_count += msg_tokens
            else:
                summary_messages.insert(0, msg)

        if summary_messages:
            conversation_text = "\n".join(
                f"{msg['role']}: {msg['content']}" for msg in summary_messages
            )
            summary = self._generate_summary(conversation_text)
            if summary:
                system_msg = {
                    "role": "system",
                    "content": SYSTEM_PROMPT + "\n\n【历史对话摘要】：" + summary,
                }

        self.messages = [system_msg] + truncated_messages
        logger.info(f"上下文已截断，当前约 {token_count} tokens")

    def ask(self, question: str, max_steps: int = 10) -> Generator[dict, None, None]:
        """
        发起新的提问，自动维护对话历史

        参数:
          question: 用户问题
          max_steps: ReAct 循环最大步数

        返回:
          Generator，yield 每一步的结构化结果
        """
        self._trim_context()

        round_start_idx = len(self.messages)

        final_result = None
        for step_data in run(question, max_steps=max_steps, messages=self.messages):
            yield step_data
            if step_data["type"] in ("final", "error", "max_steps"):
                final_result = step_data
                break

        if final_result is None:
            final_result = {
                "type": "error",
                "thought": "",
                "answer": "对话意外中断",
            }

        self.messages = self.messages[:round_start_idx]
        self.messages.append({"role": "user", "content": question})

        if final_result["type"] == "final":
            self.messages.append({
                "role": "assistant",
                "content": f"Final Answer: {final_result['answer']}",
            })
        else:
            error_msg = final_result.get("answer", final_result.get("observation", "未知错误"))
            self.messages.append({
                "role": "assistant",
                "content": f"无法回答该问题：{error_msg}",
            })

    def get_history(self) -> list:
        """获取当前完整对话历史"""
        return self.messages.copy()


# ── CLI 打印 ──────────────────────────────────────────────────────────────────

COLORS = {
    "thought":  "\033[36m",   # cyan
    "action":   "\033[33m",   # yellow
    "obs":      "\033[32m",   # green
    "final":    "\033[35m",   # magenta
    "error":    "\033[31m",   # red
    "reset":    "\033[0m",
}

def _c(color: str, text: str) -> str:
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def run_and_print(question: str, max_steps: int = 10, session: ConversationSession = None):
    """
    执行提问并打印结果，支持单次提问和会话模式

    参数:
      question: 用户问题
      max_steps: 最大步骤数
      session: 可选的对话会话对象，提供短期记忆功能
    """
    mode = "手写Prompt解析（带短期记忆）" if session else "手写Prompt解析"
    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print(f"模型: {MODEL}  实现: {mode}")
    print('='*60)

    start = time.time()

    if session:
        step_generator = session.ask(question, max_steps=max_steps)
    else:
        step_generator = run(question, max_steps=max_steps)

    for step_data in step_generator:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question",  default=None, help="单次提问的问题")
    parser.add_argument("--max_steps", type=int, default=10, help="最大步骤数")
    args = parser.parse_args()

    session = ConversationSession()

    if args.question:
        run_and_print(args.question, args.max_steps, session)
    else:
        print(f"欢迎使用 A股金融分析助手！")
        print(f"模型: {MODEL}")
        print(f"输入 'exit' 或 'quit' 退出\n")

        while True:
            try:
                question = input("请输入问题: ")
            except EOFError:
                break

            question = question.strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit", "q"):
                break

            run_and_print(question, args.max_steps, session)
