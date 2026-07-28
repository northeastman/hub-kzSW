"""
Function Calling API 版 ReAct Agent

教学重点：
  1. 与手写版对比：框架帮你处理格式解析，但 Thought 过程在内部不可见
  2. tool_choice="auto" 让模型自己决定调用哪个工具或直接回答
  3. finish_reason 判断：tool_calls 表示继续调用，stop 表示给出最终答案
  4. 相同工具集，相同问题，对比两种实现的稳定性和步骤数
  5. 短期记忆：维护跨轮次对话历史，支持连续提问

使用方式：
  python react_function_calling.py
  python react_function_calling.py --question "茅台近一年股价涨跌幅如何？"
  python react_function_calling.py --question "..." --max_steps 8

依赖：
  pip install openai faiss-cpu sentence-transformers akshare
  export DASHSCOPE_API_KEY="sk-xxx"
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


def _count_tokens(content: str) -> int:
    """统一的 token 估算函数，兼顾中英文混合文本"""
    chinese_chars = len([c for c in content if '\u4e00' <= c <= '\u9fff'])
    english_chars = len(content) - chinese_chars
    return int(chinese_chars * 1.5 + english_chars * 0.75)


def run(question: str, max_steps: int = 10, messages: list = None) -> Generator[dict, None, None]:
    """
    执行 Function Calling 版 ReAct 循环，yield 每一步结构化结果

    参数:
      question: 当前问题
      max_steps: 最大迭代步数
      messages: 外部传入的对话历史列表（用于短期记忆），为 None 时创建新列表

    格式与 react_manual.run() 保持一致，便于 evaluate.py 统一对比
    """
    from tools import TOOLS_MAP, TOOLS_SCHEMA

    if messages is None:
        messages = [
            {"role": "system", "content": FC_SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ]
    else:
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

        if reason == "stop" or not msg.tool_calls:
            yield {
                "step":   step,
                "type":   "final",
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
        self.messages = [{"role": "system", "content": FC_SYSTEM_PROMPT}]
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
                    "content": FC_SYSTEM_PROMPT + "\n\n【历史对话摘要】：" + summary,
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
    "thought": "\033[36m",
    "action":  "\033[33m",
    "obs":     "\033[32m",
    "final":   "\033[35m",
    "error":   "\033[31m",
    "reset":   "\033[0m",
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
    mode = "Function Calling（带短期记忆）" if session else "Function Calling"
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
            # Thought 在 FC 版不可见，显示提示
            print(_c("thought", "🧠 Thought: （模型内部推理，Function Calling 版不可见）"))
            print(_c("action",  f"🔧 Action:  {step_data['action']}"))
            print(_c("action",  f"   Input:   {json.dumps(step_data['action_input'], ensure_ascii=False)}"))
            print(_c("obs",     f"👁  Obs:     {step_data['observation'][:300]}"))

        elif stype == "final":
            elapsed = time.time() - start
            print(f"\n{'─'*60}")
            print(_c("final", f"\n✅ Final Answer:\n{step_data['answer']}"))
            print(f"\n共 {step_data['step']} 步，耗时 {elapsed:.1f}s")

        elif stype in ("error", "max_steps"):
            print(_c("error", f"\n⚠️  {step_data.get('answer', '')}"))


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