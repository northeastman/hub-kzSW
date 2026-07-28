"""
run_multi_turn.py — 第十二周作业：多轮对话 ReAct Agent

交互式 CLI，支持连续追问。会话内保留历史，可理解「那五粮液呢？」类省略主语的问题。

使用方式：
  python run_multi_turn.py
  python run_multi_turn.py --mode fc
  python run_multi_turn.py --demo

依赖：
  pip install -r requirements.txt
  环境变量 DASHSCOPE_API_KEY（manual 模式）或 DEEPSEEK_API_KEY（fc 模式）
"""

import os
import sys
import json
import time
import argparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from conversation_agent import ConversationSession

COLORS = {
    "user": "\033[94m",
    "assistant": "\033[35m",
    "thought": "\033[36m",
    "action": "\033[33m",
    "obs": "\033[32m",
    "meta": "\033[90m",
    "reset": "\033[0m",
}


def _c(color: str, text: str) -> str:
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def print_turn(result, mode: str, elapsed: float):
    print(f"\n{'─' * 60}")
    print(_c("user", f"👤 用户 [第{result.turn_index}轮]: {result.question}"))
    print()

    for step_data in result.steps:
        stype = step_data.get("type")
        if stype == "action":
            print(f"  [Step {step_data['step']}]")
            if mode == "manual" and step_data.get("thought"):
                print(_c("thought", f"  🧠 {step_data['thought'][:120]}..."))
            print(_c("action", f"  🔧 {step_data['action']}({json.dumps(step_data['action_input'], ensure_ascii=False)})"))
            obs = (step_data.get("observation") or "")[:200]
            print(_c("obs", f"  👁  {obs}..."))

    print()
    print(_c("assistant", f"🤖 助手:\n{result.answer}"))
    print(_c("meta", f"  （{elapsed:.1f}s，历史轮数: {result.turn_index}）"))


def run_interactive(mode: str, max_steps: int):
    session = ConversationSession(mode=mode, max_steps=max_steps)
    print("=" * 60)
    print("  ReAct 金融 Agent — 多轮对话模式")
    print(f"  实现: {'手写Prompt解析' if mode == 'manual' else 'Function Calling'}")
    print("  输入 quit / exit 退出，输入 reset 清空历史")
    print("=" * 60)

    while True:
        try:
            question = input("\n👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if question.lower() == "reset":
            session.reset()
            print(_c("meta", "✓ 对话历史已清空"))
            continue

        start = time.time()
        result = session.ask(question)
        print_turn(result, mode, time.time() - start)


DEMO_TURNS = [
    "贵州茅台2023年的毛利率是多少？",
    "那五粮液呢？",
    "两者差多少个百分点？",
]


def run_demo(mode: str, max_steps: int):
    """脚本化演示：三轮追问，展示多轮上下文传递"""
    session = ConversationSession(mode=mode, max_steps=max_steps)
    print("=" * 60)
    print("  多轮对话 Demo — 三轮连续追问")
    print("=" * 60)

    for q in DEMO_TURNS:
        start = time.time()
        result = session.ask(q)
        print_turn(result, mode, time.time() - start)
        time.sleep(0.5)

    print(f"\n{'=' * 60}")
    print(f"  会话历史共 {len(session.history)} 条消息（{len(session.history)//2} 轮 Q&A）")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多轮对话 ReAct Agent")
    parser.add_argument("--mode", choices=["manual", "fc"], default="manual")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--demo", action="store_true", help="运行内置三轮追问 demo")
    args = parser.parse_args()

    if args.demo:
        run_demo(args.mode, args.max_steps)
    else:
        run_interactive(args.mode, args.max_steps)
