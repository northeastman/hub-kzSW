"""
run_single_turn.py — 对照组：单轮模式（每问独立，无历史）

与 run_multi_turn.py 对比：同一追问「那五粮液呢？」在单轮模式下无法理解上文。

使用方式：
  python run_single_turn.py --demo
  python run_single_turn.py -q "那五粮液呢？"
"""

import os
import sys
import json
import time
import argparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from conversation_agent import run_single_turn

COLORS = {
    "user": "\033[94m",
    "assistant": "\033[35m",
    "action": "\033[33m",
    "meta": "\033[90m",
    "warn": "\033[31m",
    "reset": "\033[0m",
}


def _c(color: str, text: str) -> str:
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def print_result(result, elapsed: float):
    print(f"\n{'─' * 60}")
    print(_c("user", f"👤 用户: {result.question}"))
    for step_data in result.steps:
        if step_data.get("type") == "action":
            print(_c("action", f"  🔧 {step_data['action']}"))
    print(_c("assistant", f"\n🤖 助手:\n{result.answer}"))
    print(_c("meta", f"  （{elapsed:.1f}s，无历史上下文）"))


def run_demo(mode: str, max_steps: int):
    """
    对照实验：
      第1轮正常提问 → 单轮/多轮都能答
      第2轮追问「那五粮液呢？」→ 单轮模式缺少上文，行为明显不同
    """
    print("=" * 60)
    print("  单轮模式对照 Demo")
    print("  注意：每轮都是全新会话，不保留上一轮答案")
    print("=" * 60)

    turns = [
        "贵州茅台2023年的毛利率是多少？",
        "那五粮液呢？",
    ]

    for i, q in enumerate(turns, 1):
        print(_c("meta", f"\n>>> 独立第 {i} 问（history 为空）"))
        start = time.time()
        result = run_single_turn(q, mode=mode, max_steps=max_steps)
        print_result(result, time.time() - start)

        if i == 2:
            print(_c("warn", "\n⚠️  单轮模式无法知道「那」指什么 — 这正是多轮对话要解决的问题"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="单轮 ReAct Agent（对照组）")
    parser.add_argument("--mode", choices=["manual", "fc"], default="manual")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("-q", "--question", default="")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        run_demo(args.mode, args.max_steps)
    elif args.question:
        start = time.time()
        result = run_single_turn(args.question, mode=args.mode, max_steps=args.max_steps)
        print_result(result, time.time() - start)
    else:
        print("请指定 -q 问题或 --demo", file=sys.stderr)
        sys.exit(1)
