"""
统一入口：简化版 ReAct Agent

使用方式：
  python agent.py                    # 单轮模式
  python agent.py --question "茅台毛利率是多少？"
  python agent.py --chat             # 多轮对话模式
"""

import os
import argparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

DEFAULT_QUESTION = "贵州茅台和五粮液2023年的毛利率哪家更高？差多少？"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReAct Financial Agent")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--chat", action="store_true", help="多轮对话模式")
    args = parser.parse_args()

    if args.chat:
        from react_chat import chat_main
        chat_main(max_steps=args.max_steps)
    else:
        from react_manual import run_and_print
        run_and_print(args.question, args.max_steps)