"""
统一入口：切换手写版 / Function Calling 版 ReAct Agent

使用方式：
  python agent.py
  python agent.py --mode manual   --question "茅台2023年毛利率是多少？"
  python agent.py --mode fc       --question "五粮液近一年股价涨跌幅？"
  python agent.py --mode manual   --interactive          # 多轮对话
  python agent.py --mode fc       -i                     # 多轮对话（简写）

环境变量：
  DASHSCOPE_API_KEY  手写版必填（也用于 RAG Embedding）
  DEEPSEEK_API_KEY   Function Calling 版必填（见 react_function_calling.py）
  AGENT_MODEL        默认随实现而异
"""

import os
import argparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

DEFAULT_QUESTION = "贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReAct Financial Agent")
    parser.add_argument(
        "--mode", choices=["manual", "fc"], default="manual",
        help="manual=手写Prompt解析版  fc=Function Calling版",
    )
    parser.add_argument("--question",  default=None)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="多轮对话模式：可连续追问（如「那茅台呢？」）",
    )
    args = parser.parse_args()

    if args.mode == "manual":
        from react_manual import run_and_print, run_interactive
    else:
        from react_function_calling import run_and_print, run_interactive

    if args.interactive:
        run_interactive(args.max_steps)
    elif args.question:
        run_and_print(args.question, args.max_steps)
    else:
        # 无参数时默认进入多轮对话，方便体验追问能力
        run_interactive(args.max_steps)
