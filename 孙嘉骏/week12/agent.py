"""
统一入口：切换手写版 / Function Calling 版 ReAct Agent

使用方式：
  python agent.py
  python agent.py --mode manual   --question "茅台2023年毛利率是多少？"
  python agent.py --mode fc       --question "五粮液近一年股价涨跌幅？"
  python agent.py --mode manual   --question "..." --max_steps 8

环境变量：
  DASHSCOPE_API_KEY  必填
  AGENT_MODEL        默认 qwen-max，可换 deepseek-v3 等
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
    parser.add_argument("--question",  type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=10)
    args = parser.parse_args()

    if args.mode == "manual":
        from react_manual import run_and_print
    else:
        from react_function_calling import run_and_print


    # 为agent流程增加多轮对话能力
    if args.question:
        run_and_print(args.question, args.max_steps)
    else:
        mode = "手写Prompt解析版" if args.mode == "manual" else "Function Calling版"
        print(f"Agent 系统（{mode}）")
        print("输入 'exit' 退出\n")
        while True:
            try:
                q = input("问题：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() == "exit":
                break
            run_and_print(args.question, args.max_steps)
