import os
import argparse
from typing import List, Dict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

DEFAULT_QUESTION = "贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？"

def run_multiturn_repl(mode: str, max_steps: int):
    """
    多轮对话交互式 REPL 循环。
    用户输入问题，Agent 执行 ReAct 循环，然后继续等待下一个问题。
    支持 exit/quit/q 退出。
    """
    if mode == "manual":
        from react_manual import run_and_print
    else:
        from react_function_calling import run_and_print

    messages: List[Dict] = None
    turn_count = 0

    print(f"\n{'='*60}")
    print(f"ReAct Financial Agent — 多轮对话模式")
    print(f"模式: {'手写Prompt解析' if mode == 'manual' else 'Function Calling'}")
    print(f"最大步数: {max_steps}")
    print(f"输入 exit/quit/q 退出")
    print(f"{'='*60}\n")

    while True:
        try:
            if turn_count == 0:
                question = input("请输入问题（或直接回车使用示例问题）: ")
                if not question.strip():
                    question = DEFAULT_QUESTION
            else:
                question = input("\n请输入下一个问题（或输入 exit/quit/q 退出）: ")

            if question.lower() in ("exit", "quit", "q"):
                print("\n对话结束，再见！")
                break

            turn_count += 1
            print(f"\n{'─'*60}")
            print(f"第 {turn_count} 轮")
            print(f"{'─'*60}")

            messages = run_and_print(question, max_steps=max_steps, messages=messages)

        except KeyboardInterrupt:
            print("\n\n对话被中断，再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            print("请重试或输入 exit 退出")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReAct Financial Agent — 多轮对话模式")
    parser.add_argument(
        "--mode", choices=["manual", "fc"], default="manual",
        help="manual=手写Prompt解析版  fc=Function Calling版",
    )
    parser.add_argument("--max_steps", type=int, default=10)
    args = parser.parse_args()

    run_multiturn_repl(args.mode, args.max_steps)