from __future__ import annotations

import argparse
import asyncio
import sys

from .config import Settings
from .factory import build_main_agent


class ChineseArgumentParser(argparse.ArgumentParser):
    """将 argparse 自动生成的帮助信息和错误信息统一为中文。"""

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(2, f"错误：{message}\n")


def _parser() -> argparse.ArgumentParser:
    parser = ChineseArgumentParser(
        description="支持任务委派的 DeepSeek Agent",
        add_help=False,
        usage="%(prog)s [-h] [任务内容 ...]",
    )
    parser._positionals.title = "位置参数"
    parser._optionals.title = "选项"
    parser.add_argument("prompt", nargs="*", help="任务内容；省略后进入交互模式")
    parser.add_argument("-h", "--help", action="help", help="显示帮助信息并退出")
    return parser


async def _run() -> int:
    args = _parser().parse_args()
    try:
        agent = build_main_agent(Settings.from_env())
    except ValueError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2

    if args.prompt:
        print(await agent.run(" ".join(args.prompt)))
        return 0

    print("DeepSeek Agent 已就绪，输入 /exit 退出。")
    while True:
        try:
            prompt = await asyncio.to_thread(input, "你> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if prompt.strip().lower() in {"/exit", "/quit"}:
            return 0
        if prompt.strip():
            try:
                print(f"用户输入问题：{prompt}")
                print(f"Agent> {await agent.run(prompt)}")
            except Exception as exc:
                print(f"错误> {exc}", file=sys.stderr)


def main() -> None:
    print("DeepSeek Agent 正在启动...")
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
