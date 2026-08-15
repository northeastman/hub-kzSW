#!/usr/bin/env python3
"""完整模式：使用真实 LLM + Tavily 搜索运行调研 agent。"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents import run_research  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Subagent 并行调研 Agent")
    parser.add_argument(
        "-q",
        "--question",
        default="2024年中国新能源汽车市场调研：销量规模、主要厂商竞争格局、政策趋势",
    )
    parser.add_argument("--serial", action="store_true", help="subagent 串行执行（对比基线）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print(f"问题: {args.question}")
    print(f"模式: {'串行' if args.serial else '并行'}\n")

    def on_dispatch(info):
        print(f"[派发] {len(info['subtopics'])} 个子 agent: {info['subtopics']}")

    def on_subagent_done(sid, duration, topic):
        print(f"[完成] {sid} ({duration}s) — {topic[:40]}")

    r = run_research(
        args.question,
        on_dispatch=on_dispatch,
        on_subagent_done=on_subagent_done,
        serial=args.serial,
    )

    print(f"\n主 agent 动作: {[s['action'] for s in r['main_trace']]}")
    print(f"subagent 数: {len(r['subagents'])}")
    if r["parallel_stats"]:
        print(f"并行统计: {r['parallel_stats'][-1]}")
    print(f"\n{'=' * 60}\n报告:\n{r['final_answer']}")


if __name__ == "__main__":
    main()
