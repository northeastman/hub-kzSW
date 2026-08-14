#!/usr/bin/env python3
"""离线演示：无需 API Key，展示 subagent 并行派发与加速效果。"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
os.environ.setdefault("MOCK_MODE", "1")

from agents import run_research  # noqa: E402


def print_trace(label, trace):
    print(f"\n--- {label} ---")
    for step in trace:
        action = step.get("action", "")
        inp = (step.get("action_input") or "")[:50]
        obs = (step.get("observation") or "")[:60]
        print(f"  [{step['idx']}] {action}({inp})")
        if obs:
            print(f"       → {obs}...")


def main():
    question = "2024年中国新能源汽车市场调研：销量规模、主要厂商竞争格局、政策趋势"
    print("=" * 60)
    print("第十五周作业 Demo — Subagent 并行派发（Mock 模式）")
    print("=" * 60)
    print(f"\n问题: {question}\n")

    print(">>> 并行模式 (ThreadPoolExecutor)")
    r = run_research(question, serial=False)
    print_trace("主 Agent", r["main_trace"])
    for sid, info in r["subagents"].items():
        print_trace(f"Subagent {sid} — {info['subtopic'][:30]}", info["trace"])

    stats = r["parallel_stats"][-1] if r["parallel_stats"] else {}
    print(f"\n派发 subagent 数: {stats.get('n_subagents', 0)}")
    print(f"并行墙钟: {stats.get('wall_clock')}s | 串行等价: {stats.get('serial_sum')}s")
    print(f"dispatch 加速: {stats.get('speedup')}×")
    print(f"\n最终报告:\n{r['final_answer'][:400]}")

    print("\n>>> 串行模式对比 (for 循环)")
    r2 = run_research(question, serial=True)
    stats2 = r2["parallel_stats"][-1] if r2["parallel_stats"] else {}
    print(f"串行 dispatch 墙钟: {stats2.get('wall_clock')}s")
    print(f"\n结论: 并行把 N 个独立子任务墙钟从 sum 压到 ≈max")


if __name__ == "__main__":
    main()
