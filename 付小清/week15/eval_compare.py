#!/usr/bin/env python3
"""Parallel vs Serial 量化对比。"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents import run_research  # noqa: E402

logging.basicConfig(level=logging.WARNING)

EVAL_QUESTIONS = [
    "2024年中国新能源汽车市场调研：销量规模、主要厂商竞争格局、政策趋势",
    "中国咖啡市场调研：市场规模、主要品牌、消费趋势",
    "中国扫地机器人市场调研：市场规模、主要品牌、技术趋势",
]


def run_one(question: str, serial: bool) -> dict:
    t0 = time.time()
    r = run_research(question, serial=serial)
    wall = round(time.time() - t0, 2)
    ps = r["parallel_stats"][-1] if r["parallel_stats"] else {}
    return {
        "wall": wall,
        "n_subagents": ps.get("n_subagents", 0),
        "dispatch_wall": ps.get("wall_clock", 0),
        "serial_sum": ps.get("serial_sum", 0),
        "speedup": ps.get("speedup", 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mock", action="store_true", help="使用 Mock 模式（无需 API Key）")
    args = parser.parse_args()

    if args.mock:
        os.environ["MOCK_MODE"] = "1"

    qs = EVAL_QUESTIONS[: args.limit] if args.limit else EVAL_QUESTIONS
    results = []

    for i, q in enumerate(qs):
        print(f"[{i + 1}/{len(qs)}] {q[:40]}...")
        p = run_one(q, serial=False)
        s = run_one(q, serial=True)
        results.append({"question": q, "parallel": p, "serial": s})
        print(
            f"  并行 {p['wall']}s vs 串行 {s['wall']}s "
            f"(subagent {p['n_subagents']}, 加速 {p['speedup']}×)"
        )

    avg_p = sum(r["parallel"]["wall"] for r in results) / len(results)
    avg_s = sum(r["serial"]["wall"] for r in results) / len(results)
    avg_spd = sum(r["parallel"]["speedup"] for r in results) / len(results)

    print(f"\n{'=' * 60}")
    print(f"Parallel vs Serial（{len(results)} 题）")
    print(f"{'=' * 60}")
    print(f"平均墙钟 — 并行: {avg_p:.2f}s | 串行: {avg_s:.2f}s")
    print(f"平均 dispatch 加速: {avg_spd:.2f}×")

    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "eval_compare.json"
    out_path.write_text(
        json.dumps(
            {
                "summary": {
                    "avg_parallel_s": round(avg_p, 2),
                    "avg_serial_s": round(avg_s, 2),
                    "avg_speedup": round(avg_spd, 2),
                },
                "details": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
