#!/usr/bin/env python3
"""
离线 Demo — 展示渐进式加载四阶段（无需 API Key）

用法:
  python run_demo.py
  python run_demo.py --query "给我做张 crazy 词的闪卡"
  python run_demo.py --query "画一个 skills harness 架构图"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from harness import ProgressiveSkillHarness


DEMO_QUERIES = [
    "给我做张 crazy 词的闪卡",
    "画一个 agent skills 加载流程的架构图",
]


def main():
    parser = argparse.ArgumentParser(description="渐进式 Skill Harness 离线 Demo")
    parser.add_argument("--query", "-q", help="单条用户请求")
    parser.add_argument("--all", action="store_true", help="跑内置两条 demo query")
    args = parser.parse_args()

    harness = ProgressiveSkillHarness()
    queries = DEMO_QUERIES if args.all or not args.query else [args.query]

    for q in queries:
        print("\n")
        report = harness.run(q, prefer_llm_match=False, use_llm_exec=False, verbose=True)
        print(harness.format_report(report))


if __name__ == "__main__":
    main()
