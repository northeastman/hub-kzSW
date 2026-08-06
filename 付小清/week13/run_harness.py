#!/usr/bin/env python3
"""
交互式 Skill Harness

命令:
  quit / exit  — 退出
  index       — 仅展示 Stage 0 索引
  load <name> — 手动加载指定 skill 正文
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from harness import ProgressiveSkillHarness
from skill_loader import ProgressiveSkillLoader
from skill_matcher import get_skill_by_name


def cmd_index(harness: ProgressiveSkillHarness) -> None:
    from skill_registry import build_index_prompt

    skills = harness.index()
    print(f"\n[Stage 0] 共 {len(skills)} 个 skill，索引 {sum(s.index_chars for s in skills)} 字符\n")
    print(build_index_prompt(skills))


def cmd_load(harness: ProgressiveSkillHarness, name: str) -> None:
    meta = get_skill_by_name(harness.index(), name)
    if not meta:
        print(f"未找到 skill: {name}")
        return
    loader = ProgressiveSkillLoader(meta)
    body = loader.ensure_body()
    print(f"\n[Stage 2] {name} 正文 ({len(body)} chars):\n")
    print(body[:1500])
    if len(body) > 1500:
        print("\n…(截断)")


def interactive(harness: ProgressiveSkillHarness, use_llm: bool) -> None:
    print("渐进式 Skill Harness — 输入用户请求，或 index / load <name> / quit")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        low = line.lower()
        if low in ("quit", "exit", "q"):
            break
        if low == "index":
            cmd_index(harness)
            continue
        if low.startswith("load "):
            cmd_load(harness, line[5:].strip())
            continue

        report = harness.run(
            line,
            prefer_llm_match=use_llm,
            use_llm_exec=use_llm,
            verbose=True,
        )
        print(harness.format_report(report))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="匹配与执行均使用 LLM（需 API Key）")
    parser.add_argument("-q", "--query", help="单条非交互请求")
    args = parser.parse_args()

    harness = ProgressiveSkillHarness()

    if args.query:
        report = harness.run(
            args.query,
            prefer_llm_match=args.llm,
            use_llm_exec=args.llm,
            verbose=True,
        )
        print(harness.format_report(report))
        return

    interactive(harness, use_llm=args.llm)


if __name__ == "__main__":
    main()
