"""渐进式 Skill Agent 的 CLI 入口。

运行：
    python -m src.agent
    python -m src.agent --question "给 crazy 做一张英语闪卡" --verbose
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agent_loop import AgentLoop, TurnMetrics
from src.llm_config import create_client
from src.skill_registry import SkillRegistry


def _print_metrics(metrics: TurnMetrics) -> None:
    payload = {
        "catalog_chars": metrics.catalog_chars,
        "loaded_skills": metrics.loaded_skills,
        "loaded_resources": metrics.loaded_resources,
        "written_files": metrics.written_files,
        "tool_calls": metrics.tool_calls,
        "steps": metrics.steps,
        "finish_reason": metrics.finish_reason,
    }
    print("\n[turn metrics]")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="一个支持渐进式加载 SKILL.md 的最小 Agent Loop"
    )
    parser.add_argument("--question", help="单次提问；不传则进入交互模式")
    parser.add_argument("--skills-dir", default="skills")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    workspace = Path.cwd().resolve()
    skills_dir = (workspace / args.skills_dir).resolve()

    registry = SkillRegistry(skills_dir)
    client, config = create_client()
    loop = AgentLoop(
        client=client,
        model=config.model,
        skills=registry,
        workspace=workspace,
        max_steps=args.max_steps,
    )

    print(
        f"Progressive Skills Agent | {config.display_name} / {config.model} | "
        f"{len(registry.list_skills())} skills"
    )
    for warning in registry.warnings:
        print(f"[skill warning] {warning}")

    history: list[dict[str, str]] = []

    if args.question:
        result = loop.run_turn(args.question, history)
        print(result.answer)
        if args.verbose:
            _print_metrics(result.metrics)
        return

    print("命令：/skills 查看索引，/clear 清空对话，/exit 退出")
    while True:
        try:
            user_input = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input == "/exit":
            break
        if user_input == "/clear":
            history.clear()
            print("对话历史已清空。")
            continue
        if user_input == "/skills":
            print(registry.catalog())
            continue

        result = loop.run_turn(user_input, history)
        print(f"Agent：{result.answer}")
        if args.verbose:
            _print_metrics(result.metrics)

        # 只持久化最终问答，不把内部 Skill/Tool 消息带入下一轮。
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": result.answer})


if __name__ == "__main__":
    main()

