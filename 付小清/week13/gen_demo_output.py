"""生成 demo_output.txt（UTF-8）"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from harness import ProgressiveSkillHarness

QUERIES = [
    "给我做张 crazy 词的闪卡",
    "画一个 agent skills 加载流程的架构图",
]


def main():
    harness = ProgressiveSkillHarness()
    parts = [
        "=" * 70,
        "Week13 渐进式 Skill Harness — 离线 Demo 运行记录",
        "=" * 70,
        "",
    ]
    for i, q in enumerate(QUERIES, 1):
        parts.append(f"\n### 实验 {chr(64 + i)}: {q}\n")
        report = harness.run(q, prefer_llm_match=False, use_llm_exec=False, verbose=True)
        parts.append(harness.format_report(report))

    text = "\n".join(parts)
    out = Path(__file__).parent / "demo_output.txt"
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out} ({len(text)} chars)")


if __name__ == "__main__":
    main()
