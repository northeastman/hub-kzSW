#!/usr/bin/env python3
"""
Measure skill optimization metrics: token estimate, file size, load scope.
"""

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BEFORE = ROOT / "skill-before" / "code-review"
AFTER = ROOT / "skill-after" / "code-review"


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1 token per 4 chars (English/mixed)."""
    return max(1, len(text) // 4)


def collect_files(skill_dir: Path) -> dict[str, Path]:
    files = {}
    for p in sorted(skill_dir.rglob("*")):
        if p.is_file() and p.suffix in {".md", ".py"}:
            rel = str(p.relative_to(skill_dir))
            files[rel] = p
    return files


def measure_skill(skill_dir: Path, label: str) -> dict:
    files = collect_files(skill_dir)
    contents = {rel: p.read_text(encoding="utf-8") for rel, p in files.items()}

    # Simulate agent load: before = all in SKILL.md; after = SKILL.md only (lazy load reference)
    always_loaded = contents.get("SKILL.md", "")
    lazy_loaded = "".join(
        c for rel, c in contents.items() if rel != "SKILL.md"
    )

    total_chars = sum(len(c) for c in contents.values())
    total_lines = sum(c.count("\n") + 1 for c in contents.values())

    # Simulated parse time (AST walk proxy for script complexity)
    script_path = skill_dir / "scripts" / "lint_check.py"
    parse_ms = 0.0
    if script_path.exists():
        start = time.perf_counter()
        compile(script_path.read_text(encoding="utf-8"), str(script_path), "exec")
        parse_ms = (time.perf_counter() - start) * 1000

    return {
        "label": label,
        "file_count": len(files),
        "files": list(files.keys()),
        "total_chars": total_chars,
        "total_lines": total_lines,
        "total_words": sum(len(c.split()) for c in contents.values()),
        "estimated_tokens_total": estimate_tokens("".join(contents.values())),
        "estimated_tokens_always_loaded": estimate_tokens(always_loaded),
        "estimated_tokens_lazy": estimate_tokens(lazy_loaded),
        "skill_md_chars": len(always_loaded),
        "skill_md_lines": always_loaded.count("\n") + 1,
        "script_parse_ms": round(parse_ms, 3),
    }


def compare(before: dict, after: dict) -> dict:
    def pct(old: float, new: float) -> float:
        return round((old - new) / old * 100, 1) if old else 0.0

    return {
        "token_always_loaded_reduction_pct": pct(
            before["estimated_tokens_always_loaded"],
            after["estimated_tokens_always_loaded"],
        ),
        "token_total_reduction_pct": pct(
            before["estimated_tokens_total"],
            after["estimated_tokens_total"],
        ),
        "chars_reduction_pct": pct(before["total_chars"], after["total_chars"]),
        "lines_reduction_pct": pct(before["total_lines"], after["total_lines"]),
        "always_loaded_tokens_before": before["estimated_tokens_always_loaded"],
        "always_loaded_tokens_after": after["estimated_tokens_always_loaded"],
        "lazy_tokens_after": after["estimated_tokens_lazy"],
        "effective_context_saving": (
            "Agent 默认只加载 SKILL.md；reference.md 按需读取，"
            f"首次上下文节省约 {pct(before['estimated_tokens_total'], after['estimated_tokens_always_loaded'])}%"
        ),
    }


def main() -> None:
    before = measure_skill(BEFORE, "优化前 (skill-before)")
    after = measure_skill(AFTER, "优化后 (skill-after)")
    comparison = compare(before, after)

    # Functional equivalence check: both cover same review dimensions
    dimensions = ["security", "SQL", "correctness", "test", "Critical", "Suggestion"]
    before_text = (BEFORE / "SKILL.md").read_text(encoding="utf-8").lower()
    after_all = "".join(
        p.read_text(encoding="utf-8")
        for p in AFTER.rglob("*")
        if p.suffix in {".md", ".py"}
    ).lower()

    coverage = {
        dim: {"before": dim.lower() in before_text, "after": dim.lower() in after_all}
        for dim in dimensions
    }

    result = {
        "before": before,
        "after": after,
        "comparison": comparison,
        "functional_coverage": coverage,
    }

    out = ROOT / "benchmark" / "results.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 60)
    print("Skill 优化对比测量结果")
    print("=" * 60)
    print(f"\n{'指标':<30} {'优化前':>12} {'优化后':>12} {'变化':>10}")
    print("-" * 60)
    rows = [
        ("SKILL.md 字符数", before["skill_md_chars"], after["skill_md_chars"],
         f"-{comparison['chars_reduction_pct']:.1f}%*"),
        ("始终加载 Token 估算", before["estimated_tokens_always_loaded"],
         after["estimated_tokens_always_loaded"],
         f"-{comparison['token_always_loaded_reduction_pct']:.1f}%"),
        ("全部文件 Token 估算", before["estimated_tokens_total"],
         after["estimated_tokens_total"],
         f"-{comparison['token_total_reduction_pct']:.1f}%"),
        ("总行数", before["total_lines"], after["total_lines"],
         f"-{comparison['lines_reduction_pct']:.1f}%"),
        ("文件数", before["file_count"], after["file_count"], ""),
    ]
    for name, b, a, change in rows:
        print(f"{name:<30} {b:>12} {a:>12} {change:>10}")
    print("\n* SKILL.md 单文件对比；优化后详情见 reference.md（按需加载）")
    print(f"\n按需加载节省: 首次上下文约 {comparison['token_always_loaded_reduction_pct']:.1f}% token")
    print(f"结果已写入: {out}")


if __name__ == "__main__":
    main()
