"""
MaaS API 一键 Demo — 输出写入 demo_output.txt

环境变量:
  DASHSCOPE_API_KEY
  AGENT_BASE_URL  (可选，默认 MaaS 工作空间)
  AGENT_MODEL     (可选，默认 qwen3.7-plus)
"""

from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

MAAS_BASE_URL = "https://ws-an0d0vqov4zjj1qx.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.7-plus"


def main():
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        print("错误：请设置 DASHSCOPE_API_KEY", file=sys.stderr)
        sys.exit(1)

    os.environ.setdefault("AGENT_BASE_URL", MAAS_BASE_URL)
    os.environ.setdefault("AGENT_MODEL", DEFAULT_MODEL)

    from harness import ProgressiveSkillHarness

    harness = ProgressiveSkillHarness()
    buf = io.StringIO()

    with redirect_stdout(buf):
        print("=" * 70)
        print("Week13 渐进式 Skill Harness 运行记录")
        print(f"API Host : {os.environ.get('AGENT_BASE_URL')}")
        print(f"Model    : {os.environ.get('AGENT_MODEL')}")
        print("=" * 70)

        # A: 离线渐进加载 + flash-card 执行（复用已有 crazy.json）
        print("\n\n### 实验 A：离线模式 — crazy 闪卡（规则匹配 + 脚本执行）\n")
        r1 = harness.run(
            "给我做张 crazy 词的闪卡",
            prefer_llm_match=False,
            use_llm_exec=False,
            verbose=True,
        )
        print(harness.format_report(r1))

        # B: LLM 匹配 thrill 闪卡（需生成 JSON）
        print("\n\n### 实验 B：LLM 模式 — thrill 闪卡（LLM 匹配可选 + 生成 JSON）\n")
        r2 = harness.run(
            "帮我生成 thrill 单词的 flash card",
            prefer_llm_match=True,
            use_llm_exec=True,
            verbose=True,
        )
        print(harness.format_report(r2))

        # C: baoyu-diagram 渐进加载 reference
        print("\n\n### 实验 C：架构图 — 渐进加载 reference + LLM SVG\n")
        r3 = harness.run(
            "画一个 skills harness 四阶段加载的架构图",
            prefer_llm_match=False,
            use_llm_exec=True,
            verbose=True,
        )
        print(harness.format_report(r3))

    out_path = Path(__file__).parent / "demo_output.txt"
    text = buf.getvalue()
    out_path.write_text(text, encoding="utf-8")
    print(f"已写入 {out_path} ({len(text)} 字符)")
    # 同时打印摘要到 stdout
    print(text[:4000])
    if len(text) > 4000:
        print("\n…(完整日志见 demo_output.txt)")


if __name__ == "__main__":
    main()
