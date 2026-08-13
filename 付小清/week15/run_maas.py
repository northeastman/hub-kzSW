#!/usr/bin/env python3
"""MaaS API 一键运行 — 实时输出进度，结果写入 outputs/"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

MAAS_BASE_URL = "https://ws-an0d0vqov4zjj1qx.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.7-plus"

EVAL_QUESTIONS = [
    "2024年中国新能源汽车市场调研：销量规模、主要厂商竞争格局、政策趋势",
    "中国咖啡市场调研：市场规模、主要品牌、消费趋势",
]

OUT_DIR = Path(__file__).parent / "outputs"
LOG_FILE = OUT_DIR / "maas_output.txt"
PROGRESS_FILE = OUT_DIR / "progress.txt"


def log(msg: str = ""):
    """同时打印到终端和日志文件，立即 flush。"""
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        sys.stdout.write(line)
        sys.stdout.flush()
    except UnicodeEncodeError:
        safe = line.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
            sys.stdout.encoding or "utf-8", errors="replace"
        )
        sys.stdout.write(safe)
        sys.stdout.flush()
    OUT_DIR.mkdir(exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)


def _safe_text(text: str, limit: int = 50) -> str:
    return text.encode("ascii", errors="replace").decode("ascii")[:limit]


def progress(msg: str):
    """写入 progress.txt，方便在 IDE 里查看当前进度。"""
    OUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%H:%M:%S")
    text = f"[{ts}] {msg}\n"
    PROGRESS_FILE.write_text(text, encoding="utf-8")
    log(msg)


def run_one(question: str, serial: bool, label: str) -> dict:
    from agents import run_research

    mode = "串行" if serial else "并行"
    progress(f"{label} — 开始 ({mode})")

    t0 = time.time()

    def on_main_step(step):
        action = step.get("action") or "(思考中)"
        inp = _safe_text(step.get("action_input") or "", 50)
        progress(f"{label} — 主 agent 步骤 {step['idx']}: {action}({inp})")

    def on_dispatch(info):
        log(f"  [派发] {len(info['subtopics'])} 个 subagent")
        for t in info["subtopics"]:
            log(f"    - {t[:60]}")
        progress(f"{label} — 已派发 {len(info['subtopics'])} 个 subagent")

    def on_subagent_step(sid, step):
        if step.get("observation") is None and step.get("action"):
            progress(f"{label} — {sid} 步骤 {step['idx']}: {step['action']}")

    def on_subagent_done(sid, duration, topic):
        log(f"  [完成] {sid} ({duration}s) — {topic[:50]}")
        progress(f"{label} — {sid} 完成 ({duration}s)")

    r = run_research(
        question,
        on_main_step=on_main_step,
        on_subagent_step=on_subagent_step,
        on_subagent_done=on_subagent_done,
        on_dispatch=on_dispatch,
        serial=serial,
    )
    wall = round(time.time() - t0, 2)
    ps = r["parallel_stats"][-1] if r["parallel_stats"] else {}
    result = {
        "question": question,
        "wall": wall,
        "main_actions": [s["action"] for s in r["main_trace"]],
        "n_subagents": ps.get("n_subagents", 0),
        "dispatch_wall": ps.get("wall_clock", 0),
        "serial_sum": ps.get("serial_sum", 0),
        "speedup": ps.get("speedup", 0),
        "final_answer": r["final_answer"],
        "dispatched": len(r["dispatches"]) > 0,
    }
    progress(f"{label} — 完成 ({mode}) 总墙钟 {wall}s, subagent {result['n_subagents']} 个")
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题")
    args = parser.parse_args()

    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        print("错误：请设置 DASHSCOPE_API_KEY", file=sys.stderr)
        sys.exit(1)

    os.environ.setdefault("AGENT_BASE_URL", MAAS_BASE_URL)
    os.environ.setdefault("AGENT_MODEL", DEFAULT_MODEL)

    OUT_DIR.mkdir(exist_ok=True)
    LOG_FILE.write_text("", encoding="utf-8")
    PROGRESS_FILE.write_text("启动中...\n", encoding="utf-8")

    qs = EVAL_QUESTIONS[: args.limit] if args.limit else EVAL_QUESTIONS

    log("=" * 70)
    log("Week15 Subagent 并行 Agent — MaaS API 运行记录")
    log(f"Workspace : 默认业务空间 (ws-an0d0vqov4zjj1qx)")
    log(f"API Host  : {os.environ.get('AGENT_BASE_URL')}")
    log(f"Model     : {os.environ.get('AGENT_MODEL')}")
    log(f"Search    : {'Tavily' if os.getenv('TAVILY_API_KEY') else 'Mock（无 TAVILY_API_KEY）'}")
    log(f"题目数    : {len(qs)}")
    log("=" * 70)
    progress(f"共 {len(qs)} 题，每题跑并行+串行两种模式")

    results = []
    total_steps = len(qs) * 2
    step_no = 0

    for i, q in enumerate(qs):
        label = f"实验 {i + 1}/{len(qs)}"
        log(f"\n### {label}：{q}\n")

        step_no += 1
        log(f"--- [{step_no}/{total_steps}] 并行模式 ---")
        p = run_one(q, serial=False, label=label)
        log(f"  总墙钟: {p['wall']}s | dispatch: {p['dispatch_wall']}s | 加速: {p['speedup']}×")
        log(f"  主 agent 动作: {p['main_actions']}")
        log(f"  subagent 数: {p['n_subagents']}")
        log(f"\n  报告摘要:\n  {p['final_answer'][:500]}...\n")

        step_no += 1
        log(f"--- [{step_no}/{total_steps}] 串行模式（对比基线）---")
        s = run_one(q, serial=True, label=label)
        log(f"  总墙钟: {s['wall']}s | dispatch: {s['dispatch_wall']}s")
        results.append({"question": q, "parallel": p, "serial": s})

    avg_p = sum(r["parallel"]["wall"] for r in results) / len(results)
    avg_s = sum(r["serial"]["wall"] for r in results) / len(results)
    spd_list = [r["parallel"]["speedup"] for r in results if r["parallel"]["speedup"] > 0]
    avg_spd = sum(spd_list) / len(spd_list) if spd_list else 0

    log(f"\n{'=' * 70}")
    log("Parallel vs Serial 汇总")
    log(f"{'=' * 70}")
    log(f"{'指标':<20} {'并行':<15} {'串行':<15}")
    log(f"{'平均总墙钟(s)':<20} {avg_p:<15.2f} {avg_s:<15.2f}")
    log(f"{'平均 dispatch 加速':<20} {avg_spd:<15.2f}×")
    log(f"\n结论: subagent 并行把独立子任务墙钟从 sum 压到 max，平均加速 {avg_spd:.2f}×")

    summary = {
        "api_host": os.environ.get("AGENT_BASE_URL"),
        "model": os.environ.get("AGENT_MODEL"),
        "summary": {
            "avg_parallel_s": round(avg_p, 2),
            "avg_serial_s": round(avg_s, 2),
            "avg_speedup": round(avg_spd, 2),
        },
        "details": results,
    }
    (OUT_DIR / "maas_eval.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    progress(f"全部完成！平均并行 {avg_p:.1f}s vs 串行 {avg_s:.1f}s，加速 {avg_spd:.2f}×")
    log(f"\n结果已保存: {LOG_FILE}")
    log(f"JSON 已保存: {OUT_DIR / 'maas_eval.json'}")
    log(f"进度文件: {PROGRESS_FILE}")


if __name__ == "__main__":
    main()
