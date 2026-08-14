"""命令行入口：下发 subagent 并行完成多项工作，并量化并行加速比。

用法：
  python main.py                       # 用默认示例任务
  python main.py "你的复杂任务描述"      # 自定义任务
  python main.py --serial "任务"        # 额外跑一遍串行版做真实对照

输出：控制台打印并行时间线 + 最终报告；完整结果存到 outputs/last_run.json。
"""
import sys
import json
import time
import os

from orchestrator import Orchestrator

DEFAULT_TASK = "对「2024年中国新能源汽车行业」做一份市场调研：涵盖市场规模与增长、主要厂商竞争格局、政策与补贴导向、以及出海与未来趋势。"

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def draw_timeline(results):
    """用 ASCII 甘特图展示各 subagent 的并行重叠情况。"""
    if not results:
        return
    base = min(r.started_at for r in results)
    span = max(r.finished_at for r in results) - base
    if span <= 0:
        return
    width = 40
    print("\n=== 并行时间线（甘特图，越重叠说明并行度越高）===")
    for r in results:
        s = int((r.started_at - base) / span * width)
        e = max(s + 1, int((r.finished_at - base) / span * width))
        bar = " " * s + "█" * (e - s)
        print(f"子{r.id} |{bar:<{width}}| {r.elapsed:4.1f}s  {r.title[:18]}")
    print(f"     0s{'':<{width-4}}{span:4.1f}s")


def main():
    args = [a for a in sys.argv[1:]]
    run_serial = "--serial" in args
    args = [a for a in args if a != "--serial"]
    task = args[0] if args else DEFAULT_TASK

    orch = Orchestrator(max_workers=5, verbose=True)
    result = orch.run(task)

    draw_timeline([_R(x) for x in result["results"]])

    t = result["timing"]
    print("\n" + "=" * 56)
    print("并行 vs 串行 效果对比")
    print("=" * 56)
    print(f"子任务数量           : {len(result['subtasks'])}")
    print(f"并行墙钟耗时         : {t['parallel_wall_sec']} s  ← 实际花费")
    print(f"串行理论耗时(各子之和): {t['serial_sum_sec']} s  ← 若一个个跑")
    print(f"并行加速比           : {t['speedup']}x")

    # 可选：真跑一遍串行做实证对照
    if run_serial:
        print("\n[对照] 正在真实串行重跑各子任务...")
        from subagent import SubTask
        sts = [SubTask(**s) for s in result["subtasks"]]
        t0 = time.perf_counter()
        _ = orch.dispatch_serial(sts)
        real_serial = time.perf_counter() - t0
        print(f"[对照] 真实串行墙钟耗时: {real_serial:.1f}s  "
              f"(实测加速比 {real_serial / t['parallel_wall_sec']:.2f}x)")

    print("\n" + "=" * 56)
    print("最终交付物（主 Agent 汇总）")
    print("=" * 56)
    print(result["final_report"])

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "last_run.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[已保存] 完整结果 -> {out_path}")


class _R:
    """把 dict 结果还原成带属性的对象，供 draw_timeline 使用。"""
    def __init__(self, d):
        self.__dict__.update(d)


if __name__ == "__main__":
    main()
