"""
Parallel vs Serial 量化对比（凸显 reviewer subagent 并行优势）

教学重点：
  同一个项目目录，主 agent 派发的 reviewer 分别用「并行(ThreadPool)」和
  「串行(for 循环)」两种方式执行，对比 wall-clock，量化并行加速。

  并行的意义不是少做事，而是把 N 个独立子任务（审查 N 个文件）的墙钟时间
  从 sum 压到 max。本项目的 dispatch_reviewers 用 ThreadPoolExecutor 实现并行，
  serial=True 时退化为串行（eval 基线）。

使用方式：
  python eval_compare.py            # 默认全部目录，parallel vs serial
  python eval_compare.py --limit 1  # 快速版
"""
import os, sys, time, json, logging, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent
# 测试目录集：示例待审项目 + 本项目 src（都有多个代码文件，能派发多个 reviewer）
EVAL_DIRS = [
    str(BASE / "sample_project"),
    str(BASE / "src"),
]


def run_one(project_dir, serial):
    """跑一次审查，返回并行/串行统计。
    serial=True/False 控制 reviewer 执行方式。"""
    import agents
    t0 = time.time()
    r = agents.run_review(project_dir, serial=serial)
    wall = time.time() - t0
    ps = r["parallel_stats"][-1] if r["parallel_stats"] else None
    return {
        "wall": round(wall, 2),
        "n_reviewers": ps["n_reviewers"] if ps else 0,
        "dispatch_wall": ps["wall_clock"] if ps else 0,
        "serial_sum": ps["serial_sum"] if ps else 0,
        "speedup": ps["speedup"] if ps else 0,
        "dispatched": len(r["dispatches"]) > 0,
    }


def main():
    parser = argparse.ArgumentParser(description="parallel vs serial 对比")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    dirs = EVAL_DIRS[:args.limit] if args.limit else EVAL_DIRS

    results = []
    for i, d in enumerate(dirs):
        name = os.path.basename(d.rstrip("/\\"))
        logger.warning(f"[{i+1}/{len(dirs)}] 审查 {name}")
        p = run_one(d, serial=False)
        s = run_one(d, serial=True)
        results.append({"project": name, "parallel": p, "serial": s})
        print(f"  {name:<20} 并行 {p['wall']}s vs 串行 {s['wall']}s "
              f"(reviewer {p['n_reviewers']}, 加速 {p['speedup']}×)")

    avg_p = sum(r["parallel"]["wall"] for r in results) / len(results)
    avg_s = sum(r["serial"]["wall"] for r in results) / len(results)
    avg_spd = sum(r["parallel"]["speedup"] for r in results) / len(results)

    print(f"\n{'='*60}\nParallel vs Serial 对比（{len(results)} 个项目）\n{'='*60}")
    print(f"{'指标':<16} {'并行(ThreadPool)':<18} {'串行(for循环)':<18}")
    print(f"{'平均墙钟(s)':<16} {avg_p:<18.2f} {avg_s:<18.2f}")
    print(f"{'平均加速':<16} {avg_spd:<18.2f}× {'—':<18}")
    print(f"\n结论：reviewer 并行把 N 个独立文件审查的墙钟从 sum 压到 ≈max，"
          f"平均加速 {avg_spd:.2f}×")

    out = {"summary": {"avg_parallel_s": round(avg_p, 2),
                        "avg_serial_s": round(avg_s, 2),
                        "avg_speedup": round(avg_spd, 2)},
           "details": results}
    out_dir = BASE / "outputs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "eval_compare.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 outputs/eval_compare.json")


if __name__ == "__main__":
    main()
