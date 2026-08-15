#!/usr/bin/env python3
"""对比 v1 / v2 两个脚本的执行效率与内存占用。

- 用 time.perf_counter() 测多次取最优时间。
- 用子进程 + tracemalloc 不方便跨进程，这里改用 resource 拿子进程峰值 RSS。
- 同时校验两个版本输出的统计结果是否一致（保证优化没改变正确性）。

用法: python benchmark.py <日志文件路径> [重复次数]
"""
import re
import subprocess
import sys
import time
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V1 = os.path.join(BASE, "skill-v1-未优化", "log-stats", "scripts", "analyze.py")
V2 = os.path.join(BASE, "skill-v2-优化后", "log-stats", "scripts", "analyze.py")
PY = sys.executable


def run_once(script, logpath):
    """运行一次，返回 (耗时秒, 峰值内存字节, stdout)。用 /usr/bin/time -l 拿峰值 RSS。"""
    t0 = time.perf_counter()
    proc = subprocess.run(
        ["/usr/bin/time", "-l", PY, script, logpath, "5"],
        capture_output=True, text=True,
    )
    dt = time.perf_counter() - t0
    # macOS /usr/bin/time -l 把资源信息打到 stderr，含 "maximum resident set size"
    peak = None
    m = re.search(r"(\d+)\s+maximum resident set size", proc.stderr)
    if m:
        peak = int(m.group(1))  # macOS 上单位是字节
    return dt, peak, proc.stdout


def bench(script, logpath, repeat):
    best_t = float("inf")
    peak = None
    out = None
    for _ in range(repeat):
        dt, pk, so = run_once(script, logpath)
        best_t = min(best_t, dt)
        if pk is not None:
            peak = pk if peak is None else max(peak, pk)
        out = so
    return best_t, peak, out


def main():
    if len(sys.argv) < 2:
        print("用法: python benchmark.py <日志文件路径> [重复次数]")
        sys.exit(1)
    logpath = sys.argv[1]
    repeat = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    size_mb = os.path.getsize(logpath) / 1024 / 1024
    print("测试文件：%s（%.1f MB）  每版本跑 %d 次取最优\n" % (logpath, size_mb, repeat))

    t1, p1, o1 = bench(V1, logpath, repeat)
    t2, p2, o2 = bench(V2, logpath, repeat)

    same = (o1 == o2)
    print("=== 结果一致性校验 ===")
    print("两个版本输出完全一致：%s\n" % ("是 ✅" if same else "否 ❌"))
    if not same:
        print("[v1 输出]\n%s\n[v2 输出]\n%s\n" % (o1, o2))

    def mb(x):
        return "%.1f MB" % (x / 1024 / 1024) if x else "N/A"

    print("=== 执行效率对比 ===")
    print("%-12s %-12s %-14s" % ("版本", "耗时(s)", "峰值内存"))
    print("%-12s %-12.3f %-14s" % ("v1 未优化", t1, mb(p1)))
    print("%-12s %-12.3f %-14s" % ("v2 优化后", t2, mb(p2)))
    print()
    print("速度提升：%.2fx（v1/v2 = %.3f/%.3f）" % (t1 / t2, t1, t2))
    if p1 and p2:
        print("内存下降：%.2fx（v1/v2 = %s / %s）" % (p1 / p2, mb(p1), mb(p2)))


if __name__ == "__main__":
    main()
