#!/usr/bin/env python3
"""日志统计分析脚本 —— v2（优化版本）。

优化点：
1. 流式逐行读取，不把整个文件读进内存 —— 内存占用与文件大小无关（O(1)）。
2. 只遍历一遍，级别计数 / ERROR 计数 / 每小时计数在同一趟循环里完成。
3. 正则在循环外预编译一次，循环内直接复用。
4. 用 collections.Counter，取 Top N 用 most_common(n)（基于堆，O(n log k)），
   避免对全部错误做一次完整排序。
"""
import sys
from collections import Counter

# 正则只编译一次；用一个正则同时捕获 小时 / 级别 / 内容
import re
LINE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} (\d{2}):\d{2}:\d{2} +(DEBUG|INFO|WARN|ERROR) +(.*)$"
)
KNOWN_LEVELS = ("DEBUG", "INFO", "WARN", "ERROR")


def main():
    if len(sys.argv) < 2:
        print("用法: python analyze.py <日志文件路径> [top_n]")
        sys.exit(1)

    path = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    total = 0
    level_counts = Counter()
    error_freq = Counter()
    hourly = Counter()

    # 单次流式遍历，常量内存
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            total += 1
            m = LINE_RE.match(line)
            if not m:
                continue
            hour, level, msg = m.group(1), m.group(2), m.group(3)
            level_counts[level] += 1
            hourly[hour] += 1
            if level == "ERROR":
                error_freq[msg.strip()] += 1

    top_errors = error_freq.most_common(top_n)  # 基于堆，无需全排序

    # ---- 打印报告 ----
    out = []
    out.append("========== 日志统计报告 ==========")
    out.append("文件：%s" % path)
    out.append("总行数：%d" % total)
    out.append("")
    out.append("--- 各级别统计 ---")
    for level in KNOWN_LEVELS:
        c = level_counts[level]
        pct = (c / total * 100) if total else 0
        out.append("%-5s : %7d  (%4.1f%%)" % (level, c, pct))
    out.append("")
    out.append("--- Top %d 错误 ---" % top_n)
    if top_errors:
        for i, (msg, cnt) in enumerate(top_errors, 1):
            out.append("%2d. [%5d 次] %s" % (i, cnt, msg))
    else:
        out.append("（没有 ERROR 级别日志）")
    out.append("")
    out.append("--- 每小时请求量 ---")
    for h in range(24):
        key = "%02d" % h
        out.append("%s 时: %d" % (key, hourly[key]))
    out.append("==================================")
    print("\n".join(out))


if __name__ == "__main__":
    main()
