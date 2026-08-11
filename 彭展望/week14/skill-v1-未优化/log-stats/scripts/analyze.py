#!/usr/bin/env python3
"""日志统计分析脚本 —— v1（未优化版本）。

实现思路很直白：把整个文件读进内存，然后针对不同的统计目标分别遍历若干遍。
功能正确，但在大文件上执行效率和内存占用都不理想。
"""
import re
import sys


def main():
    if len(sys.argv) < 2:
        print("用法: python analyze.py <日志文件路径> [top_n]")
        sys.exit(1)

    path = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    # 一次性把整个文件读进内存（大文件时内存占用高）
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    total = len(lines)

    # ---- 第 1 遍：统计各级别数量 ----
    level_counts = {"DEBUG": 0, "INFO": 0, "WARN": 0, "ERROR": 0}
    for line in lines:
        # 正则在循环内部反复编译（低效）
        m = re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} +(\w+) +(.*)$", line)
        if m:
            level = m.group(1)
            if level in level_counts:
                level_counts[level] += 1

    # ---- 第 2 遍：收集所有 ERROR 内容，再统计 Top N ----
    error_messages = []
    for line in lines:
        m = re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} +(\w+) +(.*)$", line)
        if m and m.group(1) == "ERROR":
            error_messages.append(m.group(2).strip())

    # 用手动 dict 计数
    error_freq = {}
    for msg in error_messages:
        if msg in error_freq:
            error_freq[msg] = error_freq[msg] + 1
        else:
            error_freq[msg] = 1

    # 全量排序后取前 top_n（即便只要前 5 条也把全部排一遍）
    sorted_errors = sorted(error_freq.items(), key=lambda kv: kv[1], reverse=True)
    top_errors = sorted_errors[:top_n]

    # ---- 第 3 遍：统计每小时请求量 ----
    hourly = {}
    for h in range(24):
        hourly["%02d" % h] = 0
    for line in lines:
        m = re.match(r"^\d{4}-\d{2}-\d{2} (\d{2}):\d{2}:\d{2} +(\w+) +(.*)$", line)
        if m:
            hour = m.group(1)
            if hour in hourly:
                hourly[hour] = hourly[hour] + 1

    # ---- 打印报告 ----
    print("========== 日志统计报告 ==========")
    print("文件：%s" % path)
    print("总行数：%d" % total)
    print()
    print("--- 各级别统计 ---")
    for level in ["DEBUG", "INFO", "WARN", "ERROR"]:
        c = level_counts[level]
        pct = (c / total * 100) if total else 0
        print("%-5s : %7d  (%4.1f%%)" % (level, c, pct))
    print()
    print("--- Top %d 错误 ---" % top_n)
    if top_errors:
        for i, (msg, cnt) in enumerate(top_errors, 1):
            print("%2d. [%5d 次] %s" % (i, cnt, msg))
    else:
        print("（没有 ERROR 级别日志）")
    print()
    print("--- 每小时请求量 ---")
    for h in range(24):
        key = "%02d" % h
        print("%s 时: %d" % (key, hourly[key]))
    print("==================================")


if __name__ == "__main__":
    main()
