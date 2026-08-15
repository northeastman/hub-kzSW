#!/usr/bin/env python3
"""生成一个用于压测的模拟日志文件。

用法: python gen_log.py <输出路径> <行数>
"""
import random
import sys

LEVELS = ["DEBUG"] * 33 + ["INFO"] * 50 + ["WARN"] * 12 + ["ERROR"] * 5  # 大致比例

INFO_MSGS = [
    "User login succeeded for user_id=%d",
    "Request handled in %dms",
    "Cache hit for key=session_%d",
]
DEBUG_MSGS = [
    "Entering function handle_request()",
    "Loaded config value=%d",
    "Heartbeat tick %d",
]
WARN_MSGS = [
    "Cache miss for key=session_%d",
    "Slow query took %dms",
    "Retrying request attempt %d",
]
# 只用少数几种 ERROR，让 Top N 有明显集中度
ERROR_MSGS = [
    "Database connection timeout after 30s",
    "Failed to parse request body",
    "Redis connection refused",
    "Null pointer in handler",
    "Rate limit exceeded",
    "Upstream service 503",
]


def main():
    if len(sys.argv) < 3:
        print("用法: python gen_log.py <输出路径> <行数>")
        sys.exit(1)
    path = sys.argv[1]
    n = int(sys.argv[2])
    rnd = random.Random(42)  # 固定种子，保证可复现

    with open(path, "w", encoding="utf-8") as f:
        for _ in range(n):
            hh = rnd.randint(0, 23)
            mm = rnd.randint(0, 59)
            ss = rnd.randint(0, 59)
            level = rnd.choice(LEVELS)
            if level == "INFO":
                msg = rnd.choice(INFO_MSGS) % rnd.randint(1, 9999)
            elif level == "DEBUG":
                tmpl = rnd.choice(DEBUG_MSGS)
                msg = tmpl % rnd.randint(1, 9999) if "%d" in tmpl else tmpl
            elif level == "WARN":
                msg = rnd.choice(WARN_MSGS) % rnd.randint(1, 9999)
            else:  # ERROR：固定文案，便于归并计数
                msg = rnd.choice(ERROR_MSGS)
            f.write("2026-08-06 %02d:%02d:%02d %-5s %s\n" % (hh, mm, ss, level, msg))

    print("已生成 %d 行 -> %s" % (n, path))


if __name__ == "__main__":
    main()
