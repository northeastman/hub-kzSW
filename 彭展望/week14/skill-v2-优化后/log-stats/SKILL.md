---
name: log-stats
description: >-
  统计日志文件的级别分布、Top N 错误、每小时请求量，输出文本报告。
  Use when the user wants to analyze a .log/.txt file, count log levels,
  find the most frequent errors, or see hourly traffic. 例如
  "分析下 app.log"、"这日志有多少 ERROR"、"哪个小时请求最多"。
---

# Log Stats 日志统计分析

对日志文件做统计并打印文本报告：总行数、各级别数量与占比、Top N 错误、每小时请求量。

## 执行流程

1. 从用户话语中取日志文件路径；缺失就问。
2. 运行脚本（第二参数为可选的 Top N，默认 5）：
   ```bash
   python skill-v2-优化后/log-stats/scripts/analyze.py <日志路径> [top_n]
   ```
3. 把脚本打印的报告展示给用户，并用一两句话点出关键情况（错误是否偏多、流量高峰等）。

## 说明

- 日志格式、报告样例、边界情况见 `reference.md`（仅在需要核对格式细节时再读取）。
- 无法解析的行计入总行数但不计入级别统计；文件很大时也无需担心内存。
