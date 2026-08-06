---
name: calculator
description: 当需要进行精确算术计算（加减乘除、带括号的表达式）时使用
run: main.py
---

# 计算器能力

当用户需要精确计算时，调用本能力的脚本 run_skill_script("calculator", args)。
args 为表达式的分词数组，例如 ["2", "+", "3", "*", "4"]。
脚本会返回计算结果。将结果自然地告诉用户，不要暴露脚本细节。
