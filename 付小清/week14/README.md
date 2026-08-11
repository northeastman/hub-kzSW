# 第十四周作业：Skill 优化对比实验

## 作业目标

1. 编写（或获取）一个 Cursor Agent Skill
2. 从 **Token 消耗** 和 **执行效率** 角度优化
3. 量化对比优化前后效果

## 目录结构

```
work14/
├── README.md                          # 本文件
├── report.md                          # 实验报告（优化思路 + 对比结论）
├── skill-before/code-review/SKILL.md  # 优化前：冗长版 Python 代码审查 Skill
├── skill-after/code-review/
│   ├── SKILL.md                       # 优化后：精简主文件
│   ├── reference.md                   # 按需加载的详细参考
│   └── scripts/lint_check.py          # 可执行脚本（替代内嵌长代码）
└── benchmark/
    ├── measure_skill.py               # 测量脚本
    ├── sample_bad.py                  # lint 演示用例
    └── results.json                   # 测量结果（运行脚本后生成）
```

## 快速运行

```bash
# 1. 测量 Token / 体量对比
python benchmark/measure_skill.py

# 2. 演示优化后 skill 附带的自动化检查脚本
python skill-after/code-review/scripts/lint_check.py benchmark/sample_bad.py
```

## Skill 说明

本作业选用 **Python 代码审查（code-review）** Skill：

| 版本 | 特点 |
|------|------|
| 优化前 | 单文件 ~200 行，重复解释概念、多个工具选项、冗长示例 |
| 优化后 | 主文件 ~50 行 + 按需 reference + 可执行 lint 脚本 |

优化策略遵循 Cursor 官方 [create-skill](https://cursor.com/docs) 指南：
- 精简 SKILL.md，假设 Agent 已有通用知识
- 渐进式披露（Progressive Disclosure）
- 用脚本替代上下文中的长代码块

## 参考

- 项目内 `self_evolving_agent/` 目录有完整的 Skill 自进化演示
- Cursor Skill 规范：`~/.cursor/skills-cursor/create-skill/SKILL.md`
