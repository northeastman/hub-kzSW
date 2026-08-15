# 第十四周作业报告：Cursor Skill 优化对比实验

## 1. 实验背景

Cursor Agent Skill 是以 Markdown 文件形式注入系统提示词的工作流指令。Skill 越长，每次对话消耗的 **Token 越多**，Agent 推理时也更容易被冗余信息干扰。

本实验选取 **Python 代码审查（code-review）** Skill，对比「冗长初稿」与「精简优化版」在 Token 消耗与执行效率上的差异。

---

## 2. Skill 来源与优化策略

### 2.1 优化前（skill-before）

由大模型编写的「教学向」Skill，典型问题：

| 问题 | 示例 |
|------|------|
| 重复解释基础概念 | 大段介绍「什么是 Code Review」「Python 是什么」 |
| 内嵌冗长代码示例 | SQL 注入、文件读写等完整示例占大量 Token |
| 六步流程 + 重复 checklist | 步骤间内容重叠 |
| 单文件承载全部信息 | Agent 每次必须加载 ~1900 Token |

### 2.2 优化后（skill-after）

遵循 Cursor 官方 `create-skill` 指南，采用四项策略：

1. **假设 Agent 已有通用知识** — 删除「Code Review 是什么」等废话
2. **渐进式披露（Progressive Disclosure）** — 核心流程放 `SKILL.md`，细节放 `reference.md` 按需读取
3. **用脚本替代内嵌代码** — `scripts/lint_check.py` 可直接执行，无需把 AST 检查逻辑写进 Skill
4. **表格化 checklist** — 同等信息密度，行数更少

---

## 3. 量化对比结果

运行 `python benchmark/measure_skill.py` 得到：

| 指标 | 优化前 | 优化后 | 变化 |
|------|--------|--------|------|
| SKILL.md 字符数 | 7,499 | 1,287 | **-82.8%** |
| 始终加载 Token（估算） | 1,874 | 321 | **-82.9%** |
| 全部文件 Token（估算） | 1,874 | 1,056 | **-43.6%** |
| 总行数 | 197 | 151 | -23.4% |
| 文件数 | 1 | 3 | +2（按需加载） |

> Token 估算方法：字符数 ÷ 4（业界常用粗估，与 tiktoken 误差约 ±10%）

### 3.1 Token 消耗分析

```
优化前：Agent 启动 → 加载 SKILL.md (1874 tokens) → 开始审查

优化后：Agent 启动 → 加载 SKILL.md (321 tokens) → 开始审查
                  ↘ 需要安全示例时 → 读取 reference.md (+~200 tokens)
                  ↘ 需要自动检查时 → 执行 lint_check.py (0 tokens 注入)
```

**关键收益**：首次上下文节省 **82.9%** Token。在 Skill 较多的 Agent 场景下，多个 Skill 叠加后节省更显著。

### 3.2 执行效率分析

| 维度 | 优化前 | 优化后 |
|------|--------|--------|
| Agent 阅读时间 | 需解析 ~197 行 Markdown | 仅 ~54 行核心指令 |
| 自动化检查 | Agent 自行推理反模式 | 调用 `lint_check.py`，毫秒级 AST 扫描 |
| 检查一致性 | 依赖 LLM 记忆，可能遗漏 | 脚本规则固定，结果可复现 |

**lint_check.py 演示**（对 `benchmark/sample_bad.py`）：

```
ISSUES FOUND:
  - mutable default argument in `process_items`
  - bare `except:` in `process_items`
  - use of eval()
```

脚本在 **<3ms** 内完成编译，运行时扫描几乎瞬时，而让 LLM 逐行审查同样文件需消耗数百 Token 且可能漏检。

### 3.3 功能覆盖验证

| 审查维度 | 优化前 | 优化后 |
|----------|--------|--------|
| 安全性（SQL 注入等） | ✅ | ✅ |
| 正确性 / 边界条件 | ✅ | ✅ |
| 测试覆盖 | ✅ | ✅ |
| Critical / Suggestion 分级 | ✅ | ✅ |

优化后 **功能无损失**，仅去除冗余叙述。

---

## 4. 优化手法总结

| 手法 | Token 效果 | 效率效果 |
|------|-----------|----------|
| 删除 Agent 已知的概念解释 | ⭐⭐⭐ | ⭐ |
| SKILL.md 控制在 500 行以内 | ⭐⭐⭐ | ⭐⭐ |
| 详情拆到 reference.md | ⭐⭐⭐ | ⭐ |
| 可执行脚本替代内嵌代码 | ⭐⭐ | ⭐⭐⭐ |
| 表格 / checklist 替代段落 | ⭐⭐ | ⭐ |

---

## 5. 结论

1. **Token 方面**：优化后 Agent 默认加载上下文从 ~1874 Token 降至 ~321 Token，**节省 82.9%**，多 Skill 场景下收益更大。
2. **效率方面**：自动化脚本将「让 LLM 读代码找问题」变为「脚本秒级扫描 + LLM 聚焦复杂逻辑」，更快且更一致。
3. **质量方面**：精简不等于删减功能；渐进式披露保证细节可查，核心 workflow 更清晰。

**最佳实践**：Skill 应像 API 文档一样——只写 Agent **不知道**的信息，把能自动化的交给脚本。

---

## 6. 附录：文件清单

- `skill-before/code-review/SKILL.md` — 优化前完整 Skill
- `skill-after/code-review/SKILL.md` — 优化后主文件
- `skill-after/code-review/reference.md` — 按需参考
- `skill-after/code-review/scripts/lint_check.py` — 自动化检查脚本
- `benchmark/measure_skill.py` — 测量工具
- `benchmark/results.json` — 原始测量数据
