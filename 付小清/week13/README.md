# Week13 作业：渐进式加载执行 Skills 的 Harness

在 `../skills/` 目录下的 Cursor Agent Skills 基础上，实现一套 **渐进式披露（Progressive Disclosure）** 的 harness：先只加载元数据索引，匹配后再读正文，最后按需加载 references/scripts 并执行。

## 快速开始

```powershell
cd work13

# 离线 Demo（无需 API Key）— 展示四阶段加载 + crazy 闪卡脚本执行
python run_demo.py --all

# 单条 query
python run_demo.py -q "给我做张 crazy 词的闪卡"

# 交互式
python run_harness.py

# LLM 完整模式（需 API Key）
$env:DASHSCOPE_API_KEY = "sk-xxx"
python run_harness.py --llm -q "帮我做 meticulous 的闪卡"

# MaaS 一键跑 demo → demo_output.txt
$env:DASHSCOPE_API_KEY = "sk-ws-xxx"
$env:AGENT_BASE_URL = "https://ws-an0d0vqov4zjj1qx.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
$env:AGENT_MODEL = "qwen3.7-plus"
python run_demo_maas.py
```

## 四阶段流水线

```
Stage 0 INDEX     scan_skills_dir → 仅 name + description（YAML frontmatter）
Stage 1 MATCH     规则 / LLM 选 skill（仍不读正文）
Stage 2 BODY      加载 SKILL.md 正文
Stage 3 RESOURCE  按用户意图加载 references/*.md、scripts/*
Stage 4 EXECUTE   调用脚本或 LLM 完成任务
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `skill_registry.py` | Stage 0：扫描 skills、解析 frontmatter |
| `skill_matcher.py` | Stage 1：规则 / LLM 匹配 |
| `skill_loader.py` | Stage 2–3：正文与资源渐进加载 |
| `skill_executor.py` | Stage 4：flash-card / baoyu-diagram 执行器 |
| `harness.py` | 主编排器 `ProgressiveSkillHarness` |
| `run_demo.py` | 离线演示 |
| `run_harness.py` | 交互式 CLI |
| `run_demo_maas.py` | MaaS API 批量 demo |
| `作业提交说明.md` | 完整作业文档 |

Skills 复用 `../skills/`（`flash-card`、`baoyu-diagram`）。

## 设计要点

| 要点 | 说明 |
|------|------|
| 为何渐进加载 | 多个 skill 正文可达数千 token；索引层仅数百字符，匹配前不污染上下文 |
| INDEX vs BODY | 模拟 Cursor `<available_skills>` 注入 vs `Read SKILL.md` |
| RESOURCE 按需 | baoyu-diagram 的 `references/architecture.md` 仅在用户要「架构图」时加载 |
| EXECUTE 分离 | flash-card 走确定性脚本；复杂 skill 走 LLM + 已加载指令 |

## 典型输出

```
[Stage 0 INDEX] 扫描 .../skills → 2 个 skill，索引共 446 字符
[Stage 1 MATCH] rule → flash-card (confidence=0.60, 规则命中: 闪卡)
[Stage 2 BODY] 加载 SKILL.md → 1842 字符
[Stage 3 RESOURCE] 预加载: scripts/make_flashcard.py
[Stage 4 EXECUTE] make_flashcard.py → output/crazy.html
```

## 实际运行（MaaS API）

已使用默认业务空间（qwen3.7-plus）跑通三组实验，详见 `demo_output.txt` 与 `作业提交说明.md` 第十一节。

| 实验 | Query | 输出 |
|------|-------|------|
| A | crazy 闪卡（规则匹配） | `output/crazy.html` |
| B | thrill flash card（LLM 匹配） | `output/thrill.html` |
| C | 四阶段架构图（渐进 reference + LLM） | `output/demo-diagram.svg` |
