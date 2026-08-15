# 优化版自进化 Agent

本目录是 `note/week14 自进化agent/self_evolving_agent` 的**工程优化版**，
在保留原版教学透明度的前提下，通过五个独立方案降低 token 消耗、提升运行效率。

## 与原版的关系

| 维度 | 原版（教学） | 优化版（本目录） |
|------|------------|----------------|
| Skill 注入 | 每次全量 | 按需路由 + 教学开关 |
| SkillManager IO | 每次读盘 | mtime 缓存 |
| 评估执行 | 串行 | asyncio 并发 |
| Reviewer 提示词 | policies.md 全量 | 仅注入相关小节 |
| 基线评估 | 每次重跑 | 结果缓存复用 |

**核心设计原则**：所有优化都通过 `teaching_mode` 开关切换，`True` 退化为原版行为。
教学模式下学生能看到完整的 prompt 与流程，优化模式下学生能看到工程优化空间。

## 数据文件复用

本目录**不包含** `data/`、`skills/`、`index.html` 等运行时数据。
首次运行前，请从原项目复制：

```powershell
$src = "c:\Users\chieh\PycharmProjects\nlp_class\nlp_chieh\note\week14 自进化agent\self_evolving_agent"
$dst = "c:\Users\chieh\PycharmProjects\nlp_class\nlp_chieh\钱杰\week14\self_evolving_agent_optimized"
Copy-Item "$src\data"     $dst\ -Recurse -Force
Copy-Item "$src\skills"   $dst\ -Recurse -Force
Copy-Item "$src\index.html" $dst\ -Force
```

或者建立符号链接（更省空间）。

## 五个优化方案详解

### 方案 1：Skill 按需注入（省 50%+ Agent token）

**文件**：[src/agent.py](file:///c:/Users/chieh/PycharmProjects/nlp_class/nlp_chieh/钱杰/week14/self_evolving_agent_optimized/src/agent.py)

**原版问题**：每次应答都把**所有 Skill** 全文塞进 system prompt。Skill 数量增长后，
单次 token 从 ~1.5K 涨到 ~3K+，但每题实际只用到 1-2 个 Skill。

**优化实现**：
- 在 `_build_system_prompt` 前先调用 `_select_relevant_skills(question)`
- 基于 `SKILL_ROUTING_RULES` 关键词表路由（顺序敏感：digital_goods 优先于 refund）
- 始终兜底注入 `refund` Skill，避免边界场景漏判
- 多个 Skill 关键词同时命中时全部注入（如"VIP 退款"同时命中 vip_benefits + refund）
- 没有任何命中时回退到全量（保证安全）

**开关**：`CustomerServiceAgent(teaching_mode=True)` → 全量注入（原版行为）

**统计**：`agent.routing_summary()` 返回节省字符数与节省率

```python
agent = CustomerServiceAgent(sm, teaching_mode=False)
# ... 跑完实验后
print(agent.routing_summary())
# {'mode': 'optimized (partial injection)', 'saved_chars': 45000, 'char_save_rate': 0.62}
```

**风险**：关键词路由可能漏判边界 case（如"我是VIP买的T恤能退吗"需 refund + vip）。
教学上需说明这是"工程优化"，演示时可选关闭。

---

### 方案 2：SkillManager mtime 缓存（省磁盘 IO）

**文件**：[src/skill_manager.py](file:///c:/Users/chieh/PycharmProjects/nlp_class/nlp_chieh/钱杰/week14/self_evolving_agent_optimized/src/skill_manager.py)

**原版问题**：每题都 `load_all()` 遍历磁盘读 SKILL.md。Skill 在块内不变，重复 IO 浪费。

**优化实现**：
- `_cache: dict[str, str]` 保存 skill_name → content
- `_mtimes: dict[str, float]` 保存 skill_name → 文件 mtime
- `load_all()` / `get()` 命中条件：skill_name 在缓存 且 mtime 未变
- `create()` / `patch()` 自动调用 `invalidate(skill_name)` 失效该条目

**对外接口完全兼容原版**，教学上可演示"工程优化"的透明性。

**统计**：`sm.cache_stats()` 返回命中率

```python
sm = SkillManager("skills/")
# ... 跑完实验后
print(sm.cache_stats())
# {'hits': 180, 'misses': 20, 'hit_rate': 0.9}
```

---

### 方案 3：评估并发化（省 60%+ 墙钟时间）

**文件**：[src/async_eval.py](file:///c:/Users/chieh/PycharmProjects/nlp_class/nlp_chieh/钱杰/week14/self_evolving_agent_optimized/src/async_eval.py)

**原版问题**：baseline/final/probe 都是串行跑 60/30 题，60 题 eval 约 3 分钟。

**优化实现**：
- `run_eval_concurrent_silent`：静默并发，等所有题跑完返回
- `run_eval_concurrent_streaming`：流式并发，每题完成立即回调（供 SSE 用）
- 用 `asyncio.Semaphore(max_concurrency)` 控制并发数（默认 5）
- `Agent.answer()` 是同步阻塞，用 `asyncio.to_thread` 包装成异步
- Agent 无状态（每题独立），天然支持并发

**开关**：`teaching_mode=True` 或 `max_concurrency=1` → 退化为串行

**使用方式**：
```python
import asyncio
from async_eval import run_eval_concurrent_silent

result = asyncio.run(run_eval_concurrent_silent(
    agent, evaluator, question_ids=list(range(1, 61)),
    max_concurrency=5, teaching_mode=False,
))
```

**注意**：
- DeepSeek API 默认支持并发，建议并发数 5（避免限流）
- SSE 流式输出顺序会乱，回调里带 qid 让客户端按 id 排序
- 演示脚本（demo_script）的逐题应答**保持串行**，教学上需要按顺序展示失败累积

---

### 方案 5：Reviewer 提示词压缩（省 60% Reviewer token）

**文件**：[src/background_reviewer.py](file:///c:/Users/chieh/PycharmProjects/nlp_class/nlp_chieh/钱杰/week14/self_evolving_agent_optimized/src/background_reviewer.py)

**原版问题**：把 `policies.md` 全文（~3K token）+ 当前所有 Skill 全文都塞进 prompt，
单次 ~6K token。Reviewer 调用 ≤8 次，累计消耗明显。

**优化实现**：
- 新增 `_select_relevant_policies(failed_turns)` 方法
- 根据 `POLICY_SECTION_RULES` 关键词表匹配失败样本涉及的章节
- 永远包含 `ALWAYS_INCLUDE_SECTIONS`（基础退款规则，作为判定基准）
- 按 markdown 标题（## / ###）切分 policies.md，只保留命中章节
- 没有任何命中时回退到全量（保证安全）

**开关**：`BackgroundReviewer(teaching_mode=True)` → 全量注入 policies

**统计**：`reviewer.compression_summary()` 返回平均压缩率

```python
reviewer = BackgroundReviewer("data/policies.md", sm, teaching_mode=False)
# ... 跑完实验后
print(reviewer.compression_summary())
# {'mode': 'optimized (sectioned policies)', 'avg_save_rate': 0.65}
```

---

### 方案 6：基线评估结果缓存（重跑实验省 60 次调用）

**文件**：[src/baseline_cache.py](file:///c:/Users/chieh/PycharmProjects/nlp_class/nlp_chieh/钱杰/week14/self_evolving_agent_optimized/src/baseline_cache.py)

**原版问题**：每次 `python src/demo_runner.py` 都重新跑 60 题基线。但基线 Skills
不变时（temperature=0），基线答案也基本不变。

**优化实现**：
- 缓存 key = Skills 内容 SHA256 指纹 + eval_set.json 的 mtime+size
- 命中条件：缓存文件存在 + Skills 指纹匹配 + eval_set 指纹匹配
- 命中时直接返回，跳过 60 次 LLM 调用
- 缓存文件：`outputs/baseline_cache.json`
- 重置实验时自动清空（`restore_from_original(clear_cache=True)`）

**开关**：`--no-baseline-cache` 命令行参数 / `use_baseline_cache=False`

**注意**：
- DeepSeek 即便 temperature=0 也有微小波动，缓存是"近似复现"
- 教学场景下建议禁用缓存（`--teaching` 自动禁用），保证学生看到实时结果
- 最终评估**不缓存**（Skills 已变）

---

## 运行方式

### 命令行（推荐首次运行）

```powershell
cd c:\Users\chieh\PycharmProjects\nlp_class\nlp_chieh\钱杰\week14\self_evolving_agent_optimized
$env:DEEPSEEK_API_KEY = "your_key"

# 默认优化模式
python src/demo_runner.py

# 教学模式（原版行为，全量注入 + 串行 + 不缓存）
python src/demo_runner.py --teaching

# 自定义并发数
python src/demo_runner.py --concurrency 10

# 禁用基线缓存
python src/demo_runner.py --no-baseline-cache

# 输出优化统计
python src/demo_runner.py --stats
```

### Web UI（步进式演示）

```powershell
cd c:\Users\chieh\PycharmProjects\nlp_class\nlp_chieh\钱杰\week14\self_evolving_agent_optimized
uvicorn serve:app --host 0.0.0.0 --port 8000 --reload
```

新增接口：
- `GET  /mode` — 查询当前模式
- `POST /mode` — 切换模式（需重置后才生效）
- `GET  /stats` — 查询优化统计（路由命中率、压缩率等）

## 预期收益

基于原版数据估算（Skill 数 2→8 时）：

| 指标 | 原版 | 优化版 | 节省 |
|------|------|--------|------|
| Agent 单次 token | ~1.5-3K | ~400-600 | 50%+ |
| Reviewer 单次 token | ~6K | ~2K | 60% |
| 60 题 eval 墙钟时间 | ~3 分钟 | ~40 秒 | 78% |
| 重跑实验基线调用 | 60 次 | 0 次（命中缓存） | 100% |

整体单次实验成本从 ~0.2 元降到 ~0.08 元，墙钟时间从 ~15 分钟降到 ~6 分钟。

## 文件结构

```
self_evolving_agent_optimized/
├── src/
│   ├── __init__.py
│   ├── skill_manager.py        # 方案 2：mtime 缓存
│   ├── agent.py                # 方案 1：按需注入
│   ├── background_reviewer.py  # 方案 5：提示词压缩
│   ├── evaluator.py            # 评估器（保持原版）
│   ├── async_eval.py           # 方案 3：并发评估
│   ├── baseline_cache.py       # 方案 6：基线缓存
│   └── demo_runner.py          # 命令行入口
├── serve.py                    # Web UI 入口
├── requirements.txt
├── OPTIMIZATION.md             # 本文档
└── (需从原项目复制)
    ├── data/                   # policies.md, eval_set.json, demo_script.json
    ├── skills/                 # 初始 Skill 文件
    └── index.html              # Web UI 前端
```

## 教学 vs 工程的权衡

项目定位是教学 Demo，几个设计是**刻意保留**的：
- 每题全量 Skill 注入 → 让学生看到 Skill 增长对 prompt 的影响
- 串行 eval → SSE 流式输出的可视化教学价值
- 30 题固定 Probe → 跨版本可比性

优化版通过 `teaching_mode` 开关切换两种模式，既保留教学透明度又能展示工程能力。
建议课堂演示时：
1. 先用 `--teaching` 跑一遍，让学生看到完整流程
2. 再用默认优化模式跑一遍，对比 `/stats` 输出的节省率
3. 讨论"工程优化"与"教学透明"的取舍
