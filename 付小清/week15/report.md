# 第十五周作业报告：Subagent 并行派发 Agent

## 1. 任务理解

作业要求实现一个能够 **下发 subagent 并行完成多项工作** 的 Agent 系统。核心挑战：

- 主 Agent 如何决策「自己做」还是「派 subagent」
- 如何并行执行多个 subagent 并汇总结果
- 如何量化并行的加速收益

## 2. 实现方案

### 2.1 ReAct 循环引擎（`react_loop.py`）

采用 ReAct（Reason + Act）范式：

1. LLM 输出 `Thought → Action → Action Input`
2. Runner 执行工具，得到 `Observation`
3. 循环直到 `Final Answer`

用 `stop=["Observation:"]` 让 LLM 在 Action Input 后停止，由 runner 补 Observation 续写——这是 ReAct 的经典实现技巧。

主 agent 与 subagent **共用同一个 `ReActLoop` 类**，区别只在 `tools` 字典：

| Agent | 工具集 |
|-------|--------|
| 主 Agent | `web_search`, `dispatch_subagents` |
| Subagent | `web_search` |

### 2.2 并行派发（`agents.py`）

`dispatch_subagents` 工具实现：

```python
# 输入: "子课题1 | 子课题2 | 子课题3"
subtopics = action_input.split("|")

# 为每个子课题创建独立 ReActLoop 实例
with ThreadPoolExecutor(max_workers=len(subtopics)) as pool:
    futures = [pool.submit(subagent.run, topic) for topic in subtopics]
    results = [f.result() for f in as_completed(futures)]
```

**并行收益**：N 个独立子任务的墙钟从 `sum(各子任务时长)` 压到 `max(各子任务时长)`。

`serial=True` 模式用 for 循环顺序执行，作为 A/B 对比基线。

### 2.3 主 Agent 自主路由

通过 `MAIN_SYSTEM` prompt 引导 LLM 决策：

- **2+ 侧面**（调研/分析/概况）→ 必须 `dispatch_subagents`
- **单一事实** → 直接 `web_search`

并附带 worked example 教 LLM 正确的 Action 格式，避免空 action 死循环。

### 2.4 Mock 离线模式

为便于演示与批改，`MOCK_MODE=1` 时：

- LLM 返回预设 ReAct 步骤（主 agent 派发 → subagent 搜索 → 综合报告）
- 搜索模拟 1.2s 延迟，使并行加速可观测

### 2.5 Tavily 联网搜索（可选，当前未使用）

subagent 的 `web_search` 底层对接 Tavily API，需 `TAVILY_API_KEY`。**本作业 MaaS 实测未配置 Tavily**，搜索自动降级为 Mock（见 `README.md`「关于 Tavily API Key」）。Mock 模式下 LLM 仍走真实 API，仅搜索结果为模拟数据；对验证 subagent 并行派发与加速统计无影响。

## 3. 实验结果

### 3.1 Mock 模式对比（`eval_compare.py --mock`）

| 指标 | 并行 (ThreadPool) | 串行 (for 循环) |
|------|-------------------|-----------------|
| dispatch 墙钟 | ≈ max(子任务) ≈ 2.4s | ≈ sum(子任务) ≈ 3.6s |
| 加速比 | **≈ 1.5×** | — |

3 个 subagent 各含 2 步 ReAct（搜索 + 总结），Mock 搜索各 1.2s，并行墙钟 ≈ 2.4s，串行 ≈ 3.6s。

### 3.2 MaaS API 实测（qwen3.7-plus，默认业务空间）

运行环境：
- API：`https://ws-an0d0vqov4zjj1qx.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`
- Model：`qwen3.7-plus`
- Search：Mock（无 TAVILY_API_KEY，搜索用模拟延迟；LLM 推理走真实 API）

| 问题 | 并行总墙钟 | 串行总墙钟 | dispatch 加速 | subagent 数 |
|------|-----------|-----------|--------------|------------|
| 新能源汽车调研 | 95.99s | 154.74s | **2.22×** | 3（并行）/ 2（串行） |
| 咖啡市场调研 | 94.80s | 103.83s | **1.70×** | 2 |
| **平均** | **95.40s** | **129.29s** | **1.96×** | — |

**总墙钟加速**：129.29 / 95.40 ≈ **1.35×**（含主 agent 串行规划/综合段）

**实验 1 并行模式报告摘要**（主 agent 派发 3 个 subagent 后综合）：

> 2024年，中国新能源汽车市场迈入"高渗透、强竞争、政策加码"的新阶段。新能源乘用车零售渗透率稳步突破 **40%**，下半年多月份历史性突破 **50%**；竞争格局呈现"一超多强"，比亚迪全年销量突破 **427万辆**。

**实验 2 并行模式报告摘要**：

> 2023年中国咖啡行业市场规模约 **2654亿-3000亿元**，同比增长超15%；瑞幸门店突破 **20000家** 领跑，星巴克约 **7000-7500家** 坚守高端，库迪快速扩张至近万店。

完整运行日志见 `outputs/maas_output.txt`，结构化数据见 `outputs/maas_eval.json`。

### 3.3 课程参考数据（DeepSeek + Tavily 联网）

| 问题 | 并行墙钟 | 串行墙钟 | dispatch 加速 |
|------|---------|---------|--------------|
| 新能源汽车调研 | 35.88s | 44.78s | 2.32× |
| 咖啡市场调研 | 30.07s | 56.16s | 2.71× |
| **平均** | **32.98s** | **50.47s** | **2.51×** |

### 3.4 结果解读

- **dispatch 加速 1.96×（MaaS 实测）**：独立 subagent 并行，墙钟从 sum 压到 max
- **总墙钟加速 1.35× < dispatch 加速**：主 agent 规划/综合 LLM 调用是串行段（Amdahl 定律）
- **qwen 输出格式**：偶发 `<tool_call>` XML 而非 ReAct 格式，已在 `_parse_tool_call` 中兜底解析

## 4. 关键工程问题与解决

| 问题 | 解决方案 |
|------|----------|
| LLM 不按 ReAct 格式输出 | `_parse` 兜底：无 Action 但有实质文本 → 当作 Final Answer |
| 主 agent 不派发 subagent | MAIN_SYSTEM 加决策原则 + worked example |
| dispatch Observation 过长 | 每个子结果截短到 500 字喂回主 agent |
| 无 API Key 无法演示 | MOCK_MODE 模拟 LLM 与搜索延迟 |
| qwen 输出 `<tool_call>` 格式 | `_parse_tool_call` 解析 XML 转为 ReAct Action |
| Windows 终端 emoji 编码报错 | log() 使用 errors='replace' 兜底 |

## 5. 总结

本作业实现了 **Orchestrator-Workers** 范式的 Agent 系统：

1. 主 Agent 是 ReAct 循环，持 `web_search` + `dispatch_subagents` 两工具
2. 多侧面任务自动拆分为 N 个 subagent，ThreadPoolExecutor 并行执行
3. 完整 trace 捕获（Thought/Action/Observation），可观测每个 agent 的决策过程
4. 并行 vs 串行 A/B 对比量化加速收益，理解 Amdahl 定律对总加速的限制
