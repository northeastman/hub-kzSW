# ARCHITECTURE.md — 旅行规划 Subagent 并行收集系统

## 1. 项目定位

**场景**：用户提一个旅行规划问题，主 agent 自主决定是否派发多个 subagent 并行联网收集交通、住宿、景点、美食等侧面，聚合成可执行行程方案。与 `market_research_subagents` 同构，换域演示「动态 Orchestrator–Workers」。

**核心设计**：
- 主 agent 是 ReAct 循环，**2 个工具**：`web_search`（单次搜索）和 `dispatch_subagents`（派发 N 个 subagent 并行收集）。主 agent **根据 query 自主路由**。
- subagent 也是 ReAct（只有 `web_search`），用 `ThreadPoolExecutor` 并行执行。
- 可视化：左侧拓扑 + 右侧 ReAct 过程流 + 下方行程方案。

**范式**：动态 Orchestrator-Workers —— 拓扑在运行时生长。

## 2. 整体流水线

```
用户问题（行程规划）
   ↓
主 agent ReAct（工具: web_search + dispatch_subagents）
   ├─ 单一事实 → 直接 web_search → Final Answer
   └─ 多维度规划 → dispatch_subagents("交通|住宿|景点|美食")
                       ↓
              ┌─ subagent1 ReAct(web_search) ─┐
              ├─ subagent2 ReAct(web_search) ─┤ 并行(ThreadPool)
              ├─ subagent3 ReAct(web_search) ─┤
              └─ subagent4 ReAct(web_search) ─┘
                       ↓ 汇总（含并行加速统计）
              主 agent 综合成行程方案 → Final Answer
```

脚本：`tavily_search.py` → `react_loop.py` → `agents.py` → `serve.py` / `eval_compare.py`。

## 3. 与市场调研项目的对照

| 维度 | market_research_subagents | travel_planner_subagents（本项目） |
|------|---------------------------|-----------------------------------|
| 场景 | 市场/竞品/行业调研 | 旅行行程规划 |
| 主入口函数 | `run_research` | `run_plan`（兼 `run_research` 别名） |
| 默认端口 | 8002 | 8003 |
| 派发侧面示例 | 销量 / 竞争 / 政策 | 交通 / 住宿 / 景点 / 美食 |
| 输出物 | 调研报告 | 行程方案（分天/维度 + 预算粗估） |
| 骨架 | 相同：ReActLoop + ThreadPool + SSE | 相同 |

换域只需改：`MAIN_SYSTEM`、默认问题、UI 文案；编排与并行逻辑原样复用。

## 4. 各环节要点

### 4.1 联网搜索：Tavily（urllib，零 SDK）
失败返回错误字符串，ReAct 兜底继续。

### 4.2 通用 ReAct（react_loop.py）
主 / 子共用 `ReActLoop`，区别只在 `tools` 字典。`stop=["Observation:"]` 截断；无 Action 但有文本时兜底为 Final Answer。

### 4.3 主 agent 自主决策（agents.py）
- ≥2 个侧面（行程/攻略/XX游）→ **必须** `dispatch_subagents`
- 单一事实 → 直接 `web_search`

### 4.4 并行执行
`ThreadPoolExecutor`；`serial=True` 供 A/B 对比。量化：`wall_clock` vs `serial_sum`，`speedup = serial_sum / wall_clock`。

### 4.5 可视化
`static/index.html` + `static/viz/topology.js`：dispatch 时动态加节点，步骤流可按节点过滤。

## 5. 教学点

1. **LLM 自主路由**：不是硬编码「永远派 4 个」，而是主 agent 根据问题拆子课题。
2. **并行价值**：N 个独立收集任务墙钟从 sum → ≈max。
3. **Amdahl**：主 agent 规划/综合仍串行，端到端加速 < dispatch 加速。
4. **Agent = ReActLoop + tools**：换场景 = 换 prompt + 换默认题，骨架不动。
