# USAGE_GUIDE.md — 代码调用与测试指南

## 1. 环境准备

### 1.1 依赖安装
```bash
cd travel_planner_subagents
pip install -r requirements.txt
```

### 1.2 API Key
```bash
# Windows PowerShell
$env:DEEPSEEK_API_KEY="sk-xxx"
$env:TAVILY_API_KEY="tvly-xxx"
```

## 2. 各步骤流程

### Step 1：CLI 跑一次规划
```bash
python src/agents.py
```
内置自测「成都三日游」，打印主 agent 动作序列 + subagent 数 + 并行统计。

或直接调 `run_plan`：
```python
import sys; sys.path.insert(0, "src")
from agents import run_plan
r = run_plan("成都三日游规划：交通到达、住宿区域、必去景点、特色美食")
print(r["final_answer"])
print("并行:", r["parallel_stats"])
```

### Step 2：HTTP 服务 + 可视化
```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8003
# 浏览器开 http://localhost:8003
```
- `GET /health` → `{tavily, llm}` 就绪状态
- `POST /query {question}` → SSE 流：
  `start` → `main_step` → `dispatch` → `subagent_step` → `subagent_done` → `final` → `done`

示例问题：
- 多维度（会派发）：`杭州两日游攻略：交通、景点、美食`
- 单事实（不派发）：`成都大熊猫基地门票多少钱`

### Step 3：Parallel vs Serial 对比
```bash
python src/eval_compare.py --limit 2
```
输出墙钟/加速对比表，并写入 `outputs/eval_compare.json`。

## 3. 作为模块调用
```python
import sys; sys.path.insert(0, "src")
from agents import run_plan

def on_main(step): print(f"[main] {step['action']}")
def on_sub(sid, step): print(f"[{sid}] {step['action']}")
def on_dispatch(info): print(f"派发: {info['subtopics']}")
def on_done(sid, dur, topic): print(f"[{sid}] done {dur}s")

r = run_plan(
    "西安周末游：交通、兵马俑与古城路线、美食",
    on_main_step=on_main,
    on_subagent_step=on_sub,
    on_dispatch=on_dispatch,
    on_subagent_done=on_done,
)
```

## 4. 与市场调研项目对照
| | 市场调研 | 本项目（旅行规划） |
|--|---------|------------------|
| 目录 | `market_research_subagents` | `travel_planner_subagents` |
| 端口 | 8002 | 8003 |
| 入口 | `run_research` | `run_plan` |
| 骨架 | 相同 | 相同 |
