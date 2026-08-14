# RESUME_GUIDE.md — 项目表述要点

## 一句话
实现「旅行规划主 agent + 动态并行 subagent」：主 ReAct 自主决定是否拆分交通/住宿/景点/美食等侧面，ThreadPool 并行收集后综合成行程方案，并用 SSE + 拓扑可视化展示并行加速。

## 技术关键词
- Orchestrator–Workers（动态拓扑）
- ReAct（Thought / Action / Observation）
- ThreadPoolExecutor 并行 vs 串行 A/B
- Tavily 联网搜索 + DeepSeek LLM
- FastAPI SSE 实时过程流

## 与市场调研 demo 的差异（面试可讲）
- **同构换域**：骨架（ReActLoop、dispatch、SSE、拓扑）不变，只换系统提示与任务侧面。
- **说明能力**：理解「Agent = 循环引擎 + 工具集 + 领域 prompt」，而非绑死某一业务。

## 可演示路径
1. CLI：`python src/agents.py`
2. UI：`uvicorn src.serve:app --port 8003` → 看派发节点与并行加速数字
3. Eval：`python src/eval_compare.py --limit 2` → 量化 speedup
