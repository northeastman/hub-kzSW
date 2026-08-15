# 第十五周作业：可下发 Subagent 的并行 Agent

## 作业目标

自己实现一个 **主 Agent**，能够：
1. 根据任务复杂度自主决策（简单问题直接搜索，多侧面问题派发 subagent）
2. 通过 `dispatch_subagents` 工具 **并行** 下发多个 subagent 完成子任务
3. 汇总各 subagent 结果，输出综合报告
4. 量化对比并行 vs 串行的加速效果

## 架构概览

```
用户问题
   ↓
主 Agent（ReAct 循环）
   工具: web_search | dispatch_subagents
   ↓
   ├─ 单一事实 → web_search → Final Answer
   └─ 多侧面任务 → dispatch_subagents("子题1|子题2|子题3")
                        ↓
              ┌─ subagent₁ ReAct(web_search) ─┐
              ├─ subagent₂ ReAct(web_search) ─┤ ThreadPoolExecutor 并行
              └─ subagent₃ ReAct(web_search) ─┘
                        ↓ 汇总 + 加速统计
              主 Agent 综合 → Final Answer
```

## 目录结构

```
work15/
├── README.md              # 本文件
├── report.md              # 实现说明与实验结论
├── requirements.txt
├── run_demo.py            # 离线演示（无需 API Key）
├── run_agent.py           # 完整模式（需 LLM + Tavily）
├── run_maas.py            # MaaS API 一键运行（实时进度）
├── eval_compare.py        # 并行 vs 串行对比实验
├── outputs/               # 实验结果（maas_output.txt / progress.txt）
└── src/
    ├── react_loop.py      # 通用 ReAct 引擎（主/sub 共用）
    ├── agents.py          # 主 Agent + dispatch_subagents 并行派发
    ├── llm_client.py      # LLM 客户端（支持 Mock 模式）
    └── search.py          # Tavily 搜索 + Mock 搜索
```

## 快速运行

### 1. 离线演示（推荐，无需 API Key）

```powershell
cd work15
python run_demo.py
```

使用 `MOCK_MODE=1` 模拟 LLM 与搜索，演示 subagent 并行派发与加速效果。

### 2. 并行 vs 串行对比（Mock 模式）

```powershell
python eval_compare.py --mock --limit 2
```

### 3. 完整模式（需 API Key）

```powershell
pip install -r requirements.txt
$env:DEEPSEEK_API_KEY = "sk-xxx"
$env:TAVILY_API_KEY = "tvly-xxx"
python run_agent.py -q "2024年中国咖啡市场调研：市场规模、主要品牌、消费趋势"
```

### 4. MaaS API 运行（默认业务空间 qwen3.7-plus）

```powershell
pip install -r requirements.txt
$env:DASHSCOPE_API_KEY = "sk-ws-xxx"
$env:AGENT_BASE_URL = "https://ws-an0d0vqov4zjj1qx.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
$env:AGENT_MODEL = "qwen3.7-plus"
$env:PYTHONIOENCODING = "utf-8"
python -u run_maas.py
```

运行时可打开 `outputs/progress.txt` 查看实时进度。

> **当前作业提交**：仅使用 MaaS LLM + Mock 搜索，**未配置 Tavily Key**。详见下文「关于 Tavily API Key」。

### 5. 真实 API 对比实验

```powershell
python eval_compare.py --limit 2
```

## 核心实现要点

| 模块 | 职责 |
|------|------|
| `ReActLoop` | 通用 ReAct 循环，主 agent 与 subagent 共用，区别仅在 `tools` 字典 |
| `dispatch_subagents` | 解析 `\|` 分隔的子课题，ThreadPoolExecutor 并行执行 |
| `serial=True` | 退化为 for 循环串行，作为 A/B 对比基线 |
| `parallel_stats` | 记录 wall_clock、serial_sum、speedup，量化并行收益 |

## 关于 Tavily API Key（可选，当前未使用）

**Tavily API Key** 是 [Tavily](https://tavily.com) 搜索服务的访问密钥，用于 subagent 的 `web_search` 工具联网检索：

```
subagent → web_search → Tavily API → 返回网页摘要/来源 → 喂给 LLM
```

### 与 MaaS Key 的分工

| 环境变量 | 服务 | 作用 | 本作业当前状态 |
|----------|------|------|----------------|
| `DASHSCOPE_API_KEY` | 阿里云 MaaS | LLM 推理（qwen3.7-plus） | ✅ 已使用 |
| `TAVILY_API_KEY` | Tavily | 联网搜索（subagent 收集资料） | ❌ **未配置，走 Mock** |

未设置 `TAVILY_API_KEY` 时，`src/search.py` 自动降级为 Mock 搜索（模拟延迟 + 占位结果），**不影响 subagent 并行派发机制的演示与实验**。

### 如何获取（需要真实联网搜索时）

1. 打开 [https://tavily.com](https://tavily.com) 注册
2. Dashboard → API Keys → 创建 key（格式如 `tvly-xxxxxxxx`）
3. 免费档一般有每月一定搜索额度（以官网为准）

### 启用方式（当前作业不需要）

```powershell
$env:TAVILY_API_KEY = "tvly-你的key"
# 配合 MaaS 一起使用
$env:DASHSCOPE_API_KEY = "sk-ws-xxx"
$env:AGENT_BASE_URL = "https://ws-an0d0vqov4zjj1qx.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
$env:AGENT_MODEL = "qwen3.7-plus"
python -u run_maas.py
```

设置后日志中 `Search` 会显示 `Tavily` 而非 `Mock`，报告将基于真实网页内容生成。

## 设计决策

- **Orchestrator-Workers 拓扑**：主 agent 决定派几个、派什么，运行时动态生长
- **工具集定义能力**：主 agent 有 2 工具，subagent 只有 web_search
- **Amdahl 定律**：总墙钟加速 < dispatch 加速，因主 agent 规划/综合段不并行

## 参考

课程示例：`../week15 graph与LLM/market_research_subagents/`
