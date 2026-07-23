# Week12 作业：为 Agent 增加多轮对话能力

在 `react_financial_agent` 的 ReAct 循环基础上，增加**跨轮对话记忆**，支持连续追问。

## 快速开始

```powershell
cd work12
pip install -r requirements.txt

# MaaS 工作空间 API（见 .env.example）
$env:DASHSCOPE_API_KEY = "sk-ws-xxx"
$env:AGENT_BASE_URL = "https://ws-an0d0vqov4zjj1qx.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
$env:AGENT_MODEL = "qwen3.7-plus"

# 一键跑 demo，结果写入 demo_output.txt
python run_demo_maas.py

# 多轮对话 — 交互式
python run_multi_turn.py

# 多轮对话 — 内置三轮追问 demo
python run_multi_turn.py --demo

# 对照：单轮模式（每问独立，无历史）
python run_single_turn.py --demo
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `conversation_agent.py` | 多轮会话核心：`ConversationSession` 跨轮保留 history |
| `run_multi_turn.py` | 主交付：交互式 / demo 多轮对话 |
| `run_single_turn.py` | 对照组：单轮模式，每问 history 为空 |
| `run_demo_maas.py` | MaaS API 一键跑 demo |
| `demo_output.txt` | 实际运行结果 |
| `作业提交说明.md` | 完整作业文档 |

工具集与 ReAct 循环复用 `../react_financial_agent/src/`。

## 典型场景

```
第1轮: 贵州茅台2023年的毛利率是多少？  → 91.96%
第2轮: 那五粮液呢？                    → 理解指「2023毛利率」，查五粮液
第3轮: 两者差多少个百分点？            → 结合前两轮，调用 calculator
```

单轮模式下第2轮「那五粮液呢？」缺少上文，模型无法知道「那」指什么。
