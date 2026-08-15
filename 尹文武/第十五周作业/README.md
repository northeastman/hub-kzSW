# DeepSeek 多 Agent

一个基于 Python 的轻量 Agent：主 Agent 可把独立任务通过 `delegate_task` 工具分发给隔离的子 Agent；父、子 Agent 都能调用 `web_search` 联网搜索。模型固定默认为 `deepseek-v4-flash`，通过 DeepSeek 的 OpenAI 兼容 Chat Completions 接口调用。

## 设计

- `Agent`：通用的异步 tool-calling 循环。
- `DelegateTaskTool`：参照 Hermes 的委派语义，把完整子 Agent 封装为普通函数工具。
- 子 Agent 使用全新消息历史，只接收 `goal` 和 `context`，完成后仅把最终摘要交给父 Agent。
- 支持单任务以及最多 3 个任务的并行批处理；结果始终按输入顺序返回。
- 叶子子 Agent 只有 `web_search`，不能再次调用 `delegate_task`，避免递归失控。
- `WebSearchTool`：通过 DuckDuckGo 搜索，返回标题、URL 和摘要，无需额外搜索 API Key。

## 安装

需要 Python 3.11+。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
```

PowerShell 不会自动加载 `.env`，运行前至少设置：

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."
```

## 使用

单次任务：

```powershell
delegating-agent "搜索并比较今天三个重要的 AI 新闻，可以分给子 Agent"
```

交互模式：

```powershell
python -m delegating_agent
```

主 Agent 会自行决定是否调用 `delegate_task`。也可以在提示中明确要求拆成并行任务。批量委派的工具参数示意：

```json
{
  "tasks": [
    {"goal": "研究方案 A", "context": "关注成本，给出来源 URL"},
    {"goal": "研究方案 B", "context": "关注性能，给出来源 URL"}
  ],
  "max_iterations": 10
}
```

## 配置

| 环境变量 | 默认值 | 含义 |
|---|---:|---|
| `DEEPSEEK_API_KEY` | 必填 | DeepSeek API Key |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | OpenAI 兼容接口地址 |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` | 父、子 Agent 使用的模型 |
| `AGENT_MAX_ITERATIONS` | `20` | 主 Agent 最大循环轮数 |
| `DELEGATION_MAX_ITERATIONS` | `12` | 每个子 Agent 最大循环轮数 |
| `DELEGATION_MAX_CONCURRENT_CHILDREN` | `3` | 单批最大子 Agent 数 |

## 测试

```powershell
pytest
```

测试使用假模型客户端，不会消耗 API 额度。真实联网搜索和 DeepSeek 端到端调用需要本机网络及有效 Key。
