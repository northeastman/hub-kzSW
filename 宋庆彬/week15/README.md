# Parallel Agent

一个仅使用 Python 标准库的轻量并行 Agent 编排库。Supervisor 通过兼容 OpenAI Chat Completions 的模型拆分任务，并行执行子任务，再汇总成功与失败结果。需要 Python 3.11 或更高版本。

## Python async 调用

```python
import asyncio

from parallel_agent import AgentConfig, SupervisorAgent
from parallel_agent.openai_compat import OpenAICompatibleClient


async def main():
    client = OpenAICompatibleClient(
        model="your-model",
        base_url="http://localhost:8000/v1",
        api_key="your-api-key",
    )
    config = AgentConfig(
        max_agents=4,
        worker_timeout=60.0,
        retries=1,
    )
    report = await SupervisorAgent(client, config).run("调研并综合目标主题")
    print(report.final_answer)
    return report


report = asyncio.run(main())
```

`AgentConfig` 的参数含义：

- `max_agents`：子 Agent 数量的有效范围是 `1..4`，默认为 `4`；即使规划模型返回更多任务，也最多执行前 4 个。
- `worker_timeout`：每次 worker 尝试的超时时间（秒），默认为 `60.0`。
- `retries`：worker 失败或超时后的重试次数，默认为 `1`，因此最多尝试 2 次。

## 返回结果

`SupervisorAgent.run()` 返回 `RunReport`：

- `task`：原始任务。
- `final_answer`：模型的综合答案；若聚合调用失败或返回空白内容，则为确定性的有用汇总。
- `results`：按规划顺序保存的 `WorkerResult` 元组。

每个 `WorkerResult` 包含 `subtask`、`status`、`attempts`、`output` 和 `error`。`status` 可为 `success`、`failed` 或 `timed_out`；`subtask` 包含 `id`、`role` 和 `instruction`。单个子 Agent 失败不会取消已成功的其他子 Agent。

编排只有一层：只有 Supervisor 可以创建子 Agent，子 Agent 只执行分配的子任务，不会递归创建更多 Agent。
