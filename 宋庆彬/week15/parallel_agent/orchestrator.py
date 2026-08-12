import asyncio
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import (
    AgentConfig,
    LLMClient,
    RunReport,
    SubTask,
    WorkerResult,
)


_GENERAL_ROLE = "通用执行 Agent"
_HARD_AGENT_LIMIT = 4


class AgentPool:
    def __init__(self, client: LLMClient, config: AgentConfig):
        self._client = client
        self._config = config

    async def run(
        self, task: str, subtasks: Sequence[SubTask]
    ) -> tuple[WorkerResult, ...]:
        concurrency = max(1, min(self._config.max_agents, _HARD_AGENT_LIMIT))
        semaphore = asyncio.Semaphore(concurrency)

        async def run_guarded(subtask: SubTask) -> WorkerResult:
            async with semaphore:
                return await self._run_worker(task, subtask)

        return tuple(
            await asyncio.gather(*(run_guarded(subtask) for subtask in subtasks))
        )

    async def _run_worker(self, task: str, subtask: SubTask) -> WorkerResult:
        messages = [
            {
                "role": "system",
                "content": (
                    f"你是{subtask.role}。你只执行已分配的子任务，"
                    "不得创建或调用其他 Agent。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"原始任务：{task}\n"
                    f"你的子任务：{subtask.instruction}"
                ),
            },
        ]
        attempts_allowed = max(0, self._config.retries) + 1
        last_error: Exception | None = None
        last_status: Literal["failed", "timed_out"] = "failed"

        for attempt in range(1, attempts_allowed + 1):
            try:
                output = await asyncio.wait_for(
                    self._client.complete(messages, temperature=0.2),
                    timeout=self._config.worker_timeout,
                )
                return WorkerResult(
                    subtask=subtask,
                    status="success",
                    attempts=attempt,
                    output=output,
                )
            except TimeoutError as exc:
                last_error = exc
                last_status = "timed_out"
            except Exception as exc:
                last_error = exc
                last_status = "failed"

        return WorkerResult(
            subtask=subtask,
            status=last_status,
            attempts=attempts_allowed,
            error=str(last_error) if last_error else None,
        )


class SupervisorAgent:
    def __init__(
        self, client: LLMClient, config: AgentConfig | None = None
    ) -> None:
        self._client = client
        self._config = config or AgentConfig()

    async def run(self, task: str) -> RunReport:
        subtasks = await self._plan(task)
        results = await AgentPool(self._client, self._config).run(task, subtasks)
        final_answer = await self._aggregate(task, results)
        return RunReport(task=task, final_answer=final_answer, results=results)

    async def _plan(self, task: str) -> tuple[SubTask, ...]:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是任务规划器。将用户任务拆分为可并行执行的子任务，"
                    '只返回 JSON：{"tasks":[{"role":"...",'
                    '"instruction":"..."}]}。'
                ),
            },
            {"role": "user", "content": task},
        ]
        try:
            raw_plan = await self._client.complete(messages, temperature=0.2)
            parsed: Any = json.loads(raw_plan)
        except (Exception, json.JSONDecodeError):
            return (self._general_subtask(task),)

        if not isinstance(parsed, Mapping) or not isinstance(
            parsed.get("tasks"), list
        ):
            return (self._general_subtask(task),)

        agent_limit = max(1, min(self._config.max_agents, _HARD_AGENT_LIMIT))
        subtasks: list[SubTask] = []
        for item in parsed["tasks"]:
            if not isinstance(item, Mapping):
                continue
            role = item.get("role")
            instruction = item.get("instruction")
            if not isinstance(role, str) or not role.strip():
                continue
            if not isinstance(instruction, str) or not instruction.strip():
                continue
            subtasks.append(
                SubTask(
                    id=f"agent-{len(subtasks) + 1}",
                    role=role.strip(),
                    instruction=instruction.strip(),
                )
            )
            if len(subtasks) == agent_limit:
                break

        return tuple(subtasks) or (self._general_subtask(task),)

    async def _aggregate(
        self, task: str, results: Sequence[WorkerResult]
    ) -> str:
        payload = [
            {
                "id": result.subtask.id,
                "role": result.subtask.role,
                "instruction": result.subtask.instruction,
                "status": result.status,
                "output": result.output,
                "error": result.error,
            }
            for result in results
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "你是结果汇总器。基于各 Agent 的成功或失败结果，"
                    "给出对用户有用的综合答案。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"原始任务：{task}\n"
                    f"Agent 结果：{json.dumps(payload, ensure_ascii=False)}"
                ),
            },
        ]
        try:
            answer = await self._client.complete(messages, temperature=0.2)
        except Exception:
            return self._fallback_aggregation(results)
        return answer if answer.strip() else self._fallback_aggregation(results)

    @staticmethod
    def _fallback_aggregation(results: Sequence[WorkerResult]) -> str:
        successful = [result for result in results if result.status == "success"]
        unsuccessful = [result for result in results if result.status != "success"]
        lines = ["聚合服务不可用，以下为确定性结果汇总。", "成功 Agent："]

        if successful:
            lines.extend(
                f"- {result.subtask.id}（{result.subtask.role}）："
                f"{result.output or '（无输出）'}"
                for result in successful
            )
        else:
            lines.append("- 无")

        lines.append("失败 Agent：")
        if unsuccessful:
            for result in unsuccessful:
                reason = result.error or (
                    "执行超时"
                    if result.status == "timed_out"
                    else "未提供错误信息"
                )
                lines.append(
                    f"- {result.subtask.id}（{result.subtask.role}，"
                    f"尝试 {result.attempts} 次）：{reason}"
                )
        else:
            lines.append("- 无")

        return "\n".join(lines)

    @staticmethod
    def _general_subtask(task: str) -> SubTask:
        return SubTask(id="agent-1", role=_GENERAL_ROLE, instruction=task)
