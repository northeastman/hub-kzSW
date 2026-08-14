from dataclasses import dataclass
from typing import Literal, Mapping, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class AgentConfig:
    max_agents: int = 4
    worker_timeout: float = 60.0
    retries: int = 1


class LLMClient(Protocol):
    async def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.2,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class SubTask:
    id: str
    role: str
    instruction: str


@dataclass(frozen=True, slots=True)
class WorkerResult:
    subtask: SubTask
    status: Literal["success", "failed", "timed_out"]
    attempts: int
    output: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RunReport:
    task: str
    final_answer: str
    results: tuple[WorkerResult, ...]
