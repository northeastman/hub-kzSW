"""SubAgent —— 被主 agent 下发的子智能体。

每个 subagent 专注完成一个被拆解出来的子任务（一个独立的 LLM 调用），
返回结构化结果（含耗时、线程名等元信息），供主 agent 汇总。
subagent 之间互相独立、无共享状态，因此天然可并行。
"""
import time
import threading
from dataclasses import dataclass, field

from llm_client import llm_chat

SUBAGENT_SYSTEM = """你是一名专家型子研究员，只负责完成分配给你的这一个子课题。
要求：
- 只聚焦你被分配的子课题，不要越界去做别的子课题。
- 给出结构化、要点式的结论，信息密度高，不说废话、不加寒暄。
- 控制在约 250 字以内。若涉及数据/事实，注明这是基于模型知识的概述而非实时检索。"""


@dataclass
class SubTask:
    id: int
    title: str          # 子任务标题（简短）
    instruction: str    # 交给 subagent 的具体指令


@dataclass
class SubResult:
    id: int
    title: str
    content: str
    elapsed: float
    thread: str
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0


def run_subagent(task: SubTask, log=None) -> SubResult:
    """执行单个子任务。这是会被线程池并行调度的工作单元。"""
    thread = threading.current_thread().name
    started = time.perf_counter()
    wall_start = time.time()
    if log:
        log(f"  ▶ [子{task.id}·{thread}] 开始：{task.title}")
    try:
        content = llm_chat(SUBAGENT_SYSTEM, task.instruction)
        err = ""
    except Exception as e:  # noqa: BLE001
        content = ""
        err = str(e)
    elapsed = time.perf_counter() - started
    if log:
        status = "失败" if err else "完成"
        log(f"  {'✖' if err else '◀'} [子{task.id}·{thread}] {status}（{elapsed:.1f}s）")
    return SubResult(
        id=task.id, title=task.title, content=content, elapsed=elapsed,
        thread=thread, error=err, started_at=wall_start, finished_at=time.time(),
    )
