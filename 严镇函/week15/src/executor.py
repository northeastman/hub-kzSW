"""
并行执行器模块

核心职责：
1. 接收 SubTask 列表
2. 根据依赖关系构建执行拓扑
3. 使用 asyncio 并行执行无依赖的任务
4. 收集所有 SubTaskResult

关键技术点：
- asyncio.gather: 并发执行多个协程
- asyncio.Semaphore: 控制并发数，防止同时发起过多请求
- 依赖图解析: 确保依赖任务先完成

为什么用 asyncio 而不是多线程/多进程？
- LLM API 调用是 IO 密集型（等待网络响应）
- asyncio 是单线程协程，开销远小于多线程
- 适合大量并发网络请求场景
"""

import asyncio
import logging
import time
from typing import Optional

from config import settings
from llm_client import llm_client
from models import SubTask, SubTaskResult, TaskStatus

logger = logging.getLogger(__name__)


# SubAgent 的系统提示词
SUBAGENT_SYSTEM_PROMPT = """你是一个专业的任务执行助手。你的职责是：
1. 根据给定的任务描述，完成具体工作
2. 输出简洁、准确的结果
3. 如果任务涉及数据查询，模拟返回合理的结果
4. 如果无法完成任务，说明原因

注意：
- 直接输出结果，不要解释过程
- 保持回答简洁，重点突出
"""


class SubAgent:
    """
    子任务执行 Agent

    每个 SubAgent 负责执行一个 SubTask。
    实际项目中，这里可以扩展为调用不同工具（搜索、数据库、API等）。
    """

    async def execute(self, subtask: SubTask) -> SubTaskResult:
        """
        执行单个子任务

        Args:
            subtask: 要执行的子任务

        Returns:
            SubTaskResult: 执行结果
        """
        task_id = subtask.task_id
        start_time = time.time()

        logger.info(f"[SubAgent] 开始执行任务: {task_id}")
        subtask.status = TaskStatus.RUNNING

        try:
            # 构建提示词
            user_prompt = f"请完成以下任务：\n\n{subtask.description}"

            # 调用 LLM 执行任务
            result = await llm_client.chat_completion(
                system_prompt=SUBAGENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.5,
            )

            execution_time = time.time() - start_time
            logger.info(f"[SubAgent] 任务完成: {task_id} (耗时: {execution_time:.2f}s)")

            subtask.status = TaskStatus.SUCCESS
            subtask.result = result

            return SubTaskResult(
                task_id=task_id,
                status=TaskStatus.SUCCESS,
                result=result,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"[SubAgent] 任务失败: {task_id}, 错误: {error_msg}")

            subtask.status = TaskStatus.FAILED
            subtask.error = error_msg

            return SubTaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error=error_msg,
                execution_time=execution_time,
            )


class ParallelExecutor:
    """
    并行执行器

    管理多个 SubAgent 的并发执行，处理任务依赖关系。
    """

    def __init__(self):
        # 使用信号量控制最大并发数
        self.semaphore = asyncio.Semaphore(settings.max_concurrent_tasks)
        self.subagent = SubAgent()

    async def _execute_with_semaphore(self, subtask: SubTask) -> SubTaskResult:
        """
        在信号量控制下执行任务

        Semaphore 确保同时执行的 SubAgent 不超过 max_concurrent_tasks，
        防止对 LLM API 造成过大压力或触发限流。
        """
        async with self.semaphore:
            return await self.subagent.execute(subtask)

    def _build_dependency_graph(self, subtasks: list[SubTask]) -> dict[str, set[str]]:
        """
        构建依赖图

        Returns:
            dict: {task_id: {依赖的task_id集合}}
        """
        graph = {}
        task_ids = {st.task_id for st in subtasks}

        for st in subtasks:
            # 过滤掉不存在的依赖
            valid_deps = [dep for dep in st.dependencies if dep in task_ids]
            graph[st.task_id] = set(valid_deps)

        return graph

    async def execute(self, subtasks: list[SubTask]) -> list[SubTaskResult]:
        """
        并行执行所有子任务（考虑依赖关系）

        执行策略：
        1. 每轮找出所有依赖已满足的任务
        2. 使用 asyncio.gather 并行执行这些任务
        3. 等待本轮所有任务完成
        4. 重复直到所有任务完成

        Args:
            subtasks: 子任务列表

        Returns:
            list[SubTaskResult]: 所有任务的执行结果
        """
        if not subtasks:
            return []

        logger.info(f"开始并行执行 {len(subtasks)} 个子任务")

        # 构建依赖图
        dependency_graph = self._build_dependency_graph(subtasks)
        completed_tasks: dict[str, SubTaskResult] = {}
        pending_tasks = {st.task_id: st for st in subtasks}

        round_num = 0
        while pending_tasks:
            round_num += 1

            # 找出当前可以执行的任务（依赖已满足）
            ready_tasks = []
            for task_id, subtask in list(pending_tasks.items()):
                deps = dependency_graph.get(task_id, set())
                if deps.issubset(completed_tasks.keys()):
                    ready_tasks.append(subtask)

            if not ready_tasks:
                # 有任务但未就绪，说明存在循环依赖
                remaining = ", ".join(pending_tasks.keys())
                logger.error(f"检测到循环依赖或无法执行的任务: {remaining}")
                break

            logger.info(f"第 {round_num} 轮: 并行执行 {len(ready_tasks)} 个任务")

            # 并行执行就绪的任务
            tasks = [self._execute_with_semaphore(st) for st in ready_tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理结果
            for subtask, result in zip(ready_tasks, results):
                if isinstance(result, Exception):
                    # 异常情况
                    result = SubTaskResult(
                        task_id=subtask.task_id,
                        status=TaskStatus.FAILED,
                        error=str(result),
                    )
                completed_tasks[subtask.task_id] = result
                del pending_tasks[subtask.task_id]

                status_icon = "✓" if result.status == TaskStatus.SUCCESS else "✗"
                logger.info(f"  {status_icon} {subtask.task_id}: {result.status.value}")

        logger.info(f"所有任务执行完成，共 {len(completed_tasks)} 个")
        return list(completed_tasks.values())