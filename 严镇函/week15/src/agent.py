"""
MainAgent 模块 - 主编排器

核心职责：
1. 接收用户请求
2. 调用 Planner 拆分任务
3. 调用 ParallelExecutor 并行执行
4. 调用 Aggregator 汇总结果
5. 返回最终答案

这是整个 Multi-Agent 系统的入口和协调中心，
类似 Spring 中的 Service 层，负责业务流程编排。
"""

import logging
import time
from typing import Optional

from aggregator import Aggregator
from executor import ParallelExecutor
from models import AgentRequest, AgentResponse, Plan, SubTaskResult
from planner import Planner

logger = logging.getLogger(__name__)


class MainAgent:
    """
    主 Agent

    协调 Planner → Executor → Aggregator 的完整流程。
    """

    def __init__(self):
        self.planner = Planner()
        self.executor = ParallelExecutor()
        self.aggregator = Aggregator()
        logger.info("MainAgent 初始化完成")

    async def run(self, request: AgentRequest) -> AgentResponse:
        """
        执行完整的 Multi-Agent 流程

        流程：
        用户请求 → Planner 拆分 → Executor 并行执行 → Aggregator 汇总 → 返回结果

        Args:
            request: 用户请求（包含 query 和可选 context）

        Returns:
            AgentResponse: 包含执行计划和最终答案的完整响应
        """
        total_start = time.time()
        query = request.query

        logger.info(f"{'='*50}")
        logger.info(f"MainAgent 开始处理: {query}")
        logger.info(f"{'='*50}")

        try:
            # Step 1: Planner 拆分任务
            logger.info("[Step 1/3] Planner 拆分任务...")
            plan = await self.planner.plan(query, request.context)

            if not plan.subtasks:
                return AgentResponse(
                    success=False,
                    query=query,
                    error="Planner 未生成任何子任务",
                )

            # Step 2: ParallelExecutor 并行执行
            logger.info("[Step 2/3] ParallelExecutor 并行执行子任务...")
            results = await self.executor.execute(plan.subtasks)

            # Step 3: Aggregator 汇总结果
            logger.info("[Step 3/3] Aggregator 汇总结果...")
            final_answer = await self.aggregator.aggregate(plan, results)

            total_time = time.time() - total_start

            logger.info(f"{'='*50}")
            logger.info(f"MainAgent 处理完成，总耗时: {total_time:.2f}s")
            logger.info(f"{'='*50}")

            return AgentResponse(
                success=True,
                query=query,
                plan=plan,
                results=results,
                final_answer=final_answer,
                total_time=total_time,
            )

        except Exception as e:
            total_time = time.time() - total_start
            error_msg = str(e)
            logger.error(f"MainAgent 执行失败: {error_msg}")

            return AgentResponse(
                success=False,
                query=query,
                error=error_msg,
                total_time=total_time,
            )


# 全局 MainAgent 实例（单例）
main_agent = MainAgent()