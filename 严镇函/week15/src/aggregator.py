"""
Aggregator 模块 - 结果汇总器

核心职责：
1. 收集所有 SubAgent 的执行结果
2. 整合、去重、排序
3. 生成最终的统一答案

为什么需要 Aggregator？
- 多个 SubAgent 的结果可能是碎片化的
- 需要统一视角整合信息
- 可能需要处理冲突信息
- 需要按照原始问题的逻辑组织答案
"""

import logging
from typing import Optional

from llm_client import llm_client
from models import Plan, SubTaskResult, TaskStatus

logger = logging.getLogger(__name__)


AGGREGATOR_SYSTEM_PROMPT = """你是一个结果汇总专家。你的职责是：

1. 整合多个子任务的执行结果
2. 消除重复信息
3. 解决可能的冲突
4. 按照原始问题的逻辑组织答案
5. 保持回答简洁、结构清晰

汇总原则：
- 如果子任务结果是数据，进行对比分析
- 如果子任务结果是信息，按重要性排序
- 如果存在失败的任务，说明哪些信息缺失
- 使用 Markdown 格式让答案更易读
"""


class Aggregator:
    """
    结果汇总器

    将多个 SubTask 的结果整合为最终答案。
    """

    async def aggregate(
        self,
        plan: Plan,
        results: list[SubTaskResult],
    ) -> str:
        """
        汇总所有子任务结果，生成最终答案

        Args:
            plan: 原始执行计划
            results: 所有子任务的执行结果

        Returns:
            str: 最终汇总的答案
        """
        logger.info("开始汇总结果")

        # 分类结果
        success_results = [r for r in results if r.status == TaskStatus.SUCCESS]
        failed_results = [r for r in results if r.status == TaskStatus.FAILED]

        logger.info(f"成功: {len(success_results)} 个, 失败: {len(failed_results)} 个")

        # 构建汇总提示词
        context_parts = []
        context_parts.append(f"原始问题：{plan.original_query}\n")
        context_parts.append("=" * 50)

        # 成功的结果
        if success_results:
            context_parts.append("\n【子任务执行结果】\n")
            for result in success_results:
                context_parts.append(f"\n--- {result.task_id} ---")
                context_parts.append(result.result or "(无结果)")

        # 失败的结果
        if failed_results:
            context_parts.append("\n\n【执行失败的任务】\n")
            for result in failed_results:
                context_parts.append(f"- {result.task_id}: {result.error}")

        user_prompt = "\n".join(context_parts)
        user_prompt += (
            "\n\n请根据以上子任务执行结果，汇总生成一个完整的最终答案。"
            "确保答案直接回应原始问题，结构清晰。"
        )

        # 调用 LLM 生成汇总结果
        try:
            final_answer = await llm_client.chat_completion(
                system_prompt=AGGREGATOR_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.5,
            )
            logger.info("结果汇总完成")
            return final_answer

        except Exception as e:
            logger.error(f"汇总失败: {e}")
            # 降级：直接拼接结果
            return self._fallback_aggregate(plan, results)

    def _fallback_aggregate(self, plan: Plan, results: list[SubTaskResult]) -> str:
        """
        降级汇总方案：直接拼接所有成功结果
        """
        lines = [f"# {plan.original_query}\n"]

        success_results = [r for r in results if r.status == TaskStatus.SUCCESS]
        if not success_results:
            return "所有任务执行失败，无法生成答案。"

        for result in success_results:
            lines.append(f"\n## {result.task_id}\n")
            lines.append(result.result or "(无结果)")

        failed = [r for r in results if r.status == TaskStatus.FAILED]
        if failed:
            lines.append("\n\n---")
            lines.append("以下任务执行失败：")
            for r in failed:
                lines.append(f"- {r.task_id}: {r.error}")

        return "\n".join(lines)