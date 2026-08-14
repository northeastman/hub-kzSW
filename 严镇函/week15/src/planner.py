"""
Planner 模块 - 任务拆分器

核心职责：
1. 接收用户的复杂任务
2. 分析任务结构，拆分为独立的 SubTask
3. 确定 SubTask 之间的依赖关系
4. 输出结构化的执行计划

为什么不用简单规则拆分，而用 LLM？
- 复杂任务的边界不清晰，需要语义理解
- LLM 能判断哪些子任务可以并行，哪些必须串行
- 能根据任务类型自适应调整拆分策略
"""

import json
import logging
import uuid
from typing import Optional

from llm_client import llm_client
from models import Plan, SubTask, TaskStatus

logger = logging.getLogger(__name__)


PLANNER_SYSTEM_PROMPT = """你是一个任务规划专家。你的职责是将用户的复杂任务拆分为多个独立的子任务。

拆分原则：
1. 每个子任务应该是独立、可执行的单元
2. 如果子任务之间没有依赖关系，应该标记为可并行执行
3. 子任务描述要清晰具体，包含完成任务所需的所有信息
4. 不要过度拆分，保持任务粒度适中（一般 2-5 个子任务）

输出格式（必须严格遵循 JSON）：
{
    "reasoning": "拆分的思考过程",
    "subtasks": [
        {
            "task_id": "task_1",
            "description": "具体任务描述",
            "dependencies": []  // 依赖的其他 task_id 列表，无依赖则为空
        }
    ]
}

示例：
用户输入："帮我查一下北京和上海的天气，然后告诉我哪个更适合出行"
拆分结果：
{
    "reasoning": "需要分别查询两个城市的天气，然后进行对比分析",
    "subtasks": [
        {
            "task_id": "task_1",
            "description": "查询北京今天的天气情况，包括温度、降水、风力",
            "dependencies": []
        },
        {
            "task_id": "task_2",
            "description": "查询上海今天的天气情况，包括温度、降水、风力",
            "dependencies": []
        },
        {
            "task_id": "task_3",
            "description": "对比北京和上海的天气数据，分析哪个城市更适合出行，给出理由",
            "dependencies": ["task_1", "task_2"]
        }
    ]
}"""


class Planner:
    """
    任务规划器

    使用 LLM 分析复杂任务并拆分为可执行的子任务。
    """

    async def plan(self, query: str, context: Optional[dict] = None) -> Plan:
        """
        将用户查询拆分为执行计划

        Args:
            query: 用户的复杂任务描述
            context: 可选的上下文信息

        Returns:
            Plan: 包含子任务列表的执行计划
        """
        logger.info(f"开始规划任务: {query}")

        # 构建用户提示
        user_prompt = f"用户任务：{query}\n"
        if context:
            user_prompt += f"\n上下文信息：{json.dumps(context, ensure_ascii=False)}"

        # 调用 LLM 生成计划
        response = await llm_client.chat_with_json_output(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.3,  # 低温度确保输出稳定
        )

        # 解析 JSON 响应
        try:
            plan_data = json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"LLM 返回非 JSON 格式: {response}")
            # 降级处理：生成一个默认的单一任务计划
            return self._create_fallback_plan(query)

        # 构建 Plan 对象
        subtasks = []
        for i, st_data in enumerate(plan_data.get("subtasks", [])):
            subtasks.append(
                SubTask(
                    task_id=st_data.get("task_id", f"task_{i+1}"),
                    description=st_data["description"],
                    dependencies=st_data.get("dependencies", []),
                    status=TaskStatus.PENDING,
                )
            )

        plan = Plan(
            original_query=query,
            subtasks=subtasks,
            reasoning=plan_data.get("reasoning", ""),
        )

        logger.info(f"任务规划完成，生成 {len(subtasks)} 个子任务")
        for st in subtasks:
            deps = f" (依赖: {st.dependencies})" if st.dependencies else ""
            logger.info(f"  - {st.task_id}: {st.description[:50]}...{deps}")

        return plan

    def _create_fallback_plan(self, query: str) -> Plan:
        """
        降级方案：当 LLM 输出解析失败时，生成单一任务计划
        """
        logger.warning("使用降级方案：将任务作为单一子任务执行")
        return Plan(
            original_query=query,
            subtasks=[
                SubTask(
                    task_id="task_1",
                    description=query,
                    dependencies=[],
                    status=TaskStatus.PENDING,
                )
            ],
            reasoning="任务拆分失败，降级为单一任务执行",
        )