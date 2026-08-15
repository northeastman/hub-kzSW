"""
交互式命令行运行入口

直接运行后输入问题，不需要命令行参数。

使用方式：
    python cli.py
    > 帮我查一下北京和上海的天气，告诉我哪个更适合出行
"""

import asyncio
import logging

from agent import main_agent
from models import AgentRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


async def run_single(query: str):
    """运行单个任务"""
    request = AgentRequest(query=query)
    response = await main_agent.run(request)

    print("\n" + "=" * 60)
    print(f"查询: {response.query}")
    print("=" * 60)

    if not response.success:
        print(f"\n执行失败: {response.error}")
        return

    # 显示执行计划
    if response.plan:
        print(f"\n执行计划:")
        print(f"  拆分理由: {response.plan.reasoning}")
        print(f"  子任务数: {len(response.plan.subtasks)}")
        for st in response.plan.subtasks:
            deps = f" (依赖: {st.dependencies})" if st.dependencies else ""
            print(f"  - {st.task_id}: {st.description[:50]}...{deps}")

    # 显示执行结果
    print(f"\n执行结果:")
    for result in response.results:
        status_icon = "[OK]" if result.status.value == "success" else "[FAIL]"
        print(f"  {status_icon} {result.task_id}: {result.status.value} (耗时: {result.execution_time:.2f}s)")

    # 显示最终答案
    print(f"\n最终答案:")
    print("-" * 60)
    print(response.final_answer)
    print("-" * 60)
    print(f"\n总耗时: {response.total_time:.2f}s")


async def main():
    print("=" * 60)
    print("Multi-Agent 并行任务执行器")
    print("输入问题直接执行，输入 exit 退出")
    print("=" * 60)

    while True:
        print()
        query = input("> ").strip()

        if not query:
            continue

        if query.lower() in ("exit", "quit", "退出"):
            print("再见!")
            break

        await run_single(query)


if __name__ == "__main__":
    asyncio.run(main())