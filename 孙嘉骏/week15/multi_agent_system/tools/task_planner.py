# tools/task_planner.py
from .registry import global_registry
from typing import List, Dict, Any
import json

def plan_tasks(task_description: str) -> str:
    """
    将复杂任务分解为多个子任务，返回任务图 JSON。
    此处使用简单规则模拟，实际可调用 LLM 生成更优分解。
    """
    # 模拟：根据关键词简单拆分
    # 真实场景可调用 LLM 生成任务列表
    # 这里返回一个示例任务图，包含任务及其依赖关系
    tasks = [
        {"id": 1, "description": f"调研与“{task_description}”相关的背景信息", "dependencies": []},
        {"id": 2, "description": f"分析“{task_description}”的核心要素", "dependencies": [1]},
        {"id": 3, "description": f"提出针对“{task_description}”的解决方案", "dependencies": [2]},
        {"id": 4, "description": f"总结并撰写最终报告", "dependencies": [3]}
    ]
    return json.dumps(tasks, ensure_ascii=False)

def register_task_planner():
    global_registry.register(
        name="plan_tasks",
        func=plan_tasks,
        description="将一个复杂任务分解为多个子任务，返回任务图（JSON列表，包含id、description、dependencies字段）。",
        parameters={
            "type": "object",
            "properties": {
                "task_description": {"type": "string", "description": "要分解的总体任务描述"}
            },
            "required": ["task_description"]
        }
    )