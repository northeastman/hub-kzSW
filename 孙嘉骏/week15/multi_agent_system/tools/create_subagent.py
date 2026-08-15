# tools/create_subagent.py
from .registry import global_registry
from agents.sub_agent import SubAgent
from utils.config import MAX_SUB_AGENT_STEPS, MAX_SUBAGENT_DEPTH
from layers.memory import Memory
import threading

# 全局上下文，用于传递主Agent状态（实际可封装到上下文对象）
_main_agent_context = {
    "memory": None,          # 主Agent的记忆对象
    "depth": 0,              # 当前深度
    "max_depth": MAX_SUBAGENT_DEPTH,
    "allowed_tools": []      # 允许子Agent使用的工具列表
}

def _create_subagent(description: str, tools: list = None, max_steps: int = MAX_SUB_AGENT_STEPS, **kwargs) -> str:
    """
    创建子Agent并执行子任务，返回子Agent的最终结果。
    多个 create_subagent 调用会被并行执行（在执行层实现）。
    """
    memory = _main_agent_context.get("memory")
    depth = _main_agent_context.get("depth", 0)
    max_depth = _main_agent_context.get("max_depth", MAX_SUBAGENT_DEPTH)
    
    if depth >= max_depth:
        return f"Error: 已达到最大子Agent嵌套深度（{max_depth}）"

    # 创建子Agent
    sub = SubAgent(
        memory=memory,                # 共享记忆对象
        allowed_tools=tools or [],    # 子Agent允许使用的工具
        max_steps=max_steps,
        depth=depth + 1
    )
    result = sub.run(description)
    
    # 将子Agent结果存入长期记忆
    if memory:
        memory.add_subagent_result(description, result)
    
    return result

def register_create_subagent():
    global_registry.register(
        name="create_subagent",
        func=_create_subagent,
        description="创建一个子Agent来执行独立子任务，返回子任务的最终结果。可以同时创建多个子Agent并行处理。",
        parameters={
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "子任务的详细描述，包括目标和期望输出格式"
                },
                "tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "子Agent可以使用的工具名称列表，默认为空表示仅基础工具"
                },
                "max_steps": {
                    "type": "integer",
                    "description": "子Agent最大ReAct步数，默认为5"
                }
            },
            "required": ["description"]
        },
        require_permission=False
    )

def set_main_agent_context(memory, depth=0, max_depth=MAX_SUBAGENT_DEPTH, allowed_tools=None):
    """设置主Agent上下文，供create_subagent使用"""
    _main_agent_context["memory"] = memory
    _main_agent_context["depth"] = depth
    _main_agent_context["max_depth"] = max_depth
    if allowed_tools is not None:
        _main_agent_context["allowed_tools"] = allowed_tools