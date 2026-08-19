# tools/builtin_tools.py
from .registry import global_registry
import math

def search(query: str) -> str:
    """模拟搜索工具，实际可替换为真实搜索引擎"""
    # 这里返回模拟结果
    return f"搜索结果（模拟）：关于“{query}”的相关信息...（请接入真实搜索API）"

def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        # 安全地计算表达式（仅允许数学函数）
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"

def register_builtin_tools():
    """注册内置工具"""
    global_registry.register(
        name="search",
        func=search,
        description="搜索外部信息，返回相关结果",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"}
            },
            "required": ["query"]
        }
    )
    global_registry.register(
        name="calculator",
        func=calculator,
        description="计算数学表达式，支持 math 模块函数",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "数学表达式，如 '2+3*4' 或 'sqrt(16)'"}
            },
            "required": ["expression"]
        }
    )