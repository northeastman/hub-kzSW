# tools/__init__.py
from .registry import global_registry
from .builtin_tools import register_builtin_tools
from .task_planner import register_task_planner
from .verify_result import register_verify_result
from .create_subagent import register_create_subagent
from .ask_human import register_ask_human

def register_all_tools():
    """注册所有工具到全局注册表"""
    register_builtin_tools()
    register_task_planner()
    register_verify_result()
    register_create_subagent()
    register_ask_human()