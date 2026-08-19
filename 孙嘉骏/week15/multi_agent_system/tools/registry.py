# tools/registry.py
from typing import Dict, Callable, Any, List
import json

class ToolRegistry:
    """工具注册表，管理所有可用工具"""
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, func: Callable, description: str, parameters: dict,
                 require_permission: bool = False):
        """注册工具
        Args:
            name: 工具名称
            func: 实际执行函数
            description: 工具描述（用于 LLM 理解）
            parameters: JSON Schema 格式的参数定义
            require_permission: 是否需要额外权限（用于权限控制）
        """
        self._tools[name] = {
            "function": func,
            "description": description,
            "parameters": parameters,
            "require_permission": require_permission
        }

    def get_tool_definitions(self, allowed_tools: List[str] = None) -> List[Dict]:
        """获取指定工具列表的 OpenAI function calling 格式定义"""
        defs = []
        for name, info in self._tools.items():
            if allowed_tools is None or name in allowed_tools:
                defs.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": info["description"],
                        "parameters": info["parameters"]
                    }
                })
        return defs

    def execute(self, name: str, arguments: dict, allowed_tools: List[str] = None) -> str:
        """执行工具，返回字符串结果"""
        if name not in self._tools:
            return f"Error: 工具 {name} 不存在"
        if allowed_tools is not None and name not in allowed_tools:
            return f"Error: 工具 {name} 不在允许列表中"
        info = self._tools[name]
        # 权限检查
        if info["require_permission"] and not self._check_permission(name):
            return f"Error: 工具 {name} 需要额外权限"
        try:
            result = info["function"](**arguments)
            # 统一转为字符串返回
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return f"Error executing {name}: {str(e)}"

    def _check_permission(self, name: str) -> bool:
        # 简化权限检查，实际可扩展为 ACL 或用户确认
        return True

# 全局工具注册表
global_registry = ToolRegistry()