"""
工具注册表
统一管理所有可用工具的注册和调用
"""
from typing import Any, Callable, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ToolDefinition:
    """工具定义类"""

    def __init__(self, name: str, func: Callable, description: str, parameters: Optional[Dict] = None):
        self.name = name
        self.func = func
        self.description = description
        self.parameters = parameters or {}

    def __call__(self, **kwargs) -> Any:
        """调用工具"""
        try:
            logger.info(f"调用工具：{self.name}, 参数：{kwargs}")
            result = self.func(**kwargs)
            logger.info(f"工具 {self.name} 执行完成")
            return result
        except Exception as e:
            logger.error(f"工具 {self.name} 执行失败：{e}")
            return f"工具执行错误：{str(e)}"


class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, name: str, func: Callable, description: str, parameters: Optional[Dict] = None):
        """注册一个工具"""
        self._tools[name] = ToolDefinition(name, func, description, parameters)
        logger.info(f"注册工具：{name}")

    def get(self, name: str) -> Optional[ToolDefinition]:
        """获取工具"""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """列出所有已注册工具名称"""
        return list(self._tools.keys())

    def get_tool_schema(self, name: str) -> Optional[Dict]:
        """获取工具的 JSON Schema 描述"""
        tool = self._tools.get(name)
        if tool:
            return {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
        return None

    def call(self, name: str, **kwargs) -> Any:
        """调用工具"""
        tool = self.get(name)
        if not tool:
            raise ValueError(f"未知工具：{name}")
        return tool(**kwargs)


# 全局工具注册表实例
TOOLS_REGISTRY = ToolRegistry()


def register_tool(name: str, description: str, parameters: Optional[Dict] = None):
    """工具注册装饰器"""
    def decorator(func: Callable):
        TOOLS_REGISTRY.register(name, func, description, parameters)
        return func
    return decorator
