# layers/execution.py
from typing import List, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from tools.registry import ToolRegistry, global_registry
from utils.config import MAX_PARALLEL_SUBAGENTS

class Execution:
    """执行层：负责解析LLM工具调用并并行执行"""
    def __init__(self, registry: ToolRegistry = global_registry):
        self.registry = registry

    def execute_tool_calls(self, tool_calls: List[Dict], allowed_tools: List[str] = None) -> List[Dict]:
        """
        并行执行所有工具调用。
        Args:
            tool_calls: LLM返回的tool_calls列表，每个元素包含id, function.name, function.arguments
            allowed_tools: 允许的工具名称列表，若为None则允许全部
        Returns:
            结果列表，每个元素包含 tool_call_id 和 content
        """
        if not tool_calls:
            return []
        
        results = []
        # 限制并发数量
        max_workers = min(len(tool_calls), MAX_PARALLEL_SUBAGENTS)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_call = {}
            for call in tool_calls:
                func_name = call["function"]["name"]
                arguments = call["function"].get("arguments", "{}")
                # arguments 可能是 JSON 字符串，需要解析
                if isinstance(arguments, str):
                    import json
                    try:
                        arguments = json.loads(arguments)
                    except:
                        arguments = {}
                future = executor.submit(
                    self.registry.execute,
                    func_name,
                    arguments,
                    allowed_tools
                )
                future_to_call[future] = call
            for future in as_completed(future_to_call):
                call = future_to_call[future]
                try:
                    content = future.result()
                except Exception as e:
                    content = f"Error: {str(e)}"
                results.append({
                    "tool_call_id": call["id"],
                    "content": content
                })
        return results