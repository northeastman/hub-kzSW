# tools/verify_result.py
from .registry import global_registry

def verify_result(task_description: str, result: str) -> str:
    """
    验证子任务结果是否满足要求。
    这里使用简单规则：检查结果是否为空或包含错误标记。
    实际可调用 LLM 进行语义验证。
    """
    if not result or "错误" in result or "Error" in result:
        return "验证失败：结果为空或包含错误"
    if len(result) < 10:
        return "验证失败：结果过短"
    return "验证通过"

def register_verify_result():
    global_registry.register(
        name="verify_result",
        func=verify_result,
        description="验证子任务结果是否符合要求，返回验证通过或失败原因",
        parameters={
            "type": "object",
            "properties": {
                "task_description": {"type": "string", "description": "子任务描述"},
                "result": {"type": "string", "description": "子任务返回的结果"}
            },
            "required": ["task_description", "result"]
        }
    )