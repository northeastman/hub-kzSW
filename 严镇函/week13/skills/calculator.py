"""
计算器技能 - 简单工具示例
"""

SKILL_METADATA = {
    "name": "calculator",
    "display_name": "计算器",
    "description": "执行数学计算",
    "version": "1.0.0",
    "author": "Agent Memory System",
    "tags": ["计算", "数学", "工具"],
    "triggers": ["计算", "等于多少", "加减乘除", "+", "-", "*", "/"],
    "requires_memory": False,
}


def execute(context: dict) -> dict:
    """执行计算技能"""
    expression = context.get("expression", "")
    
    if not expression:
        return {"error": "请提供计算表达式"}
    
    try:
        # 安全计算（仅允许数学运算）
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return {"error": "表达式包含非法字符"}
        
        result = eval(expression)
        return {
            "expression": expression,
            "result": result,
            "message": f"{expression} = {result}"
        }
    except Exception as e:
        return {"error": f"计算失败: {str(e)}"}