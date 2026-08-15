# tools/ask_human.py
from .registry import global_registry

def ask_human(question: str) -> str:
    """暂停执行，向用户提问并获取回答"""
    print(f"\n[需要用户输入] {question}")
    user_answer = input("请输入你的回答：")
    return f"用户回答：{user_answer}"

def register_ask_human():
    global_registry.register(
        name="ask_human",
        func=ask_human,
        description="向用户提问并获取回答，用于需要人工确认或补充信息的场景",
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "向用户提出的问题"}
            },
            "required": ["question"]
        },
        require_permission=False
    )