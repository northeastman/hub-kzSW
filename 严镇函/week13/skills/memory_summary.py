"""
记忆总结技能 - 演示如何使用记忆系统

这个技能需要访问用户的记忆数据
"""

SKILL_METADATA = {
    "name": "memory_summary",
    "display_name": "记忆总结",
    "description": "总结用户的记忆和偏好，生成用户画像摘要",
    "version": "1.0.0",
    "author": "Agent Memory System",
    "tags": ["记忆", "总结", "用户画像"],
    "triggers": ["总结记忆", "我的记忆", "记忆总结", "我是谁"],
    "requires_memory": True,
}


def execute(context: dict) -> dict:
    """执行记忆总结技能"""
    memory_access = context.get("memory_access", {})
    
    # 获取用户画像
    user_profile = memory_access.get("get_user_profile", lambda: {})()
    
    # 搜索相关记忆
    search_query = context.get("query", "用户偏好")
    related_memories = memory_access.get("search_memory", lambda q, k: [])(search_query, top_k=5)
    
    # 获取近期日志
    recent_logs = memory_access.get("get_recent_logs", lambda d: "")(days=2)
    
    return {
        "user_profile": user_profile,
        "related_memories_count": len(related_memories),
        "related_memories": related_memories,
        "has_recent_logs": bool(recent_logs),
        "summary": f"用户画像包含 {len(user_profile)} 个字段，找到 {len(related_memories)} 条相关记忆"
    }