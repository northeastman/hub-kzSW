from pathlib import Path
from typing import Callable

def _clean_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    return s.encode('utf-8', errors='replace').decode('utf-8')

class Tool:
    def __init__(self, name: str, description: str, func: Callable, parameters: dict):
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters

    def to_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

def create_tools(registry, state: dict) -> list:
    def load_skill(skill_name: str) -> str:
        if skill_name in state.get('loaded_skills', {}):
            return f"Skill '{skill_name}' already loaded."
        body = registry.load_skill_body(skill_name)
        if body is None:
            return f"Error: Skill '{skill_name}' not found. Available: {registry.list_skill_names()}"
        state.setdefault('loaded_skills', {})[skill_name] = body
        return _clean_text(f"✅ Skill '{skill_name}' loaded successfully.")

    def read_skill_resource(skill_name: str, resource_path: str) -> str:
        loaded = state.get('loaded_skills', {})
        if skill_name not in loaded:
            return f"Error: Skill '{skill_name}' not loaded. Call load_skill first."
        skill_path = registry.get_skill_path(skill_name)
        if not skill_path:
            return f"Error: Skill '{skill_name}' not found."
        target = skill_path / resource_path
        try:
            target.resolve().relative_to(skill_path.resolve())
        except ValueError:
            return "Error: Invalid resource path."
        if not target.exists():
            return f"Error: Resource '{resource_path}' not found."
        content = target.read_text(encoding='utf-8', errors='replace')
        return _clean_text(content)

    return [
        Tool(
            name="load_skill",
            description="加载指定技能的完整指令。当用户的问题匹配某个技能的 description 时调用。",
            func=load_skill,
            parameters={
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "要加载的技能名称"}
                },
                "required": ["skill_name"]
            }
        ),
        Tool(
            name="read_skill_resource",
            description="读取已加载技能的参考文档或脚本内容。",
            func=read_skill_resource,
            parameters={
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "已加载的技能名称"},
                    "resource_path": {"type": "string", "description": "资源文件的相对路径，如 'references/style-guide.md'"}
                },
                "required": ["skill_name", "resource_path"]
            }
        )
    ]