"""
Layer 5 能力层：工具 schema 定义 + tool_call 分发

把"工具长什么样"和"如何执行一个工具调用"从 serve.py 剥离，
让编排层（serve.py）只负责 tool-use 循环本身。

三个职责：
  build_skill_prompt_section  ── 把 skill 列表拼成注入 system prompt 的一段
  build_tools_schema          ── 返回 OpenAI tools 数组（load_skill / run_skill_script）
  dispatch_tool_call          ── 执行一个工具调用，返回回注文本 + SSE 用 meta
"""

import logging

from src.skill_loader import SkillLoader
from src.skill_runner import SkillRunner

logger = logging.getLogger(__name__)


def build_skill_prompt_section(skills: list[dict]) -> str:
    """把 list_skills() 结果拼成一段注入 system prompt 的文本；空列表返回空串。"""
    if not skills:
        return ""
    lines = ["## 可用能力（Skills）",
             "你可以在需要时调用工具加载以下能力的详细指令，或执行其脚本："]
    for s in skills:
        tag = "（含可执行脚本）" if s.get("has_script") else ""
        lines.append(f"- {s['name']}：{s['description']}{tag}")
    lines.append("需要某能力时，先用 load_skill 读取其指令；"
                 "若该能力标注含脚本且需要真实计算/数据，再用 run_skill_script 执行。")
    return "\n".join(lines)


def build_tools_schema() -> list[dict]:
    """返回传给 LLM 的 tools 数组。"""
    return [
        {
            "type": "function",
            "function": {
                "name": "load_skill",
                "description": "加载指定 skill 的完整指令正文。当用户需求匹配某个可用能力时调用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "skill 名称，取自可用能力列表"},
                    },
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_skill_script",
                "description": "执行指定 skill 的入口脚本并返回其输出。仅对标注含脚本的能力有效。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "skill 名称"},
                        "args": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "传给脚本的命令行参数（可选）",
                        },
                    },
                    "required": ["name"],
                },
            },
        },
    ]


def dispatch_tool_call(name: str, arguments: dict,
                       loader: SkillLoader, runner: SkillRunner) -> dict:
    """
    执行一个工具调用。返回：
      {"tool", "skill", "ok", "result", "meta"}
    result 是回注给 LLM 的文本；meta 供 SSE 事件用。永不抛异常。
    """
    arguments = arguments or {}
    skill_name = (arguments.get("name") or "").strip()
    try:
        if name == "load_skill":
            body = loader.load_skill(skill_name)
            if body is None:
                return {"tool": name, "skill": skill_name, "ok": False,
                        "result": f"skill 不存在：{skill_name}",
                        "meta": {"name": skill_name}}
            return {"tool": name, "skill": skill_name, "ok": True,
                    "result": body,
                    "meta": {"name": skill_name, "chars": len(body)}}

        if name == "run_skill_script":
            args = arguments.get("args") or []
            if not isinstance(args, list):
                args = [str(args)]
            res = runner.run_script(skill_name, [str(a) for a in args])
            if res["ok"]:
                result_text = res["output"] or "（脚本无输出）"
            else:
                result_text = f"脚本执行失败：{res['error']}"
                if res.get("output"):
                    result_text += f"\n已捕获输出：{res['output']}"
            return {"tool": name, "skill": skill_name, "ok": res["ok"],
                    "result": result_text,
                    "meta": {"name": skill_name, "error": res.get("error"),
                             "output_chars": len(res.get("output") or "")}}

        return {"tool": name, "skill": skill_name, "ok": False,
                "result": f"未知工具：{name}", "meta": {}}
    except Exception as e:
        logger.error(f"工具分发异常 {name}：{e}")
        return {"tool": name, "skill": skill_name, "ok": False,
                "result": f"工具执行异常：{e}", "meta": {}}
