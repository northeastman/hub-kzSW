"""Agent 可调用工具及其安全边界。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.skill_registry import SkillError, SkillRegistry


@dataclass
class TurnState:
    """只在当前用户轮次存活，轮次完成后整体丢弃。"""

    workspace: Path
    loaded_skills: list[str] = field(default_factory=list)
    loaded_resources: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    written_files: list[str] = field(default_factory=list)


class ToolRegistry:
    def __init__(
        self,
        skills: SkillRegistry,
        workspace: Path,
        *,
        max_write_chars: int = 500_000,
        script_timeout_seconds: int = 30,
    ) -> None:
        self.skills = skills
        self.workspace = workspace.resolve()
        self.max_write_chars = max_write_chars
        self.script_timeout_seconds = script_timeout_seconds
        self._schemas = self._build_schemas()

    def schemas_for(self, state: TurnState) -> list[dict[str, Any]]:
        """初始只暴露 load_skill；Skill 激活后再开放其声明的工具。"""
        enabled = {"load_skill"}
        for skill_name in state.loaded_skills:
            enabled.update(self.skills.get(skill_name).allowed_tools)
        return [
            schema
            for name, schema in self._schemas.items()
            if name in enabled
        ]

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        state: TurnState,
    ) -> str:
        state.tool_calls.append(name)
        try:
            if name == "load_skill":
                return self._load_skill(arguments, state)

            enabled = {
                schema["function"]["name"]
                for schema in self.schemas_for(state)
            }
            if name not in enabled:
                return f"工具未启用：{name}。请先加载允许该工具的 Skill。"

            handlers = {
                "read_skill_resource": self._read_skill_resource,
                "write_workspace_file": self._write_workspace_file,
                "run_skill_script": self._run_skill_script,
            }
            handler = handlers.get(name)
            if handler is None:
                return f"未知工具：{name}"
            return handler(arguments, state)
        except (
            OSError,
            SkillError,
            ValueError,
            TypeError,
            subprocess.TimeoutExpired,
        ) as exc:
            return f"工具执行失败：{exc}"

    def _load_skill(
        self,
        arguments: dict[str, Any],
        state: TurnState,
    ) -> str:
        name = self._required_string(arguments, "name")
        if name in state.loaded_skills:
            return f"Skill {name!r} 已在当前轮加载，无需重复加载。"

        content = self.skills.load_skill(name)
        state.loaded_skills.append(name)
        meta = self.skills.get(name)
        tools = ", ".join(meta.allowed_tools) or "无"
        return (
            f"Skill {name!r} 已加载，仅在当前用户轮次有效。\n"
            f"本 Skill 允许的工具：{tools}\n\n"
            f"<skill_definition name={json.dumps(name, ensure_ascii=False)}>\n"
            f"{content}\n"
            "</skill_definition>"
        )

    def _read_skill_resource(
        self,
        arguments: dict[str, Any],
        state: TurnState,
    ) -> str:
        skill_name = self._require_active_skill(arguments, state)
        relative_path = self._required_string(arguments, "path")
        self._require_skill_permission(skill_name, "read_skill_resource")
        content = self.skills.read_resource(skill_name, relative_path)
        marker = f"{skill_name}/{relative_path}"
        if marker not in state.loaded_resources:
            state.loaded_resources.append(marker)
        return (
            f"<skill_resource path={json.dumps(marker, ensure_ascii=False)}>\n"
            f"{content}\n"
            "</skill_resource>"
        )

    def _write_workspace_file(
        self,
        arguments: dict[str, Any],
        state: TurnState,
    ) -> str:
        skill_name = self._require_active_skill(arguments, state)
        self._require_skill_permission(skill_name, "write_workspace_file")
        relative_path = self._required_string(arguments, "path")
        content = self._required_string(arguments, "content", allow_empty=True)
        if len(content) > self.max_write_chars:
            raise ValueError(
                f"写入内容过大：{len(content)} 字符，上限 {self.max_write_chars}"
            )

        target = SkillRegistry.resolve_inside(self.workspace, relative_path)
        self._require_allowed_write_target(skill_name, target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        display_path = str(target.relative_to(self.workspace))
        if display_path not in state.written_files:
            state.written_files.append(display_path)
        return f"已写入 {display_path}（{len(content)} 字符）"

    def _run_skill_script(
        self,
        arguments: dict[str, Any],
        state: TurnState,
    ) -> str:
        skill_name = self._require_active_skill(arguments, state)
        self._require_skill_permission(skill_name, "run_skill_script")
        script_name = self._required_string(arguments, "script")
        meta = self.skills.get(skill_name)
        scripts_root = (meta.directory / "scripts").resolve()
        script_path = SkillRegistry.resolve_inside(scripts_root, script_name)
        if not script_path.is_file():
            raise ValueError(f"脚本不存在：{script_name}")
        if script_path.suffix != ".py":
            raise ValueError("当前只允许运行 Skill scripts/ 下的 Python 脚本")

        raw_args = arguments.get("args", [])
        if not isinstance(raw_args, list):
            raise TypeError("args 必须是字符串列表")
        args = [str(item) for item in raw_args]
        self._validate_script_args(args)

        completed = subprocess.run(
            [sys.executable, str(script_path), *args],
            cwd=self.workspace,
            env=self._safe_script_env(),
            capture_output=True,
            text=True,
            timeout=self.script_timeout_seconds,
            check=False,
        )
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        output_parts = [f"exit_code={completed.returncode}"]
        if stdout:
            output_parts.append(f"stdout:\n{stdout[:6000]}")
        if stderr:
            output_parts.append(f"stderr:\n{stderr[:3000]}")
        return "\n".join(output_parts)

    def _require_active_skill(
        self,
        arguments: dict[str, Any],
        state: TurnState,
    ) -> str:
        skill_name = self._required_string(arguments, "skill_name")
        if skill_name not in state.loaded_skills:
            raise ValueError(f"Skill {skill_name!r} 尚未在当前轮加载")
        return skill_name

    def _require_skill_permission(self, skill_name: str, tool_name: str) -> None:
        meta = self.skills.get(skill_name)
        if tool_name not in meta.allowed_tools:
            raise ValueError(
                f"Skill {skill_name!r} 未声明工具权限 {tool_name!r}"
            )

    def _require_allowed_write_target(
        self,
        skill_name: str,
        target: Path,
    ) -> None:
        """只允许写公共产物目录或当前 Skill 的 data/，保护源码与配置。"""
        meta = self.skills.get(skill_name)
        allowed_roots = (
            (self.workspace / "outputs").resolve(),
            (self.workspace / "diagram").resolve(),
            (meta.directory / "data").resolve(),
        )
        for root in allowed_roots:
            try:
                target.relative_to(root)
                return
            except ValueError:
                continue
        allowed = "、".join(str(path) for path in allowed_roots)
        raise ValueError(f"拒绝写入受保护位置；允许的目录：{allowed}")

    @staticmethod
    def _required_string(
        arguments: dict[str, Any],
        key: str,
        *,
        allow_empty: bool = False,
    ) -> str:
        value = arguments.get(key)
        if not isinstance(value, str):
            raise TypeError(f"参数 {key!r} 必须是字符串")
        if not allow_empty and not value.strip():
            raise ValueError(f"参数 {key!r} 不能为空")
        return value

    @staticmethod
    def _validate_script_args(args: list[str]) -> None:
        for value in args:
            if "\x00" in value:
                raise ValueError("脚本参数不能包含 NUL 字符")
            if value.startswith("-"):
                if ".." in value or "/" in value or "\\" in value:
                    raise ValueError(f"脚本选项中不能内嵌路径：{value}")
                _, separator, attached_value = value.partition("=")
                if separator:
                    ToolRegistry._validate_relative_arg(attached_value)
                continue
            ToolRegistry._validate_relative_arg(value)

    @staticmethod
    def _validate_relative_arg(value: str) -> None:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"脚本参数包含越界路径：{value}")

    @staticmethod
    def _safe_script_env() -> dict[str, str]:
        """脚本默认拿不到父进程中的 API Key 等秘密。"""
        allowed_names = ("PATH", "LANG", "LC_ALL", "TMPDIR")
        env = {
            name: os.environ[name]
            for name in allowed_names
            if name in os.environ
        }
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    @staticmethod
    def _build_schemas() -> dict[str, dict[str, Any]]:
        schemas = [
            {
                "type": "function",
                "function": {
                    "name": "load_skill",
                    "description": (
                        "按名称加载完整 Skill 指令。用户请求与 Skill 索引匹配时，"
                        "必须先调用此工具，再执行 Skill。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Skill 索引中的精确名称",
                            }
                        },
                        "required": ["name"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_skill_resource",
                    "description": (
                        "按需读取已加载 Skill 目录中的 references、data 等文本资源。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "skill_name": {"type": "string"},
                            "path": {
                                "type": "string",
                                "description": "相对于 Skill 目录的资源路径",
                            },
                        },
                        "required": ["skill_name", "path"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_workspace_file",
                    "description": (
                        "写入 UTF-8 文本；路径只能位于 outputs/、diagram/，"
                        "或当前 Skill 的 data/。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "skill_name": {"type": "string"},
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["skill_name", "path", "content"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_skill_script",
                    "description": (
                        "运行已加载 Skill 的 scripts/ 目录下的 Python 脚本；"
                        "不经过 shell。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "skill_name": {"type": "string"},
                            "script": {
                                "type": "string",
                                "description": "相对于 scripts/ 的 Python 文件名",
                            },
                            "args": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["skill_name", "script", "args"],
                        "additionalProperties": False,
                    },
                },
            },
        ]
        return {item["function"]["name"]: item for item in schemas}
