"""
skill_executor.py — ReAct 风格的 Skill 执行器

执行流程：
  1. 把命中的 skill 升级到 Level 2（FULL）+ Level 3（ASSETS）
  2. 用 SKILL.md 正文作为系统提示，告诉 LLM 可调用的工具
  3. LLM 每轮输出一个工具调用 JSON，执行后把结果喂回，循环直到 finish
  4. 全程通过 broadcaster 推送 SSE 事件，前端可见每一步

工具集（手动 function calling，避免依赖各家 SDK 的差异）：
  write_file(path, content)      —— 写文件（路径相对 skill 目录或工作目录）
  run_script(command)            —— 执行 shell 命令（在 skill 根目录下）
  finish(summary)                —— 结束并返回给用户的最终摘要

安全：run_script 只允许在 skill 目录或工作目录下执行，且命令可见可审计。
"""

from __future__ import annotations
import os
import re
import json
import time
import shlex
import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Callable, Awaitable

from .skill_registry import SkillRegistry, SkillEntry
from .llm_config import get_chat_client
from .memory_store import MemoryStore

logger = logging.getLogger(__name__)

Broadcaster = Callable[[str, dict], Awaitable[None]]

# 最大 ReAct 循环轮数，防止失控
MAX_STEPS = 10

_SYSTEM_TEMPLATE = """你正在执行一个名为「{skill_name}」的 Skill。下面是它的完整说明文档：

==================== SKILL.md ====================
{skill_body}
==================== SKILL.md ====================

你可以使用以下工具完成任务。每一轮你必须输出一个工具调用，格式为严格的 JSON（不要代码块包裹）：

{tools_doc}

规则：
- 每轮只调一个工具
- 路径若是相对路径，相对于工作目录：{work_dir}
- 完成任务后必须调用 finish 工具返回结果
- 最多 {max_steps} 轮，超时未 finish 会被强制结束
- 不要输出 JSON 以外的内容"""

_TOOLS_DOC = """1. write_file —— 写文件
   参数：{{"tool": "write_file", "path": "<相对路径>", "content": "<文件内容>"}}

2. run_script —— 执行命令（在 skill 目录 {skill_dir} 下运行）
   参数：{{"tool": "run_script", "command": "<shell 命令>"}}

3. finish —— 完成任务，返回最终摘要给用户
   参数：{{"tool": "finish", "summary": "<给用户的最终回复>"}}"""


class SkillExecutor:
    def __init__(
        self,
        registry: SkillRegistry,
        memory: MemoryStore,
        work_dir: str | Path,
    ):
        self.registry = registry
        self.memory = memory
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)

    async def execute(
        self,
        skill_name: str,
        user_input: str,
        broadcaster: Broadcaster | None = None,
    ) -> dict:
        """
        执行指定 skill。返回 {"status", "summary", "steps", "duration_ms"}。
        """
        t0 = time.time()

        # Step 1: 渐进式升级 —— Level 2 + Level 3
        if broadcaster:
            await broadcaster("exec_start", {"skill": skill_name, "input": user_input})

        entry = self.registry.load_full(skill_name)
        if entry is None or entry.level < 2:
            msg = f"skill '{skill_name}' 不存在或加载失败"
            if broadcaster:
                await broadcaster("exec_error", {"error": msg})
            return {"status": "failed", "summary": msg, "steps": 0, "duration_ms": 0}

        if broadcaster:
            await broadcaster(
                "exec_level_up",
                {"skill": skill_name, "level": "FULL(2)", "body_chars": len(entry.body)},
            )

        entry = self.registry.load_assets(skill_name)
        if broadcaster:
            await broadcaster(
                "exec_level_up",
                {
                    "skill": skill_name,
                    "level": "ASSETS(3)",
                    "scripts": [str(p) for p in entry.scripts],
                    "references": [str(p) for p in entry.references],
                    "data_files": [str(p) for p in entry.data_files],
                },
            )

        # Step 2: 记录开始
        usage_id = self.memory.record_start(skill_name, user_input)

        # Step 3: ReAct 循环
        system_prompt = _SYSTEM_TEMPLATE.format(
            skill_name=skill_name,
            skill_body=entry.body,
            tools_doc=_TOOLS_DOC.format(skill_dir=entry.path),
            work_dir=self.work_dir,
            max_steps=MAX_STEPS,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"用户请求：{user_input}"},
        ]

        steps = 0
        final_summary = ""
        status = "failed"

        try:
            client, model = get_chat_client()
        except Exception as e:
            err = f"LLM 初始化失败：{e}"
            self.memory.record_finish(usage_id, "failed", err, int((time.time()-t0)*1000))
            if broadcaster:
                await broadcaster("exec_error", {"error": err})
            return {"status": "failed", "summary": err, "steps": 0, "duration_ms": int((time.time()-t0)*1000)}

        for step in range(1, MAX_STEPS + 1):
            steps = step
            if broadcaster:
                await broadcaster("exec_step", {"step": step, "phase": "thinking"})

            # LLM 决策
            try:
                raw = await asyncio.to_thread(self._call_llm, client, model, messages)
            except Exception as e:
                err = f"LLM 调用失败（step {step}）：{e}"
                logger.error(err)
                if broadcaster:
                    await broadcaster("exec_error", {"error": err, "step": step})
                break

            tool_call = self._parse_tool_call(raw)
            if tool_call is None:
                # LLM 没输出有效 JSON，把原文回灌让它修正
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": "上轮输出不是合法的工具调用 JSON，请重新输出严格的 JSON。",
                })
                if broadcaster:
                    await broadcaster("exec_step", {"step": step, "phase": "parse_failed", "raw": raw[:200]})
                continue

            tool = tool_call.get("tool", "")
            messages.append({"role": "assistant", "content": json.dumps(tool_call, ensure_ascii=False)})

            if broadcaster:
                await broadcaster(
                    "exec_tool_call",
                    {"step": step, "tool": tool, "args": {k: v for k, v in tool_call.items() if k != "tool"}},
                )

            # 执行工具
            if tool == "finish":
                final_summary = tool_call.get("summary", "")
                status = "success"
                if broadcaster:
                    await broadcaster("exec_finish", {"step": step, "summary": final_summary})
                break
            elif tool == "write_file":
                obs = self._tool_write_file(tool_call, entry)
            elif tool == "run_script":
                obs = await self._tool_run_script(tool_call, entry, broadcaster, step)
            else:
                obs = f"未知工具：{tool}，可选：write_file / run_script / finish"

            messages.append({"role": "user", "content": f"工具执行结果：\n{obs}"})
            if broadcaster:
                await broadcaster("exec_tool_result", {"step": step, "observation": obs[:500]})

        else:
            # for 循环正常结束（未 break）—— 达到最大轮数
            final_summary = f"达到最大轮数 {MAX_STEPS}，强制结束。"
            status = "timeout"

        duration_ms = int((time.time() - t0) * 1000)
        self.memory.record_finish(usage_id, status, final_summary[:500], duration_ms)

        result = {
            "status": status,
            "summary": final_summary,
            "steps": steps,
            "duration_ms": duration_ms,
            "skill": skill_name,
        }
        if broadcaster:
            await broadcaster("exec_done", result)
        return result

    # ── LLM 调用 ────────────────────────────────────────────────────────
    def _call_llm(self, client, model: str, messages: list[dict]) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )
        return resp.choices[0].message.content or ""

    @staticmethod
    def _parse_tool_call(text: str) -> dict | None:
        """从 LLM 输出中抽取第一个 JSON 对象。"""
        text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
        text = re.sub(r"\n?```\s*$", "", text)
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            if "tool" in obj:
                return obj
        except json.JSONDecodeError:
            pass
        return None

    # ── 工具实现 ────────────────────────────────────────────────────────
    def _tool_write_file(self, call: dict, entry: SkillEntry) -> str:
        path_str = call.get("path", "")
        content = call.get("content", "")
        if not path_str:
            return "错误：缺少 path 参数"
        # 路径解析：相对工作目录，但禁止越狱到工作目录之外
        target = (self.work_dir / path_str).resolve()
        try:
            target.relative_to(self.work_dir)
        except ValueError:
            return f"错误：路径越界，必须在工作目录内：{self.work_dir}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        logger.info(f"[exec] write_file -> {target} ({len(content)} 字符)")
        return f"已写入 {target.relative_to(self.work_dir)}（{len(content)} 字符）"

    async def _tool_run_script(
        self,
        call: dict,
        entry: SkillEntry,
        broadcaster: Broadcaster | None,
        step: int,
    ) -> str:
        command = call.get("command", "")
        if not command:
            return "错误：缺少 command 参数"
        if broadcaster:
            await broadcaster("exec_script_start", {"step": step, "command": command})

        # 在 skill 目录下执行，让脚本能用相对路径找到 data/references
        cwd = str(entry.path)
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                shell=True,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            out = stdout.decode("utf-8", errors="replace")
            err = stderr.decode("utf-8", errors="replace")
            result = f"exit_code={proc.returncode}\nstdout:\n{out}"
            if err:
                result += f"\nstderr:\n{err}"
            logger.info(f"[exec] run_script '{command}' -> exit={proc.returncode}")
            return result[:4000]  # 截断，避免 token 爆炸
        except asyncio.TimeoutError:
            return "错误：脚本执行超时（120s）"
        except Exception as e:
            return f"错误：执行异常 {e}"
