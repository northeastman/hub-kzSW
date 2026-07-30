"""
Layer 5 能力层：在子进程中执行 skill 声明的入口脚本

教学重点：
  1. 工具调用 = 让 Agent 不止"读文本"，还能"跑代码"拿真实结果
  2. 受控执行：子进程隔离 + 硬超时 + 输出截断 + 数组传参（禁 shell=True）
  3. 降级哲学：任何执行失败都返回结构化错误、绝不抛异常，主对话链路不受影响

安全边界（教学 demo）：脚本是项目作者放入 memory/skills/ 的可信内容，
非用户上传，故不做容器/seccomp 级沙箱，仅子进程 + 超时 + 截断 + 数组传参。

使用方式：
  from src.skill_loader import SkillLoader
  from src.skill_runner import SkillRunner
  runner = SkillRunner(SkillLoader())
  res = runner.run_script("calc", ["1", "+", "2"])
  # res == {"ok": True/False, "output": "...", "error": None/"原因"}
"""

import sys
import logging
import subprocess

from src.skill_loader import SkillLoader

logger = logging.getLogger(__name__)

_TRUNC_HINT = "\n…[输出已截断]"


class SkillRunner:
    def __init__(self, loader: SkillLoader, timeout: float = 10.0, max_output: int = 8192):
        self.loader = loader
        self.timeout = timeout
        self.max_output = max_output

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_output:
            return text
        return text[: self.max_output] + _TRUNC_HINT

    def run_script(self, name: str, args: list[str] | None = None) -> dict:
        """执行 skill 入口脚本，返回 {ok, output, error}。永不抛异常。"""
        meta = self.loader.get_skill_meta(name)
        if meta is None:
            return {"ok": False, "output": "", "error": f"skill 不存在：{name}"}
        if not meta.get("run"):
            return {"ok": False, "output": "", "error": f"skill「{name}」无可执行脚本"}

        entry = meta["dir"] / meta["run"]
        if not entry.exists():
            return {"ok": False, "output": "", "error": f"入口脚本不存在：{meta['run']}"}

        cmd = [sys.executable, str(entry)] + [str(a) for a in (args or [])]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(meta["dir"]),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                # 数组传参 + 默认 shell=False，避免注入
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"skill 脚本执行超时：{name}")
            return {"ok": False, "output": "", "error": f"脚本执行超时（>{self.timeout}s）"}
        except Exception as e:
            logger.error(f"skill 脚本执行异常：{name}：{e}")
            return {"ok": False, "output": "", "error": f"执行异常：{e}"}

        stdout = self._truncate(proc.stdout or "")
        if proc.returncode != 0:
            err = (proc.stderr or "").strip() or f"非零退出码 {proc.returncode}"
            return {"ok": False, "output": stdout, "error": self._truncate(err)}
        return {"ok": True, "output": stdout, "error": None}
