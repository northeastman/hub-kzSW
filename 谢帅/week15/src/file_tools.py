"""文件工具：扫描目录 + 读文件（代码审查 subagent 项目用）
纯标准库，零网络。路径安全：只允许访问项目根目录内文件。"""
import os, logging
from pathlib import Path
logger = logging.getLogger(__name__)

CODE_EXTS = {'.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.go', '.c', '.cpp', '.h', '.hpp', '.rb', '.php', '.rs', '.swift', '.kt'}
SKIP_DIRS = {'__pycache__', '.git', 'node_modules', 'venv', '.venv', 'dist', 'build', '.next', 'target'}

def scan_directory(path: str, max_files: int = 40) -> dict:
    """递归扫描目录，返回代码文件列表。
    返回 {"root": ..., "files": [{"path", "size", "lines"}], "count"} 或 {"error": ...}"""
    try:
        root = Path(path).resolve()
        if not root.exists():
            return {"error": f"路径不存在: {path}"}
        if not root.is_dir():
            return {"error": f"不是目录: {path}"}

        files = []
        for p in root.rglob('*'):
            if not p.is_file(): continue
            if any(skip in p.parts for skip in SKIP_DIRS): continue
            if p.suffix not in CODE_EXTS: continue
            try:
                lines = len(p.read_text(encoding='utf-8', errors='ignore').splitlines())
                files.append({"path": str(p.relative_to(root)), "size": p.stat().st_size, "lines": lines})
            except Exception as e:
                logger.warning(f"跳过文件 {p}: {e}")

        files = files[:max_files]
        return {"root": str(root), "files": files, "count": len(files)}
    except Exception as e:
        return {"error": f"扫描失败: {type(e).__name__}: {str(e)[:100]}"}

def read_file(path: str, max_lines: int = 400) -> str:
    """读文件内容，带行号返回。超过 max_lines 截断并标注。"""
    try:
        p = Path(path).resolve()
        if not p.exists():
            return f"文件不存在: {path}"
        if not p.is_file():
            return f"不是文件: {path}"

        lines = p.read_text(encoding='utf-8', errors='ignore').splitlines()
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            truncated = f"\n\n（文件过长，仅显示前 {max_lines} 行）"
        else:
            truncated = ""

        numbered = "\n".join(f"{i+1:4d}│ {ln}" for i, ln in enumerate(lines))
        return numbered + truncated
    except Exception as e:
        return f"读取失败: {type(e).__name__}: {str(e)[:100]}"

def format_scan_result(r: dict) -> str:
    """把 scan 结果格式化成喂给主 agent 的文本。"""
    if "error" in r:
        return f"扫描失败: {r['error']}"
    if not r.get("files"):
        return f"目录 {r.get('root', '?')} 内无代码文件"

    parts = [f"项目根目录: {r['root']}", f"找到 {r['count']} 个代码文件:"]
    for f in r["files"]:
        parts.append(f"  - {f['path']} ({f['lines']} 行)")
    return "\n".join(parts)


if __name__ == "__main__":
    import sys, logging as _l
    _l.basicConfig(level=_l.INFO)
    # 测试扫描（用自己的 src 目录）
    test_dir = Path(__file__).parent
    r = scan_directory(str(test_dir))
    print("=== scan_directory 测试 ===")
    print(format_scan_result(r))
    # 测试读文件（读自己）
    print("\n=== read_file 测试（前 20 行）===")
    print(read_file(__file__, max_lines=20))
