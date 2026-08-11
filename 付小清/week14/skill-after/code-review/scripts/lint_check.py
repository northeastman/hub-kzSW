#!/usr/bin/env python3
"""Run quick static checks on a Python file. Used by code-review skill."""

import ast
import sys
from pathlib import Path

ISSUES: list[str] = []


def check_file(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            _check_mutable_defaults(node, path)
            _check_bare_except_in(node, path)
        if isinstance(node, ast.Call):
            _check_dangerous_calls(node, path, source)

    return ISSUES


def _check_mutable_defaults(node: ast.FunctionDef, path: Path) -> None:
    for default in node.args.defaults + node.args.kw_defaults:
        if default and isinstance(default, (ast.List, ast.Dict, ast.Set)):
            ISSUES.append(f"{path}:{node.lineno} mutable default argument in `{node.name}`")


def _check_bare_except_in(node: ast.FunctionDef, path: Path) -> None:
    for child in ast.walk(node):
        if isinstance(child, ast.ExceptHandler) and child.type is None:
            ISSUES.append(f"{path}:{child.lineno} bare `except:` in `{node.name}`")


def _check_dangerous_calls(node: ast.Call, path: Path, source: str) -> None:
    if isinstance(node.func, ast.Attribute):
        if node.func.attr == "execute" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.JoinedStr):  # f-string SQL
                ISSUES.append(f"{path}:{node.lineno} possible SQL injection (f-string in execute)")
    if isinstance(node.func, ast.Name) and node.func.id == "eval":
        ISSUES.append(f"{path}:{node.lineno} use of eval()")


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python lint_check.py <file.py>", file=sys.stderr)
        return 1

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    issues = check_file(path)
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    print("OK: no common issues detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
