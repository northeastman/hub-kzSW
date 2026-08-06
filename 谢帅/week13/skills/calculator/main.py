"""计算器 skill 入口脚本：把命令行参数拼成表达式并安全求值。"""
import sys
import ast
import operator

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.USub: operator.neg, ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("不支持的表达式")


def main():
    expr = " ".join(sys.argv[1:]).strip()
    if not expr:
        print("错误：未提供表达式")
        return
    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval(tree.body)
        print(f"{expr} = {result}")
    except Exception as e:
        print(f"计算失败：{e}")


if __name__ == "__main__":
    main()
