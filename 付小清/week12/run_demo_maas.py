"""
使用 MaaS 工作空间 API 运行 demo，输出写入 demo_output.txt

环境变量（运行前设置，勿写入代码）：
  DASHSCOPE_API_KEY
  AGENT_BASE_URL   默认见下方 MAAS_BASE_URL
  AGENT_MODEL      默认 qwen3.7-plus
"""

import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

MAAS_BASE_URL = "https://ws-an0d0vqov4zjj1qx.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.7-plus"

AGENT_SRC = Path(__file__).parent.parent / "react_financial_agent" / "src"
sys.path.insert(0, str(AGENT_SRC))
sys.path.insert(0, str(Path(__file__).parent))


def patch_llm_client():
    from openai import OpenAI

    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    base_url = os.environ.get("AGENT_BASE_URL", MAAS_BASE_URL)
    model = os.environ.get("AGENT_MODEL", DEFAULT_MODEL)

    if not api_key:
        print("错误：请设置 DASHSCOPE_API_KEY", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=base_url)

    import react_manual
    react_manual.client = client
    react_manual.MODEL = model

    return model, base_url


def main():
    model, base_url = patch_llm_client()

    from run_multi_turn import run_demo as run_multi_demo
    from run_single_turn import run_demo as run_single_demo

    buf = io.StringIO()
    with redirect_stdout(buf):
        print("=" * 70)
        print("Week12 运行记录")
        print(f"API Host : {base_url}")
        print(f"Model    : {model}")
        print(f"Workspace: ws-an0d0vqov4zjj1qx (默认业务空间 / ragTest)")
        print("=" * 70)

        print("\n\n" + "#" * 70)
        print("# 实验 A：多轮对话模式 (run_multi_turn.py --demo)")
        print("#" * 70)
        run_multi_demo("manual", 10)

        print("\n\n" + "#" * 70)
        print("# 实验 B：单轮对照模式 (run_single_turn.py --demo)")
        print("#" * 70)
        run_single_demo("manual", 10)

    text = buf.getvalue()
    out_path = Path(__file__).parent / "demo_output.txt"
    out_path.write_text(text, encoding="utf-8")
    # Windows 控制台可能无法打印 emoji，仅写文件
    try:
        print(text)
    except UnicodeEncodeError:
        print(f"输出含 Unicode 字符，已写入 {out_path}")
    else:
        print(f"\n已写入 {out_path}")


if __name__ == "__main__":
    main()
