from __future__ import annotations

import argparse
import uuid

from knowledge_assistant.core.settings import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Knowledge Assistant CLI")
    parser.add_argument("--config", default=None, help="Path to knowledge_assistant.toml")
    parser.add_argument("--session-id", default=None, help="Reuse an existing conversation session")
    parser.add_argument("--source", default=None, help="Optional source filter, such as ai/java/ops")
    args = parser.parse_args()

    settings = load_settings(args.config)
    session_id = args.session_id or str(uuid.uuid4())
    from knowledge_assistant.factory import build_orchestrator

    orchestrator = build_orchestrator(settings)

    print("\n欢迎使用 Knowledge Assistant 命令行问答系统")
    print(f"会话ID: {session_id}")
    print(f"可用知识来源: {', '.join(settings.app.valid_sources)}")
    print("输入问题开始问答；输入 /history 查看历史，/clear 清空历史，/exit 退出。\n")

    source_filter = args.source
    while True:
        query = input("你: ").strip()
        if not query:
            continue
        if query == "/exit":
            print("再见。")
            break
        if query == "/history":
            for index, item in enumerate(orchestrator.get_history(session_id), 1):
                print(f"{index}. 问: {item['question']}\n   答: {item['answer']}")
            continue
        if query == "/clear":
            orchestrator.clear_history(session_id)
            print("历史记录已清空。")
            continue

        print("助手: ", end="", flush=True)
        for token in orchestrator.stream_answer(query, session_id=session_id, source_filter=source_filter):
            print(token, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    main()
