"""
会话摘要记忆版 ReAct Agent（Function Calling）

设计要点（教学）：
  1. 本文件不重写 ReAct 循环，而是复用 react_function_calling.run()，
     只在外层加一层"记忆"。改 run() 不会让这里的逻辑走样。
  2. 记忆是"滚动摘要"：每轮结束后调模型，把「旧摘要 + 本轮问答」压缩成
     一段简短摘要存回进程内 dict（重启即丢失，适合教学演示）。
  3. 因为 run() 只接收一个 question 字符串，记忆通过"把摘要拼进问题前缀"
     注入。这是"不改 run()"前提下唯一的注入口，是本方案的自觉代价。

依赖：
  与 react_function_calling 相同（DASHSCOPE_API_KEY 等）
"""

import os
import logging

from react_function_calling import run, client, MODEL

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = """你是对话记忆压缩器。请把"已有摘要"和"最新一轮问答"合并成一段简短中文摘要。
要求：
- 保留关键事实：涉及的公司、股票代码、财务指标、数值结论、用户关注点
- 去掉寒暄、工具调用过程等无关细节
- 只输出摘要正文，不要加"摘要："之类前缀
"""


class SessionMemory:
    """进程内会话记忆：session_id -> 滚动摘要文本。"""

    def __init__(self):
        self._store: dict[str, str] = {}

    def get(self, session_id: str) -> str:
        """取该会话当前摘要，无则返回空串。"""
        return self._store.get(session_id, "")

    def update(self, session_id: str, question: str, answer: str) -> None:
        """调模型把「旧摘要 + 本轮问答」压缩成新摘要，覆盖存回。失败则保留旧记忆。"""
        old = self._store.get(session_id, "")
        user_content = (
            f"已有摘要：\n{old or '（无）'}\n\n"
            f"最新一轮：\n用户问：{question}\n助手答：{answer}\n\n"
            f"请输出更新后的摘要："
        )
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0,
            )
            summary = (resp.choices[0].message.content or "").strip()
            if summary:
                self._store[session_id] = summary
        except Exception as e:  # noqa: BLE001 教学演示，记忆更新失败不影响本轮答案
            logger.warning("摘要更新失败，保留旧记忆: %s", e)

    def clear(self, session_id: str) -> None:
        """清空指定会话记忆，对不存在的 session_id 幂等。"""
        self._store.pop(session_id, None)


# 模块级单例：serve.py 与 CLI 共用同一份记忆
_MEMORY = SessionMemory()


def run_with_memory(question: str, session_id: str = "default", max_steps: int = 10):
    """
    带记忆的 ReAct 循环：取会话摘要 → 拼进问题前缀 → 复用 run() 逐步透传 → 结束后更新摘要。

    yield 的 dict 与 react_function_calling.run() 完全一致，
    因此 serve.py / 前端可无差别复用现有渲染逻辑。
    """
    memory = _MEMORY.get(session_id)

    augmented = question
    if memory:
        augmented = f"已知对话背景：{memory}\n\n用户当前问题：{question}"

    answer = ""
    for step in run(augmented, max_steps=max_steps):
        if step.get("type") == "final":
            answer = step.get("answer", "")
        yield step   # 逐步透传给调用方（serve/CLI），格式不变

    # 循环结束后，用原始 question（不含背景前缀）+ 本轮答案更新滚动摘要
    if answer:
        _MEMORY.update(session_id, question, answer)


def clear_session(session_id: str) -> None:
    """清空指定会话记忆（供 serve.py 的 /session/clear 路由调用）。"""
    _MEMORY.clear(session_id)


# ── CLI 多轮 REPL 演示 ────────────────────────────────────────────────────────
def _run_and_print_once(question: str, session_id: str, max_steps: int = 10):
    """复用 react_function_calling 的彩色输出风格，跑一轮带记忆问答。"""
    import time
    from react_function_calling import _c

    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print(f"模型: {MODEL}  实现: Function Calling + 会话记忆")
    print('='*60)

    start = time.time()
    for step_data in run_with_memory(question, session_id=session_id, max_steps=max_steps):
        stype = step_data["type"]
        if stype == "action":
            print(f"\n[Step {step_data['step']}]")
            print(_c("action", f"🔧 Action:  {step_data['action']}"))
            print(_c("action", f"   Input:   {step_data['action_input']}"))
            print(_c("obs",    f"👁  Obs:     {str(step_data['observation'])[:300]}"))
        elif stype == "final":
            elapsed = time.time() - start
            print(_c("final", f"\n✅ Final Answer:\n{step_data['answer']}"))
            print(f"\n耗时 {elapsed:.1f}s")
        elif stype in ("error", "max_steps"):
            print(_c("error", f"\n⚠️  {step_data.get('answer', '')}"))


if __name__ == "__main__":
    SESSION = "cli"
    print("会话记忆版 ReAct Agent（输入 exit 退出，clear 清空记忆）")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() == "exit":
            break
        if q.lower() == "clear":
            clear_session(SESSION)
            print("（已清空本会话记忆）")
            continue
        _run_and_print_once(q, session_id=SESSION)
