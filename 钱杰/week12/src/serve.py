"""
FastAPI HTTP 服务 —— 多轮对话版

接口：
  POST   /session/new              新建会话，返回 session_id
  GET    /session/{sid}/history    获取会话历史
  DELETE /session/{sid}            清空会话
  GET    /session                  列出所有会话
  POST   /chat/manual              手写版多轮 ReAct，SSE 流式返回每步
  POST   /chat/fc                  Function Calling 版多轮 ReAct，SSE 流式返回每步
  GET    /health                   健康检查
  GET    /                         托管 index.html

使用方式：
  uvicorn serve:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import json
import logging
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

# 把当前目录加入 sys.path，方便 import 同级模块
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── 预加载 FAISS ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("预加载 FAISS 索引...")
    from tools import _load_rag  # 复用原项目 tools.py
    await asyncio.to_thread(_load_rag)
    logger.info("预加载完成，服务就绪")
    yield


app = FastAPI(title="ReAct Financial Agent (Multi-Turn)", lifespan=lifespan)


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    question:   str
    max_steps:  int = 10


class NewSessionResponse(BaseModel):
    session_id: str
    created_at: float


# ── SSE 工具 ──────────────────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── 会话路由 ──────────────────────────────────────────────────────────────────
from session import session_manager


@app.post("/session/new", response_model=NewSessionResponse)
async def new_session():
    """新建一个会话"""
    import time
    sid = session_manager.create_session()
    return {"session_id": sid, "created_at": time.time()}


@app.get("/session/{sid}/history")
async def get_history(sid: str):
    """获取会话历史"""
    history = session_manager.get_history(sid)
    if not history and sid not in [s["session_id"] for s in session_manager.list_sessions()]:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": sid, "history": history, "turns": len(history) // 2}


@app.delete("/session/{sid}")
async def clear_session(sid: str):
    """清空指定会话"""
    ok = session_manager.clear_session(sid)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True, "session_id": sid}


@app.get("/session")
async def list_sessions():
    """列出所有会话"""
    return {"sessions": session_manager.list_sessions()}


# ── 聊天路由（SSE 流式）───────────────────────────────────────────────────────
async def _stream_chat(question: str, max_steps: int, mode: str, session_id: str):
    """
    多轮对话核心：
      1. 从 session_manager 取出历史
      2. 在独立线程中跑 ReAct，每步通过 queue 推给 SSE
      3. 拿到 Final Answer 后，把本轮 Q&A 写回 session
    """
    if mode == "manual":
        from react_manual_mt import run as react_run
    else:
        from react_function_calling_mt import run as react_run

    # 取历史（空列表也无所谓）
    history = session_manager.get_history(session_id)

    queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    # 在线程中跑同步的 ReAct 循环
    def _worker():
        final_answer = ""
        try:
            for step_data in react_run(question, max_steps=max_steps, history=history):
                queue.put_nowait(step_data)
                if step_data.get("type") == "final":
                    final_answer = step_data.get("answer", "")
        except Exception as e:
            queue.put_nowait({
                "type": "error",
                "step": 0,
                "message": f"服务端错误: {e}",
            })
        finally:
            # 把本轮问答存入会话历史（即使失败也存空答案，便于后续诊断）
            if final_answer:
                session_manager.add_turn(session_id, question, final_answer)
            queue.put_nowait(_SENTINEL)

    yield _sse({
        "type": "start",
        "session_id": session_id,
        "question": question,
        "mode": mode,
        "history_turns": len(history) // 2,
    })

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _worker)

    while True:
        step_data = await queue.get()
        if step_data is _SENTINEL:
            break
        yield _sse(step_data)

    yield _sse({"type": "done"})


@app.post("/chat/manual")
async def chat_manual(req: ChatRequest):
    """手写版多轮 ReAct"""
    return StreamingResponse(
        _stream_chat(req.question, req.max_steps, "manual", req.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat/fc")
async def chat_fc(req: ChatRequest):
    """Function Calling 版多轮 ReAct"""
    return StreamingResponse(
        _stream_chat(req.question, req.max_steps, "fc", req.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": os.getenv("AGENT_MODEL", "qwen-max")}


# ── 托管 index.html ──────────────────────────────────────────────────────────
HTML_PATH = Path(__file__).parent.parent / "index.html"


@app.get("/")
async def root():
    if HTML_PATH.exists():
        return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>index.html not found</h2>")


# ── 启动入口 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=False)
