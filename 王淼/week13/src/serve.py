"""
FastAPI HTTP 服务 — 带 SSE 事件流的可视化后端

教学重点：
  1. SSE 事件流将四层记忆的每个步骤实时推送到前端
  2. /flush 接口展示 Memory Flush 三个 Pass 的进度
  3. lifespan 模式：索引/DB 在启动时加载一次，请求间复用

使用方式：
  uvicorn src.serve:app --host 0.0.0.0 --port 8000

接口：
  POST /chat     SSE 流式对话（含四层记忆事件）
  POST /flush    SSE 流式 Memory Flush
  GET  /memories 查看当前记忆状态
  GET  /health   健康检查

依赖：
  pip install fastapi uvicorn openai faiss-cpu
  export DASHSCOPE_API_KEY="sk-xxx"
"""

import os
import sys
import json
import sqlite3
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncIterator

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.session_db import SessionDB
from src.memory_loader import MemoryLoader
from src.vector_store import VectorStore
from src.fts_store import FTSStore
from src.retrieval import HybridRetriever
from src.memory_flush import MemoryFlusher
from src.llm_config import get_chat_client, current_model_info
from src.heartbeat_parser import HeartbeatParser
from src.scheduler import HeartbeatScheduler
from src.skill_harness import Harness, SkillContext

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── 全局单例 ─────────────────────────────────────────────────────────────────

db: SessionDB = None
loader: MemoryLoader = None
vs: VectorStore = None
fts: FTSStore = None
retriever: HybridRetriever = None
flusher: MemoryFlusher = None
harness: Harness = None
current_session_id: int = None
hb_parser: HeartbeatParser = None
hb_scheduler: HeartbeatScheduler = None

# ── SSE 广播：每个连接一个 Queue，调度器触发时 broadcast 推给所有连接 ──────────
_stream_listeners: list[asyncio.Queue] = []


async def broadcast(event_type: str, data: dict):
    payload = sse_event(event_type, data)
    logger.info(f"[broadcast] {event_type}，当前监听数：{len(_stream_listeners)}")
    for q in list(_stream_listeners):  # 拷贝一份，避免迭代时被修改
        try:
            await q.put(payload)
        except asyncio.QueueFull:
            logger.warning("[broadcast] 队列已满，丢弃一条消息")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db, loader, vs, fts, retriever, flusher, harness, current_session_id, hb_parser, hb_scheduler
    db = SessionDB()
    loader = MemoryLoader()
    vs = VectorStore()
    fts = FTSStore()
    retriever = HybridRetriever(vs, fts)
    flusher = MemoryFlusher()
    harness = Harness(db, loader, retriever, flusher)
    harness.register_all_skills()
    hb_parser = HeartbeatParser()
    current_session_id = db.new_session()
    logger.info(f"服务启动，会话 #{current_session_id}")
    logger.info(f"FTS5/BM25 可用：{fts.available}（{'混合检索' if fts.available else '退化为纯向量'})")

    hb_scheduler = HeartbeatScheduler()
    hb_scheduler.start(broadcast)
    logger.info("HEARTBEAT 调度器已启动")

    yield

    hb_scheduler.stop()
    if current_session_id:
        db.close_session(current_session_id)


app = FastAPI(title="Agent 记忆系统", lifespan=lifespan)

# ── 请求/响应模型 ──────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: int = None  # None = 使用当前会话


class FlushRequest(BaseModel):
    session_id: int = None


# ── SSE 工具函数 ──────────────────────────────────────────────────────────────

def sse_event(event_type: str, data: dict) -> str:
    payload = json.dumps({"type": event_type, **data}, ensure_ascii=False)
    return f"data: {payload}\n\n"


# ── /chat 接口（Harness 渐进式 Skill 加载）──────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest):
    sid = req.session_id or current_session_id

    async def stream():
        loop = asyncio.get_event_loop()

        # ── 调用 Harness：渐进式加载激活的 Skill ──────────────────────
        last_ctx = None
        async for event in _harness_stream(req.message, sid, loop):
            # ── 将 Harness 事件映射为前端期望的 SSE 事件类型 ──────────
            event_type = event.get("type", "")
            data = event

            if event_type == "skill_activate":
                # 前端可选择性显示 Skill 激活（新事件类型，前端忽略即可）
                yield sse_event("skill_activate", {
                    "name": event["name"],
                    "description": event["description"],
                })
                continue

            if event_type == "skill_result":
                skill_name = event.get("name", "")
                skill_data = event.get("data", {})

                if skill_name == "memory_load":
                    yield sse_event("memory_load", skill_data)
                elif skill_name == "semantic_search":
                    yield sse_event("semantic_search", skill_data)
                elif skill_name == "session_history":
                    # session_history 的结果在 context_assembly 中一起推送
                    pass
                elif skill_name == "context_assembly":
                    yield sse_event("context_assembly", skill_data)
                elif skill_name == "save_message":
                    pass  # done 事件中已包含 message_count
                elif skill_name == "word_card":
                    yield sse_event("word_card", skill_data)
                continue

            if event_type == "token":
                yield sse_event("token", {"text": event["text"]})
                continue

            if event_type == "done":
                yield sse_event("done", {
                    "response": event["response"],
                    "session_id": event["session_id"],
                    "message_count": event.get("message_count", 0),
                    "active_skills": event.get("active_skills", []),
                    "auto_flush_threshold": 20,
                })
                # 保存 ctx 供后续 heartbeat 检测
                last_ctx = event
                continue

            # 透传未知事件
            yield sse_event(event_type, data)

        # ── 后台检测调度意图（不阻塞响应）───────────────────────────
        if hb_parser and hb_parser.may_contain_schedule_intent(req.message):
            asyncio.create_task(_check_schedule_intent(req.message))

    return StreamingResponse(stream(), media_type="text/event-stream")


async def _harness_stream(user_input: str, session_id: int, loop) -> AsyncIterator[dict]:
    """
    包装 harness.run_stream() 为 async generator。
    Harness 是同步生成器，通过逐次 run_in_executor(next) 桥接到 async 世界，
    保证每个事件都能实时推送到 SSE 流，而非等全部完成后再推送。
    """
    gen = harness.run_stream(user_input, session_id)

    def _safe_next():
        try:
            return next(gen)
        except StopIteration:
            return None

    while True:
        event = await loop.run_in_executor(None, _safe_next)
        if event is None:
            break
        yield event


async def _check_schedule_intent(message: str):
    """后台任务：检测新建/取消调度意图，更新 HEARTBEAT.md 并重载调度器"""
    loop = asyncio.get_event_loop()

    # 优先检测取消意图（取消 pattern 更明确，不会误判）
    if hb_parser.may_contain_cancel_intent(message):
        task_name = await loop.run_in_executor(None, hb_parser.analyze_and_cancel, message)
        if task_name:
            hb_scheduler._load_tasks()
            await broadcast("heartbeat_task_cancelled", {
                "task_name": task_name,
                "message": f"🚫 已停止定时任务：{task_name}",
            })
            return  # 取消和新建互斥，不继续检测新建意图

    # 检测新建意图
    if hb_parser.may_contain_schedule_intent(message):
        task = await loop.run_in_executor(None, hb_parser.analyze_and_write, message)
        if task:
            hb_scheduler._load_tasks()
            await broadcast("heartbeat_task_added", {
                "task_name": task["name"],
                "trigger": task["trigger"],
                "description": task.get("description", ""),
                "message": f"✅ 已为你设置定时任务：{task.get('description', task['name'])}",
            })


# ── /flush 接口（通过 Harness 触发 MemoryFlushSkill）───────────────────────

@app.post("/flush")
async def flush_session(req: FlushRequest):
    sid = req.session_id or current_session_id

    async def stream():
        # ── 用 Harness 执行 flush：trigger_flush=True 激活 MemoryFlushSkill ──
        loop = asyncio.get_event_loop()
        ctx = await loop.run_in_executor(
            None, harness.run, "", sid, True
        )

        flush_result = ctx.extras.get("flush_result")

        yield sse_event("flush_start", {
            "session_id": sid,
            "message_count": len(ctx.history),
        })
        await asyncio.sleep(0)

        if not flush_result:
            yield sse_event("flush_done", {"error": "会话为空"})
            return

        # Pass 1 — 用户信息更新
        yield sse_event("flush_pass1", {
            "user_updates": flush_result.user_updates,
            "count": len(flush_result.user_updates),
        })
        await asyncio.sleep(0)

        # Pass 2 — 新增长期记忆
        yield sse_event("flush_pass2", {
            "new_entries": [
                {"category": e.get("category", ""), "title": e.get("title", ""),
                 "content": e.get("content", "")[:100]}
                for e in flush_result.new_memory_entries
            ],
            "count": len(flush_result.new_memory_entries),
        })
        await asyncio.sleep(0)

        # Pass 3 — 向量化
        yield sse_event("flush_pass3", {
            "vectorized": flush_result.vectorized_count,
            "total_in_index": vs.total_entries,
        })
        await asyncio.sleep(0)

        if flush_result.compacted:
            yield sse_event("flush_compaction", {
                "before": flush_result.compaction_before,
                "after": flush_result.compaction_after,
            })
            await asyncio.sleep(0)

        yield sse_event("flush_done", {
            "error": flush_result.error,
            "summary": flush_result.summary(),
        })

    return StreamingResponse(stream(), media_type="text/event-stream")


# ── /memories 接口 ────────────────────────────────────────────────────────────

@app.get("/memories")
async def get_memories():
    mem_dir = loader.memory_dir

    def read_md(name):
        p = mem_dir / name
        return p.read_text(encoding="utf-8") if p.exists() else ""

    return JSONResponse({
        "user_md":      read_md("USER.md"),
        "memory_md":    read_md("MEMORY.md"),
        "soul_md":      read_md("SOUL.md"),
        "agents_md":    read_md("AGENTS.md"),
        "heartbeat_md": read_md("HEARTBEAT.md"),
        "entry_count":  loader.get_memory_entry_count(),
        "faiss_total":  vs.total_entries,
        "fts_total":    fts.total_entries,
        "fts_available": fts.available,
        "recent_sessions": db.get_recent_sessions(5),
    })


# ── /stream 接口：持久 SSE，接收调度器广播 ───────────────────────────────────

@app.get("/stream")
async def stream_events():
    q: asyncio.Queue = asyncio.Queue(maxsize=50)
    _stream_listeners.append(q)
    logger.info(f"[/stream] 新连接，当前监听数：{len(_stream_listeners)}")

    async def generate():
        try:
            tasks = hb_parser.load_tasks() if hb_parser else []
            yield sse_event("heartbeat_connected", {
                "task_count": len(tasks),
                "tasks": [{"name": t["name"], "trigger": t["trigger"],
                            "description": t.get("description", "")} for t in tasks],
            })
            while True:
                try:
                    # 最多等 20 秒，超时发 keepalive 注释，防止连接被服务端或代理关闭
                    payload = await asyncio.wait_for(q.get(), timeout=20.0)
                    yield payload
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            if q in _stream_listeners:
                _stream_listeners.remove(q)
            logger.info(f"[/stream] 连接断开，剩余监听数：{len(_stream_listeners)}")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # 关闭 Nginx 缓冲（如有反向代理）
        },
    )


# ── /reset 接口 ──────────────────────────────────────────────────────────────

@app.post("/reset")
async def reset_to_factory():
    """回到出厂初始态：重置全部 md 文件、清空 FAISS、清空 SQLite、重载调度器"""
    global current_session_id
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _do_factory_reset)
    # 文件已重置，让调度器重新读取 HEARTBEAT.md，清除所有旧 job
    if hb_scheduler:
        hb_scheduler._load_tasks()
    current_session_id = db.new_session()
    return {"status": "ok", "session_id": current_session_id}


def _do_factory_reset():
    """
    完整出厂重置，直接操作，不依赖外部脚本：
      1. 重写全部 5 个 memory/*.md（含 SOUL / AGENTS）
      2. 删除 FAISS 索引文件，重置 vs 内存状态
      3. 清空 SQLite 数据（DELETE 而非 DROP，保留 schema）
      4. 重新建表（db._init_db）确保 schema 完整
    """
    from src.reset import INITIAL_USER_MD, INITIAL_MEMORY_MD, INITIAL_HEARTBEAT_MD

    mem_dir = loader.memory_dir

    # ① 重写 md 文件（USER / MEMORY / HEARTBEAT 用常量，SOUL / AGENTS 从 backups/initial/ 恢复）
    (mem_dir / "USER.md").write_text(INITIAL_USER_MD, encoding="utf-8")
    (mem_dir / "MEMORY.md").write_text(INITIAL_MEMORY_MD, encoding="utf-8")
    (mem_dir / "HEARTBEAT.md").write_text(INITIAL_HEARTBEAT_MD, encoding="utf-8")

    project_root = mem_dir.parent
    for fname in ("SOUL.md", "AGENTS.md"):
        backup_src = project_root / "backups" / "initial" / "memory" / fname
        if backup_src.exists():
            (mem_dir / fname).write_text(backup_src.read_text(encoding="utf-8"), encoding="utf-8")

    # ② 清空 FAISS（删文件 + 重置内存对象）
    for f in ("memory.faiss", "memory_meta.pkl"):
        p = vs.index_dir / f
        if p.exists():
            p.unlink()
    vs.index = None
    vs.metadata = []

    # ③ 清空 SQLite（不删文件，直接 DELETE 保留 schema）
    conn = sqlite3.connect(db.db_path)
    conn.executescript("DELETE FROM messages; DELETE FROM sessions;")
    try:
        conn.execute("DELETE FROM memory_fts")  # P1 引入的 FTS5 全文索引表
    except sqlite3.OperationalError:
        pass  # FTS5 不可用时该表不存在，跳过
    conn.commit()
    conn.close()

    # ④ 确保 schema 完整（防止极端情况下 schema 丢失）
    db._init_db()


# ── /session/new 接口 ─────────────────────────────────────────────────────────

@app.post("/session/new")
async def new_session():
    global current_session_id
    if current_session_id:
        db.close_session(current_session_id)
    current_session_id = db.new_session()
    return {"session_id": current_session_id}


# ── /health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "session_id": current_session_id,
        "memory_entries": loader.get_memory_entry_count(),
        "faiss_entries": vs.total_entries,
        "fts_entries":   fts.total_entries,
        "fts_available": fts.available,
        "model": current_model_info(),
    }


# ── 静态文件（index.html）────────────────────────────────────────────────────

INDEX_HTML = Path(__file__).parent.parent / "index.html"

@app.get("/")
async def root():
    return FileResponse(INDEX_HTML)
