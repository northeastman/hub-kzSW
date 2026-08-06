"""
serve.py — Skill Harness FastAPI 服务（SSE 流式）

启动：
  uvicorn src.serve:app --host 0.0.0.0 --port 8000 --reload

接口：
  POST /chat        SSE 流式：意图匹配 → 渐进加载 → ReAct 执行
  GET  /skills      列出所有 skill 及当前加载级别
  GET  /skills/{name}/load?level=N   手动升级某 skill 到指定级别
  GET  /stats       注册表与使用统计
  GET  /usage       最近 skill 使用记录
  POST /reset       清空使用记录
  GET  /            Web UI
"""

from __future__ import annotations
import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.skill_registry import SkillRegistry, LEVEL_SCAN, LEVEL_META, LEVEL_FULL, LEVEL_ASSETS
from src.intent_matcher import IntentMatcher
from src.skill_executor import SkillExecutor
from src.memory_store import MemoryStore
from src.llm_config import current_model_info

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT = Path(__file__).parent.parent
SKILLS_DIR = ROOT / "skills"
WORK_DIR = ROOT / "outputs" / "work"
DB_PATH = ROOT / "outputs" / "skill_memory.db"

# ── 全局单例 ─────────────────────────────────────────────────────────────────
registry: SkillRegistry = None
matcher: IntentMatcher = None
executor: SkillExecutor = None
memory: MemoryStore = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global registry, matcher, executor, memory
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    memory = MemoryStore(DB_PATH)
    registry = SkillRegistry(SKILLS_DIR)
    registry.scan()              # 启动只到 Level 0，秒级就绪
    matcher = IntentMatcher(registry)
    executor = SkillExecutor(registry, memory, WORK_DIR)
    logger.info(f"Harness 启动：发现 {len(registry.all_skills())} 个 skill（仅 Level 0）")
    logger.info(f"模型：{current_model_info()['display']}")
    yield
    logger.info("Harness 关闭")


app = FastAPI(title="Skill Harness", lifespan=lifespan)


# ── SSE 工具 ─────────────────────────────────────────────────────────────────
def sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class ChatRequest(BaseModel):
    message: str
    force_skill: str | None = None   # 跳过意图匹配，直接指定 skill


@app.post("/chat")
async def chat(req: ChatRequest):
    """主入口：意图匹配 → 渐进加载 → ReAct 执行，全程 SSE 流式推送。"""

    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()

        async def broadcaster(event_type: str, data: dict):
            await queue.put(sse_event(event_type, data))

        # 在后台跑主流程，结果也通过 queue 推送
        async def run():
            try:
                if req.force_skill:
                    matched = [{
                        "name":   req.force_skill,
                        "score":  1.0,
                        "reason": "用户强制指定",
                        "display_name": req.force_skill,
                        "description":  "",
                    }]
                    await broadcaster("intent_done", {"matched": matched, "forced": True})
                else:
                    matched = await matcher.match(req.message, top_k=3, broadcaster=broadcaster)

                if not matched:
                    await queue.put(sse_event("no_skill", {
                        "message": "没有 skill 匹配该请求，走普通对话。"
                    }))
                    await queue.put(sse_event("done", {"summary": ""}))
                    return

                # 取分数最高的执行
                top = matched[0]
                await queue.put(sse_event("skill_selected", top))
                result = await executor.execute(top["name"], req.message, broadcaster=broadcaster)
                await queue.put(sse_event("result", result))
            except Exception as e:
                logger.exception("chat 流程异常")
                await queue.put(sse_event("fatal_error", {"error": str(e)}))
            finally:
                await queue.put(None)  # 哨兵，结束流

        task = asyncio.create_task(run())

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/skills")
async def list_skills():
    """列出所有 skill 及其当前加载级别。"""
    return {
        "skills": [e.to_public_dict() for e in registry.all_skills()],
        "stats":  registry.stats(),
    }


@app.get("/skills/{name}/load")
async def load_skill_level(name: str, level: int = 2):
    """手动把某 skill 升级到指定级别（0~3）。"""
    if level <= LEVEL_SCAN:
        return {"name": name, "level": LEVEL_SCAN, "msg": "已是最低级别"}
    if level >= LEVEL_META:
        registry.load_metadata(name)
    if level >= LEVEL_FULL:
        registry.load_full(name)
    if level >= LEVEL_ASSETS:
        registry.load_assets(name)
    entry = registry.get(name)
    if entry is None:
        raise HTTPException(404, f"skill '{name}' 不存在")
    return entry.to_public_dict()


@app.get("/stats")
async def stats():
    return {
        "registry": registry.stats(),
        "skills": [
            {"name": e.name, "level": e.level, "display_name": e.display_name}
            for e in registry.all_skills()
        ],
        "memory": memory.skill_stats(),
        "model":  current_model_info(),
    }


@app.get("/usage")
async def usage(limit: int = 20):
    return {"records": memory.recent_usage(limit)}


@app.post("/reset")
async def reset():
    memory.reset()
    return {"msg": "使用记录已清空"}


@app.get("/")
async def index():
    index_html = ROOT / "index.html"
    if index_html.exists():
        return FileResponse(index_html)
    return JSONResponse({"msg": "index.html 不存在"}, status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.serve:app", host="0.0.0.0", port=8000, reload=True)
