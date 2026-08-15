"""爆款历史人物公众号文章 Subagent HTTP 服务（FastAPI + SSE 流式）

教学重点：
  SSE 逐事件推送，前端实时看到：
  - 主 agent 的 ReAct 每步（Thought/Action/Observation）
  - 派发 subagent 时拓扑加节点
  - 各 subagent 并行 ReAct 步骤（多列同时滚动 = 并行的直观证据）
  - 最终文章 + 并行加速统计
  - 历史已写人物列表（支撑「推荐下一篇」系列感）

启动：
  uvicorn src.serve:app --host 0.0.0.0 --port 8005
  浏览器开 http://localhost:8005

依赖：pip install fastapi uvicorn
"""
import os, sys, json, queue, threading, logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(app):
    logger.info("爆款历史人物文章 subagent 服务就绪 http://localhost:8005")
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class QueryRequest(BaseModel):
    question: str
    mode: str = "auto"  # "direct" | "recommend" | "auto"


@app.get("/health")
def health():
    has_tavily = bool(os.getenv("TAVILY_API_KEY"))
    has_llm = bool(os.getenv("DEEPSEEK_API_KEY"))
    return {"status": "ok", "tavily": has_tavily, "llm": has_llm}


@app.get("/history")
def history():
    """返回已写人物历史，供前端展示系列进度。"""
    from article_history import load_history, OUTPUT_DIR
    hist = load_history()
    # 附上单篇文件名，供前端点击查看
    for h in hist:
        h.setdefault("file", None)
    return JSONResponse({"history": hist, "dir": str(OUTPUT_DIR)})


@app.get("/article")
def article(file: str):
    """读取 outputs 下的单篇文章全文，供前端查看历史文章。"""
    from pathlib import Path
    from article_history import OUTPUT_DIR
    safe = Path(file).name  # 防目录穿越，只取文件名
    fp = OUTPUT_DIR / safe
    if not fp.exists() or not fp.is_file():
        return JSONResponse({"error": "文章不存在"}, status_code=404)
    try:
        text = fp.read_text(encoding="utf-8")
    except Exception as e:
        return JSONResponse({"error": f"读取失败: {e}"}, status_code=500)
    return JSONResponse({"file": safe, "content": text})


@app.post("/query")
def query(req: QueryRequest):
    """SSE 流式：主 agent + 各 subagent 的 ReAct 步骤逐事件推。"""
    import agents

    def event_stream():
        q = queue.Queue()
        SENTINEL = object()

        def push(ev):
            q.put(ev)

        def on_main_step(step):
            push({"type": "main_step", **step})

        def on_dispatch(info):
            push({"type": "dispatch", **info})

        def on_subagent_step(sid, step):
            push({"type": "subagent_step", "subagent_id": sid, **step})

        def on_subagent_done(sid, duration, topic):
            push({"type": "subagent_done", "subagent_id": sid,
                  "duration": duration, "subtopic": topic})

        def run():
            try:
                r = agents.run_article(
                    req.question,
                    on_main_step=on_main_step,
                    on_dispatch=on_dispatch,
                    on_subagent_step=on_subagent_step,
                    on_subagent_done=on_subagent_done,
                    mode=req.mode,
                )
                push({"type": "final", "answer": r["final_answer"],
                      "figure": r["figure"], "title": r["title"],
                      "parallel_stats": r["parallel_stats"],
                      "main_trace_len": len(r["main_trace"]),
                      "subagent_count": len(r["subagents"]),
                      "token_usage": r.get("token_usage", {})})
            except Exception as e:
                logger.exception("run_article 出错")
                push({"type": "error", "message": f"{type(e).__name__}: {str(e)[:200]}"})
            finally:
                push(SENTINEL)

        threading.Thread(target=run, daemon=True).start()

        yield "data: " + json.dumps({"type": "start", "question": req.question},
                                    ensure_ascii=False) + "\n\n"
        while True:
            ev = q.get()
            if ev is SENTINEL:
                yield "data: " + json.dumps({"type": "done"}, ensure_ascii=False) + "\n\n"
                break
            yield "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8005, reload=False)
