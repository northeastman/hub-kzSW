"""
教学演示服务（优化版）：步进式 Agent 自进化 UI，接入所有优化。

接入的优化：
  - 方案 1：Agent 按需注入 Skill（teaching_mode 开关，通过 /mode 切换）
  - 方案 2：SkillManager mtime 缓存（透明生效）
  - 方案 3：基线/最终/Probe 评估并发化（max_concurrency）
  - 方案 5：Reviewer 提示词压缩（teaching_mode 开关）
  - 方案 6：基线评估结果缓存（use_baseline_cache 开关）

新增接口：
  - GET  /mode           查询当前模式（teaching/optimized）
  - POST /mode           切换模式（需重置实验后才生效）
  - GET  /stats          查询优化统计（路由命中率、压缩率等）

启动：
  cd self_evolving_agent_optimized
  uvicorn serve:app --host 0.0.0.0 --port 8000 --reload
"""

import os, sys, json, asyncio, shutil
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from pydantic import BaseModel

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from skill_manager import SkillManager
from evaluator import Evaluator
from agent import CustomerServiceAgent
from background_reviewer import BackgroundReviewer
from async_eval import run_eval_concurrent_streaming
from baseline_cache import (
    try_load_baseline_cache,
    save_baseline_cache,
    clear_baseline_cache,
)

SKILLS_DIR    = ROOT / "skills"
SKILLS_ORIG   = ROOT / "outputs" / "skills_original"
VERSIONS_DIR  = ROOT / "outputs" / "skill_versions"
EVAL_RUNS_DIR = ROOT / "outputs" / "eval_runs"
EVAL_SET      = ROOT / "data" / "eval_set.json"
DEMO_SCRIPT   = ROOT / "data" / "demo_script.json"
POLICIES      = ROOT / "data" / "policies.md"
BASELINE_CACHE = ROOT / "outputs" / "baseline_cache.json"

_s: dict = {}   # global experiment state


# ── 初始化 / 还原 ─────────────────────────────────────────────────────────────

def _restore(clear_cache: bool = True):
    if SKILLS_ORIG.exists():
        if SKILLS_DIR.exists():
            shutil.rmtree(SKILLS_DIR)
        shutil.copytree(SKILLS_ORIG, SKILLS_DIR)
    for d in [VERSIONS_DIR, ROOT / "outputs" / "skill_snapshots", EVAL_RUNS_DIR]:
        if d.exists():
            shutil.rmtree(d)
    if clear_cache and BASELINE_CACHE.exists():
        clear_baseline_cache(BASELINE_CACHE)


def _init(teaching_mode: bool = False, max_concurrency: int = 5, use_baseline_cache: bool = True):
    global _s
    sm       = SkillManager(str(SKILLS_DIR), str(VERSIONS_DIR))
    ev       = Evaluator(str(EVAL_SET))
    agent    = CustomerServiceAgent(sm, nudge_interval=0, teaching_mode=teaching_mode)
    reviewer = BackgroundReviewer(str(POLICIES), sm, teaching_mode=teaching_mode)
    script   = json.loads(DEMO_SCRIPT.read_text(encoding="utf-8"))
    qs       = script["questions"]
    blocks   = [qs[i:i+10] for i in range(0, len(qs), 10)]
    _s = {
        "phase":        "idle",
        "current_block": 0,
        "sm": sm, "ev": ev, "agent": agent, "reviewer": reviewer,
        "blocks":       blocks,
        "probe_ids":    script.get("probe_question_ids", []),
        "eval_results": {},
        "conv_history": [],
        "nudge_count":  0,
        "teaching_mode": teaching_mode,
        "max_concurrency": max_concurrency,
        "use_baseline_cache": use_baseline_cache,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    (ROOT / "outputs").mkdir(exist_ok=True)
    if not SKILLS_ORIG.exists():
        shutil.copytree(SKILLS_DIR, SKILLS_ORIG)
    _init(teaching_mode=False, max_concurrency=5, use_baseline_cache=True)
    yield


app = FastAPI(lifespan=lifespan)

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Access-Control-Allow-Origin": "*",
}


def _evt(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── 静态 / 状态接口 ───────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(ROOT / "index.html")


@app.get("/state")
async def get_state():
    blocks = _s.get("blocks", [])
    return JSONResponse({
        "phase":         _s["phase"],
        "current_block": _s["current_block"],
        "total_blocks":  len(blocks),
        "nudge_count":   _s["nudge_count"],
        "block_names":   [b[0]["block"] for b in blocks],
        "teaching_mode": _s.get("teaching_mode", False),
        "max_concurrency": _s.get("max_concurrency", 5),
        "use_baseline_cache": _s.get("use_baseline_cache", True),
        "eval_results":  {
            k: {"accuracy": v["accuracy"], "correct": v["correct"], "total": v["total"],
                "by_category": v["by_category"]}
            for k, v in _s["eval_results"].items()
        },
    })


@app.get("/skills")
async def get_skills():
    sm = _s["sm"]
    result = {}
    for name, content in sorted(sm.load_all().items()):
        history = sm.get_version_history(name)
        result[name] = {
            "content":       content,
            "version_count": len(history),
            "history": [
                {"version": h.get("version", i+1), "time": h["time"][:19],
                 "action": h["action"], "reason": h["reason"][:120]}
                for i, h in enumerate(history)
            ],
        }
    return JSONResponse(result)


@app.get("/skill_version/{name}/{version}")
async def get_skill_version(name: str, version: int):
    history = _s["sm"].get_version_history(name)
    for h in history:
        if h.get("version") == version:
            return JSONResponse({
                "content": h["content"], "action": h["action"],
                "reason": h["reason"],  "time":   h["time"][:19],
            })
    return JSONResponse({"error": "not found"}, status_code=404)


@app.post("/reset")
async def reset():
    _restore(clear_cache=True)
    _init(
        teaching_mode=_s.get("teaching_mode", False),
        max_concurrency=_s.get("max_concurrency", 5),
        use_baseline_cache=_s.get("use_baseline_cache", True),
    )
    return {"status": "ok"}


# ── 新增：模式切换接口 ────────────────────────────────────────────────────────

class ModeConfig(BaseModel):
    teaching_mode: bool = False
    max_concurrency: int = 5
    use_baseline_cache: bool = True


@app.get("/mode")
async def get_mode():
    return JSONResponse({
        "teaching_mode": _s.get("teaching_mode", False),
        "max_concurrency": _s.get("max_concurrency", 5),
        "use_baseline_cache": _s.get("use_baseline_cache", True),
    })


@app.post("/mode")
async def set_mode(cfg: ModeConfig):
    """切换模式。需要重置实验后才生效。"""
    _s["teaching_mode"] = cfg.teaching_mode
    _s["max_concurrency"] = max(1, min(20, cfg.max_concurrency))
    _s["use_baseline_cache"] = cfg.use_baseline_cache
    return {
        "status": "ok",
        "note": "模式已记录，点击重置后生效",
        "teaching_mode": _s["teaching_mode"],
        "max_concurrency": _s["max_concurrency"],
        "use_baseline_cache": _s["use_baseline_cache"],
    }


@app.get("/stats")
async def get_stats():
    """查询优化统计。"""
    sm = _s["sm"]
    agent = _s["agent"]
    reviewer = _s["reviewer"]
    return JSONResponse({
        "skill_manager_cache": sm.cache_stats(),
        "agent_routing": agent.routing_summary(),
        "reviewer_compression": reviewer.compression_summary(),
        "teaching_mode": _s.get("teaching_mode", False),
        "max_concurrency": _s.get("max_concurrency", 5),
    })


# ── 工具函数 ──────────────────────────────────────────────────────────────────

async def _probe_streaming(ev, agent, sm, probe_ids: list[int], run_id: str,
                            emit_queue: asyncio.Queue):
    """并发跑 probe eval，每题完成通过 emit_queue 推送事件。"""
    async def on_result(qid, answer, correct, fail_reason):
        await emit_queue.put({
            "type": "probe_q", "run_id": run_id, "id": qid,
            "correct": correct,
            "category": ev.questions[qid]["category"],
            "answer": answer[:200],
            "fail_reason": fail_reason,
        })

    result = await run_eval_concurrent_streaming(
        agent, ev, probe_ids, on_result,
        max_concurrency=_s.get("max_concurrency", 5),
        teaching_mode=_s.get("teaching_mode", False),
    )
    result["run_id"] = run_id
    result["skill_versions_active"] = sm.get_active_versions()
    _s["eval_results"][run_id] = result
    return result


async def _probe_silent(ev, agent, sm, probe_ids: list[int], run_id: str) -> dict:
    """静默并发跑 probe eval（不流式），用于块内 Nudge 后的 Probe。"""
    result = await run_eval_concurrent_streaming(
        agent, ev, probe_ids, on_result=lambda *a: asyncio.sleep(0),
        max_concurrency=_s.get("max_concurrency", 5),
        teaching_mode=_s.get("teaching_mode", False),
    )
    result["run_id"] = run_id
    result["skill_versions_active"] = sm.get_active_versions()
    _s["eval_results"][run_id] = result
    return result


# ── SSE 流式接口 ──────────────────────────────────────────────────────────────

@app.get("/stream/baseline")
async def stream_baseline():
    async def gen():
        sm    = _s["sm"]
        ev    = _s["ev"]
        agent = _s["agent"]
        run_id = "baseline"
        _s["phase"] = "baseline_running"

        for name, content in sm.load_all().items():
            sm._save_version(name, content, action="initial", reason="初始版本")

        # ── 尝试基线缓存 ───────────────────────────────────────────────────
        use_cache = _s.get("use_baseline_cache", True) and not _s.get("teaching_mode", False)
        if use_cache:
            cached = try_load_baseline_cache(BASELINE_CACHE, sm.load_all(), EVAL_SET)
            if cached:
                # 命中缓存，逐题回放（保持 UI 流式体验）
                for qid_str, ans_data in cached["answers"].items():
                    qid = int(qid_str)
                    q = ev.questions[qid]
                    yield _evt({"type": "eval_q", "run_id": run_id, "id": qid,
                                "correct": ans_data["correct"],
                                "category": q["category"],
                                "question": q["question"][:60],
                                "answer": ans_data["answer"][:200],
                                "fail_reason": ans_data.get("fail_reason", ""),
                                "cached": True})
                result = {
                    "run_id": "baseline",
                    "accuracy": cached["accuracy"],
                    "correct": cached["correct"],
                    "total": cached["total"],
                    "by_category": cached["by_category"],
                    "timestamp": datetime.now().isoformat(),
                    "skill_versions_active": sm.get_active_versions(),
                    "cached": True,
                }
                _s["eval_results"]["baseline"] = result
                agent.conversation_history.clear()
                _s["phase"] = "baseline_done"
                yield _evt({"type": "eval_complete", "run_id": "baseline",
                            "correct": result["correct"], "total": result["total"],
                            "accuracy": result["accuracy"], "by_category": result["by_category"],
                            "cached": True})
                yield _evt({"type": "phase_change", "phase": "baseline_done"})
                yield _evt({"type": "done"})
                return

        # ── 缓存未命中，正常并发跑 ───────────────────────────────────────
        all_ids = sorted(ev.questions.keys())
        by_cat: dict = {}
        correct = 0
        answers: dict = {}

        async def on_result(qid, answer, ok, fail_reason):
            nonlocal correct
            if ok:
                correct += 1
            cat = ev.questions[qid]["category"]
            by_cat.setdefault(cat, {"total": 0, "correct": 0})
            by_cat[cat]["total"] += 1
            if ok:
                by_cat[cat]["correct"] += 1
            answers[str(qid)] = {"answer": answer, "correct": ok, "fail_reason": fail_reason}
            yield_evt = {
                "type": "eval_q", "run_id": run_id, "id": qid,
                "correct": ok, "category": cat,
                "question": ev.questions[qid]["question"][:60],
                "answer": answer[:200], "fail_reason": fail_reason if not ok else "",
            }
            await queue.put(yield_evt)

        # 用 queue 解耦并发回调与 SSE 输出
        queue: asyncio.Queue = asyncio.Queue()

        async def producer():
            await run_eval_concurrent_streaming(
                agent, ev, all_ids, on_result,
                max_concurrency=_s.get("max_concurrency", 5),
                teaching_mode=_s.get("teaching_mode", False),
            )
            await queue.put(None)  # 结束信号

        asyncio.create_task(producer())

        while True:
            evt = await queue.get()
            if evt is None:
                break
            yield _evt(evt)

        for v in by_cat.values():
            v["accuracy"] = round(v["correct"] / v["total"], 3)

        result = {
            "run_id":    "baseline",
            "accuracy":  round(correct / len(all_ids), 3),
            "correct":   correct,
            "total":     len(all_ids),
            "by_category": by_cat,
            "timestamp": datetime.now().isoformat(),
            "skill_versions_active": sm.get_active_versions(),
            "answers": answers,
        }
        _s["eval_results"]["baseline"] = result
        agent.conversation_history.clear()
        _s["phase"] = "baseline_done"

        # 保存基线缓存
        if use_cache:
            save_baseline_cache(BASELINE_CACHE, result, sm.load_all(), EVAL_SET)

        yield _evt({"type": "eval_complete", "run_id": "baseline",
                    "correct": correct, "total": len(all_ids),
                    "accuracy": result["accuracy"], "by_category": by_cat})
        yield _evt({"type": "phase_change", "phase": "baseline_done"})
        yield _evt({"type": "done"})

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)


@app.get("/stream/block/{block_index}")
async def stream_block(block_index: int):
    async def gen():
        if block_index != _s["current_block"]:
            yield _evt({"type": "error", "message": f"需要运行第 {_s['current_block']} 块"})
            return

        sm       = _s["sm"]
        ev       = _s["ev"]
        agent    = _s["agent"]
        reviewer = _s["reviewer"]
        block_qs = _s["blocks"][block_index]
        block_name = block_qs[0].get("block", f"block_{block_index}")
        _s["phase"] = f"block_{block_index}_running"

        # ── Phase 1: 逐题回答 + 累积失败样本（保持串行，教学需要按序展示） ──
        block_correct = 0
        block_failed_turns: list[dict] = []
        for item in block_qs:
            q = ev.questions.get(item["eval_id"], {})
            yield _evt({"type": "question_start", "seq": item["seq"],
                        "question": item["question"], "block": block_name,
                        "eval_id": item["eval_id"]})
            answer = await asyncio.to_thread(agent.answer, item["question"])
            ok, reason = ev.evaluate_answer(answer, item["eval_id"])
            if ok:
                block_correct += 1
            else:
                block_failed_turns.append({
                    "question": item["question"], "answer": answer, "fail_reason": reason,
                })
            _s["conv_history"].append({"question": item["question"], "answer": answer})
            agent.conversation_history = _s["conv_history"][-30:]
            yield _evt({"type": "question_result", "seq": item["seq"],
                        "answer": answer, "correct": ok,
                        "fail_reason": reason if not ok else "",
                        "category": q.get("category", "")})

        block_acc = block_correct / len(block_qs)
        yield _evt({"type": "block_complete", "block": block_name,
                    "correct": block_correct, "total": len(block_qs),
                    "accuracy": round(block_acc, 3)})

        # ── Phase 2: 全对则跳过进化，否则只把失败样本送 Reviewer ─────────────
        if not block_failed_turns:
            yield _evt({"type": "nudge_skipped", "block": block_name,
                        "reason": "本块全部答对，跳过 Nudge 和 Probe eval"})
        else:
            _s["nudge_count"] += 1
            yield _evt({"type": "nudge_start",
                        "nudge_num": _s["nudge_count"],
                        "block": block_name,
                        "failure_count": len(block_failed_turns)})

            actions = await asyncio.to_thread(reviewer.review, block_failed_turns)
            analysis = getattr(reviewer, "last_analysis", "")
            yield _evt({"type": "reviewer_analysis", "analysis": analysis})

            executed = []
            for act in (actions or []):
                try:
                    if act["action"] == "create":
                        ok_act = sm.create(act["skill_name"], act["content"],
                                           reason=act.get("reason", ""))
                    elif act["action"] == "patch":
                        ok_act = sm.patch(act["skill_name"], act["old_text"],
                                          act["new_text"], reason=act.get("reason", ""))
                    else:
                        ok_act = False
                    if ok_act:
                        executed.append({
                            "action":     act["action"],
                            "skill_name": act["skill_name"],
                            "reason":     act.get("reason", "")[:120],
                        })
                        yield _evt({"type": "skill_action", **executed[-1]})
                except Exception as e:
                    yield _evt({"type": "skill_error", "error": str(e)[:100]})

            yield _evt({"type": "nudge_complete", "num_actions": len(executed)})

            # ── Phase 3: Probe eval 并发跑 ───────────────────────────────────
            yield _evt({"type": "probe_start", "total": len(_s["probe_ids"])})
            run_id = f"after_block_{block_index}"

            queue: asyncio.Queue = asyncio.Queue()
            correct = 0
            by_cat: dict = {}

            async def on_probe_result(qid, answer, ok, fail_reason):
                nonlocal correct
                if ok:
                    correct += 1
                cat = ev.questions[qid]["category"]
                by_cat.setdefault(cat, {"total": 0, "correct": 0})
                by_cat[cat]["total"] += 1
                if ok:
                    by_cat[cat]["correct"] += 1
                await queue.put({
                    "type": "probe_q", "run_id": run_id, "id": qid,
                    "correct": ok, "category": cat,
                    "answer": answer[:200], "fail_reason": fail_reason,
                })

            async def producer():
                await run_eval_concurrent_streaming(
                    agent, ev, _s["probe_ids"], on_probe_result,
                    max_concurrency=_s.get("max_concurrency", 5),
                    teaching_mode=_s.get("teaching_mode", False),
                )
                await queue.put(None)

            asyncio.create_task(producer())

            while True:
                evt = await queue.get()
                if evt is None:
                    break
                yield _evt(evt)

            for v in by_cat.values():
                v["accuracy"] = round(v["correct"] / v["total"], 3)
            total = len(_s["probe_ids"])

            probe_result = {
                "run_id": run_id,
                "accuracy": round(correct / total, 3),
                "correct": correct,
                "total": total,
                "by_category": by_cat,
                "skill_versions_active": sm.get_active_versions(),
            }
            _s["eval_results"][run_id] = probe_result

            yield _evt({"type": "probe_result", "run_id": run_id,
                        "correct": correct, "total": total,
                        "accuracy": probe_result["accuracy"],
                        "by_category": by_cat,
                        "skill_versions": probe_result["skill_versions_active"]})

        # ── 推进状态 ─────────────────────────────────────────────────────────
        _s["current_block"] += 1
        is_last = (_s["current_block"] >= len(_s["blocks"]))
        _s["phase"] = "all_blocks_done" if is_last else f"block_{block_index}_done"
        agent.conversation_history = agent.conversation_history[-5:]

        yield _evt({"type": "phase_change", "phase": _s["phase"],
                    "current_block": _s["current_block"]})
        yield _evt({"type": "done"})

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)


@app.get("/stream/final")
async def stream_final():
    async def gen():
        sm    = _s["sm"]
        ev    = _s["ev"]
        agent = _s["agent"]
        run_id = "final"
        _s["phase"] = "final_running"

        all_ids = sorted(ev.questions.keys())
        by_cat: dict = {}
        correct = 0

        # ── 并发跑 + 流式回调 ──────────────────────────────────────────────
        queue: asyncio.Queue = asyncio.Queue()

        async def on_result(qid, answer, ok, fail_reason):
            nonlocal correct
            if ok:
                correct += 1
            cat = ev.questions[qid]["category"]
            by_cat.setdefault(cat, {"total": 0, "correct": 0})
            by_cat[cat]["total"] += 1
            if ok:
                by_cat[cat]["correct"] += 1
            await queue.put({
                "type": "eval_q", "run_id": run_id, "id": qid,
                "correct": ok, "category": cat,
                "question": ev.questions[qid]["question"][:60],
                "answer": answer[:200], "fail_reason": fail_reason if not ok else "",
            })

        async def producer():
            await run_eval_concurrent_streaming(
                agent, ev, all_ids, on_result,
                max_concurrency=_s.get("max_concurrency", 5),
                teaching_mode=_s.get("teaching_mode", False),
            )
            await queue.put(None)

        asyncio.create_task(producer())

        while True:
            evt = await queue.get()
            if evt is None:
                break
            yield _evt(evt)

        for v in by_cat.values():
            v["accuracy"] = round(v["correct"] / v["total"], 3)

        result = {
            "run_id":    "final",
            "accuracy":  round(correct / len(all_ids), 3),
            "correct":   correct,
            "total":     len(all_ids),
            "by_category": by_cat,
            "timestamp": datetime.now().isoformat(),
            "skill_versions_active": sm.get_active_versions(),
        }
        _s["eval_results"]["final"] = result
        _s["phase"] = "complete"

        yield _evt({"type": "eval_complete", "run_id": "final",
                    "correct": correct, "total": len(all_ids),
                    "accuracy": result["accuracy"], "by_category": by_cat})
        yield _evt({"type": "phase_change", "phase": "complete"})
        yield _evt({"type": "done"})

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)
