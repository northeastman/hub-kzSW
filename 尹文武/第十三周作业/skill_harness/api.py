from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, HTTPException

from .config import Settings
from .executor import SkillExecutor
from .llm import AIService
from .memory import SemanticMemory
from .models import (
    ExecuteRequest,
    ExecuteResponse,
    LoadLayer,
    MemoryHit,
    MemorySearchRequest,
    SkillSnapshot,
)
from .registry import SkillRegistry


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    ai = AIService(settings)
    memory = SemanticMemory(settings.data_dir, settings.embedding_dimension, ai)
    registry = SkillRegistry(settings.skills_dir, ai)
    executor = SkillExecutor(settings, registry, memory, ai)
    scheduler = AsyncIOScheduler()

    async def heartbeat() -> None:
        await registry.refresh()
        memory.save()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        errors = registry.discover()
        app.state.discovery_errors = errors
        scheduler.add_job(
            heartbeat,
            "interval",
            seconds=settings.heartbeat_seconds,
            id="skill-heartbeat",
            max_instances=1,
            coalesce=True,
        )
        scheduler.start()
        yield
        scheduler.shutdown(wait=False)
        memory.save()

    app = FastAPI(
        title="Progressive Skill Harness",
        version="1.0.0",
        description="按需从元数据逐层加载到语义记忆的 skill 执行服务",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.ai = ai
    app.state.memory = memory
    app.state.registry = registry
    app.state.executor = executor

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "skills": len(registry.list()),
            "memory_records": memory.size,
            "embedding_backend": ai.embedding_backend,
            "discovery_errors": getattr(app.state, "discovery_errors", []),
        }

    @app.get("/skills", response_model=list[SkillSnapshot])
    async def list_skills() -> list[SkillSnapshot]:
        return registry.list()

    @app.post("/skills/refresh")
    async def refresh_skills() -> dict:
        errors = await registry.refresh()
        app.state.discovery_errors = errors
        return {"skills": len(registry.list()), "errors": errors}

    @app.post("/skills/{name}/load", response_model=SkillSnapshot)
    async def load_skill(name: str, layer: LoadLayer) -> SkillSnapshot:
        try:
            return (await registry.ensure_layer(name, layer)).snapshot()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError, TypeError, ImportError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/execute", response_model=ExecuteResponse)
    async def execute(request: ExecuteRequest) -> ExecuteResponse:
        try:
            return await executor.execute(request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="Skill execution timed out") from exc
        except (OSError, ValueError, TypeError, ImportError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/memory/search", response_model=list[MemoryHit])
    async def search_memory(request: MemorySearchRequest) -> list[MemoryHit]:
        return await memory.search(
            request.query, top_k=request.top_k, skill=request.skill
        )

    return app
