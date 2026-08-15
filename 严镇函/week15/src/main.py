"""
FastAPI 服务入口

提供 HTTP API 接口，接收用户请求并返回 Multi-Agent 执行结果。

启动方式：
    uvicorn main:app --reload --port 8000

API 端点：
    POST /api/agent/run    - 执行 Multi-Agent 任务
    GET  /api/agent/health - 健康检查
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent import main_agent
from models import AgentRequest, AgentResponse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# 创建 FastAPI 应用
app = FastAPI(
    title="Multi-Agent 并行任务执行器",
    description="基于 LLM 的 Multi-Agent 系统，支持任务拆分和并行执行",
    version="1.0.0",
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/agent/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok", "service": "multi-agent-executor"}


@app.post("/api/agent/run", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    """
    执行 Multi-Agent 任务

    请求示例：
    ```json
    {
        "query": "帮我查一下北京和上海的天气，告诉我哪个更适合出行",
        "context": {}
    }
    ```

    响应示例：
    ```json
    {
        "success": true,
        "query": "帮我查一下北京和上海的天气...",
        "plan": { ... },
        "results": [ ... ],
        "final_answer": "根据查询结果...",
        "total_time": 5.23
    }
    ```
    """
    return await main_agent.run(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)