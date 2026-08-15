@echo off
REM ============================================================
REM  Historical Figure Viral Article Generator Launcher
REM  Main agent dispatches concurrent subagents to gather
REM  materials (life / anecdotes / controversy), then writes
REM  a viral WeChat-style article.
REM ============================================================
cd /d "%~dp0"

REM Load .env if present (keys: DEEPSEEK_API_KEY, TAVILY_API_KEY)
if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
        set "%%a=%%b"
    )
)

if not defined DEEPSEEK_API_KEY (
    echo [WARN] DEEPSEEK_API_KEY not set. Copy .env.example to .env and fill it in.
)
if not defined TAVILY_API_KEY (
    echo [WARN] TAVILY_API_KEY not set. Subagents need it for web search.
)

echo.
echo Starting article generator on http://localhost:8005
echo Press Ctrl+C to stop.
echo.
python -m uvicorn src.serve:app --host 0.0.0.0 --port 8005
pause
