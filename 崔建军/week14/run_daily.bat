@echo off
REM ============================================================
REM  AI Daily News - Windows Task Scheduler Entry
REM
REM  First-time setup:
REM    1. pip install openai feedparser python-dotenv
REM    2. copy .env.example .env  then fill DEEPSEEK_API_KEY in .env
REM
REM  Task Scheduler config:
REM    - Program: full path of this bat file
REM    - Start in: this bat's folder
REM    - Check "Run whether user is logged on or not"
REM ============================================================

REM Switch to this bat's folder
cd /d "%~dp0"

REM Run scheduler. If python not in PATH, use full path like:
REM "D:\Python311\python.exe" run_daily.py
python run_daily.py

REM If failed, pause only when run by double-click (no arg).
REM For Task Scheduler auto-run, call "run_daily.bat silent" to skip pause.
if errorlevel 1 (
  echo.
  echo [FAILED] python exit code: see messages above.
  echo Check run_daily.log for details.
  if /i not "%~1"=="silent" pause
)

exit /b %ERRORLEVEL%
