@echo off
echo ===========================================
echo  Starting CycloVision Platform
echo  SIH 2026 - Problem Statement: SIH26070
echo ===========================================

set "PATH=C:\Users\albin\.local\bin;C:\Users\albin\.local\node;%PATH%"
set "PYTHONPATH=%~dp0"
cd /d "%~dp0"

echo [1/2] Starting Backend in new window...
start "CycloVision Backend (FastAPI)" cmd /k "cd /d %~dp0 && .venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload"

timeout /t 3 /nobreak >nul

echo [2/2] Starting Frontend...
cd frontend
call npm run dev -- --host 127.0.0.1 --port 5173