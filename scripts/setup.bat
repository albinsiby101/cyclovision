@echo off
echo ===================================================
echo  CycloVision - Project Environment Setup
echo  Smart India Hackathon 2026 (Problem ID: SIH26070)
echo ===================================================

echo [1/4] Checking environment variables and tools...
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] uv is not installed or not in PATH. Please run .\scripts\setup.ps1
    exit /b 1
)

echo [2/4] Initializing Python Virtual Environment (.venv)...
if not exist ".venv" (
    uv venv .venv --python 3.11
)

echo [3/4] Installing Python AI and Backend Dependencies...
uv pip install -r backend\requirements.txt

echo [4/4] Installing Frontend NPM Packages...
cd frontend
call npm install
cd ..

echo ===================================================
echo  Setup Complete! Launch system with run_cyclovision.bat
echo ===================================================