# CycloVision One-Command Launcher (Windows PowerShell)
$projectRoot = $PSScriptRoot
Set-Location $projectRoot

$env:Path = "C:\Users\albin\.local\bin;C:\Users\albin\.local\node;$env:Path"
$env:PYTHONPATH = $projectRoot

Write-Host "===========================================" -ForegroundColor Cyan
Write-Host " Starting CycloVision Platform" -ForegroundColor Cyan
Write-Host " AI-Powered Cyclone Pattern Intelligence" -ForegroundColor Cyan
Write-Host " SIH 2026 - Problem Statement: SIH26070" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan

# 1. Start FastAPI Backend in background job
Write-Host "[1/2] Starting FastAPI Backend on http://localhost:8000 ..." -ForegroundColor Yellow
$backendJob = Start-Job -ScriptBlock {
    param($root)
    Set-Location $root
    $env:Path = "C:\Users\albin\.local\bin;C:\Users\albin\.local\node;$env:Path"
    $env:PYTHONPATH = $root
    & ".\.venv\Scripts\python" -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
} -ArgumentList $projectRoot

Start-Sleep -Seconds 3

# 2. Start Vite Frontend
Write-Host "[2/2] Starting React Vite Frontend on http://localhost:5173 ..." -ForegroundColor Yellow
Write-Host "`n===========================================" -ForegroundColor Green
Write-Host " CycloVision is running!" -ForegroundColor Green
Write-Host "===========================================" -ForegroundColor Green
Write-Host " Frontend: http://localhost:5173" -ForegroundColor White
Write-Host " Backend:  http://localhost:8000" -ForegroundColor White
Write-Host " API Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host " Mode:     DEMO / OPERATIONAL" -ForegroundColor White
Write-Host " Press Ctrl+C in terminal to stop." -ForegroundColor Gray
Write-Host "===========================================`n" -ForegroundColor Green

Set-Location (Join-Path $projectRoot "frontend")
try {
    & npm run dev -- --host 127.0.0.1 --port 5173
} finally {
    Stop-Job $backendJob -ErrorAction SilentlyContinue
    Remove-Job $backendJob -ErrorAction SilentlyContinue
    Write-Host "`nCycloVision servers stopped gracefully." -ForegroundColor Yellow
}