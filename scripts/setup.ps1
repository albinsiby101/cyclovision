# CycloVision - PowerShell Automated Setup Script
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host " CycloVision - Automated Project Setup" -ForegroundColor Cyan
Write-Host " Smart India Hackathon 2026 (Problem ID: SIH26070)" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

# Ensure uv and node in PATH
$env:Path = "C:\Users\albin\.local\bin;C:\Users\albin\.local\node;$env:Path"

Write-Host "`n[1/5] Checking Python virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path ".venv")) {
    uv venv .venv --python 3.11
}

Write-Host "`n[2/5] Installing PyTorch & backend requirements..." -ForegroundColor Yellow
uv pip install -r backend\requirements.txt

Write-Host "`n[3/5] Installing Frontend Dependencies..." -ForegroundColor Yellow
Set-Location frontend
npm install
npm run build
Set-Location $projectRoot

Write-Host "`n[4/5] Generating Demo Satellite Sequences..." -ForegroundColor Yellow
.\.venv\Scripts\python scripts\generate_demo_data.py

Write-Host "`n[5/5] Running Smoke Tests..." -ForegroundColor Yellow
$env:PYTHONPATH = $projectRoot
.\.venv\Scripts\pytest backend\tests\test_api.py

Write-Host "`n===================================================" -ForegroundColor Green
Write-Host " Setup Successful! Ready to launch CycloVision." -ForegroundColor Green
Write-Host " Run: .\run_cyclovision.ps1" -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Green