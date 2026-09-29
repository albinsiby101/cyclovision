# CycloVision RUNBOOK
## Exact Commands to Run Everything

### Prerequisites

1. uv installed: `C:\Users\albin\.local\bin\uv.exe`
2. Node.js v20 installed: `C:\Users\albin\.local\node\node.exe`
3. NVIDIA GPU with CUDA 12.x driver (CPU fallback also works)

---

## Quick Start (Windows)

### Option A: One-Command (opens 2 terminals)
```cmd
cd C:\Users\albin\Documents\cyclo
run_cyclovision.bat
```

### Option B: PowerShell Launcher
```powershell
cd C:\Users\albin\Documents\cyclo
.\run_cyclovision.ps1
```

---

## Manual Start (for debugging)

### Terminal 1 — Backend Server

```powershell
cd C:\Users\albin\Documents\cyclo
$env:Path = "C:\Users\albin\.local\bin;C:\Users\albin\.local\node;$env:Path"
$env:PYTHONPATH = "C:\Users\albin\Documents\cyclo"
.\.venv\Scripts\python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Expected output:
```
INFO:     Loading detection model
INFO:     Loading intensity model
INFO:     Loading RI model
INFO:     CycloVision AI Models initialized. Models loaded: True | Device: cuda
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Terminal 2 — Frontend Dev Server

```powershell
cd C:\Users\albin\Documents\cyclo\frontend
$env:Path = "C:\Users\albin\.local\bin;C:\Users\albin\.local\node;$env:Path"
npm run dev
```

Expected output:
```
  VITE v5.4.0  ready in 1234 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

---

## URLs After Startup

| URL | Description |
|---|---|
| http://localhost:5173 | React Mission Control Dashboard |
| http://localhost:8000 | FastAPI root |
| http://localhost:8000/health | System health check |
| http://localhost:8000/docs | OpenAPI Swagger UI |
| http://localhost:8000/api/status | Model + GPU status |
| http://localhost:8000/api/storms | Storm catalog (IBTrACS) |
| http://localhost:8000/api/demo/storm/demo_cyclone_amphan | Full AI analysis |

---

## Running Tests

```powershell
cd C:\Users\albin\Documents\cyclo
$env:PYTHONPATH = "C:\Users\albin\Documents\cyclo"
.\.venv\Scripts\pytest backend\tests\test_api.py -v
```

Expected output:
```
PASSED backend/tests/test_api.py::test_health
PASSED backend/tests/test_api.py::test_status
PASSED backend/tests/test_api.py::test_storms_catalog
PASSED backend/tests/test_api.py::test_demo_analysis_amphan
4 passed in 5.xx s
```

---

## Retrain Models

```powershell
cd C:\Users\albin\Documents\cyclo
$env:PYTHONPATH = "C:\Users\albin\Documents\cyclo"

# Detection model
.\.venv\Scripts\python -m ml.training.train_detection

# Intensity model (CNN + ConvLSTM)
.\.venv\Scripts\python -m ml.training.train_intensity

# Rapid Intensification model
.\.venv\Scripts\python -m ml.training.train_ri
```

---

## Run Evaluation

```powershell
$env:PYTHONPATH = "C:\Users\albin\Documents\cyclo"
.\.venv\Scripts\python -m ml.evaluation.evaluate
# Results: outputs/evaluation_results.json
```

---

## Regenerate Demo Data

```powershell
$env:PYTHONPATH = "C:\Users\albin\Documents\cyclo"
.\.venv\Scripts\python scripts\generate_demo_data.py
```

---

## Environment Variables

Copy `.env.example` to `.env` and adjust as needed:

```
APP_NAME=CycloVision
APP_ENV=development
PORT=8000
HOST=127.0.0.1
CORS_ORIGINS=["http://localhost:5173","http://localhost:3000"]
DEMO_MODE=true
DEVICE=auto
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `uv: command not found` | Add `C:\Users\albin\.local\bin` to `$env:Path` |
| `node: command not found` | Add `C:\Users\albin\.local\node` to `$env:Path` |
| Port 8000 already in use | `netstat -ano | findstr 8000` then kill the PID |
| CUDA not detected | Verify NVIDIA driver 591+ and rerun; CPU fallback activates automatically |
| `ModuleNotFoundError: backend` | Set `$env:PYTHONPATH = "C:\Users\albin\Documents\cyclo"` |
| Frontend CORS error | Ensure backend is running on port 8000 before opening frontend |