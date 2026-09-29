# CycloVision — Project Status & Implementation Tracking

**Problem Statement ID:** SIH26070  
**Title:** AI/ML-based System for Identification, Classification, and Prediction of Tropical Cyclone Patterns using Multi-Source Satellite Data  
**Theme:** Disaster Management | **Category:** Software  
**Team:** La Squadra  
**Status:** ✅ COMPLETE — Demo Ready

---

## Status Legend
- `[✓]` Complete
- `[~]` In Progress
- `[ ]` Remaining / Scheduled
- `[!]` Blocked / Risk

---

## Workspace Audit & Baseline Findings

- **Reference Presentation:** `SIH26070_CycloVision_LaSquadra.pptx` — audited. 3-stage AI pipeline confirmed.
- **UI Design System:** `stitch_cyclone_ai_platform (4).zip` — Deep Space Atmospheric Intelligence design spec extracted and implemented.
- **Environment:**
  - OS: Windows 11
  - GPU: NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 13.1 driver / PyTorch detects CUDA 12.1)
  - Python: 3.11.16 (via uv)
  - Node.js: v20.18.0 (portable)
  - PyTorch: 2.5.1+cu121

---

## Component Status Checklist

### 1. Project Scaffolding & Configuration
- [✓] Repository structure at `C:\Users\albin\Documents\cyclo`
- [✓] Central YAML configuration (`ml/configs/default.yaml`) with all hyperparameters
- [✓] Environment templates (`.env.example`, `.gitignore`)
- [✓] All `__init__.py` files for Python package structure

### 2. ML Core & Data Pipeline
- [✓] Satellite preprocessor (`ml/preprocessing/preprocessor.py`) — IR+WV normalization, crop, sequence assembly
- [✓] IBTrACS loader (`ml/datasets/ibtracs_loader.py`) — demo catalog with Amphan + Biparjoy tracks
- [✓] Model 1: Cyclone Detection — ResNet-18 (2-channel), Val Acc: 100%
- [✓] Model 2: Intensity Classification — CNN+ConvLSTM (6-frame, 7 IMD classes), Val Acc: 82.1%
- [✓] Model 3: Rapid Intensification — Multimodal Focal Net (γ=2.0), Val Focal Loss: 0.0108
- [✓] Grad-CAM explainability (`ml/explainability/gradcam.py`) — Layer4 heatmaps, base64 overlay
- [✓] Training scripts for all 3 models
- [✓] Evaluation suite with Accuracy, F1, PR-AUC, Confusion Matrix
- [✓] Demo satellite sequences generated (Amphan, Biparjoy, Non-Cyclonic — 6 frames × 2 channels × 128×128)
- [✓] All model checkpoints saved (`models/detection/`, `models/intensity/`, `models/ri/`)

### 3. Backend (FastAPI)
- [✓] FastAPI application with CORS middleware
- [✓] Pydantic v2 schemas for all request/response types
- [✓] ModelManager — GPU/CPU auto-detection, graceful demo fallback
- [✓] Alert engine — NDMA/SDMA action advisories with precautionary radii
- [✓] `/health` endpoint with GPU name
- [✓] `/api/status` with model inventory
- [✓] `/api/storms` — IBTrACS storm catalog + 24h AI forecast projections
- [✓] `/api/demo/storm/{key}` — Full inference pipeline: detect → classify → RI → Grad-CAM → alert
- [✓] `POST /api/analyse` — Satellite upload analysis: multipart IR (required) + WV (optional) → detect → classify → RI → Grad-CAM → alert; IR-only uploads use a clearly-labelled IR-derived WV proxy
- [✓] Decision-support report block (`backend/app/services/insight_engine.py`) — formal IMD `system_type`, `effects` impact profile, and 24/48/72 h `outlook`; attached to BOTH demo and upload responses. Deterministic expert-system interpretation of real model outputs (never fabricates model results)
- [✓] Static serving of demo satellite images
- [✓] 8/8 unit tests passing

### 4. Frontend (React + Vite 5 + Tailwind CSS)
- [✓] React 18 + Vite 5 + Tailwind CSS — Apple-style clean light redesign (white/gray canvas, rounded-2xl cards, system SF font stack, `#0071e3` accent)
- [✓] `npm run build` — production bundle: ~624 kB (build verified)
- [✓] Sticky top navigation (7 pill tabs: Overview · Analyze · Satellite · Track · Analytics · Response · Architecture) + target-system segmented control
- [✓] Tab 1: Overview — hero with CTA, 4 stat tiles, Grad-CAM glance, honest projected-intensity line chart (built from backend `outlook`)
- [✓] Tab 2: Analyze — dropzone single-observation upload (IR required, WV recommended) → full Intelligence Report: storm type, impacts, RI gauge, 72 h outlook, advisory; proxy-WV note when IR-only
- [✓] Tab 3: Satellite — IR/WV/Grad-CAM pill switcher + Apple-style timeline scrubber
- [✓] Tab 4: Track — IBTrACS best-track table with AI forecast rows (amber)
- [✓] Tab 5: Analytics — 7-class posterior distribution bars + prominent RI early-warning gauge
- [✓] Tab 6: Response — NDMA/SDMA action matrix + maritime advisory + current advisory
- [✓] Tab 7: Architecture — pipeline + 3 model spec cards
- [✓] Header storm switcher: Amphan | Biparjoy | Baseline Convection (+ "UPLOAD ANALYSIS" chip when an upload is active, with Back-to-Demo reset)
- [✓] Disclaimer banner + footer (IMD verification reminder)
- [✓] Offline fallback in api.js (works if backend is down)

### 5. Documentation & Demo Artifacts
- [✓] `README.md` — full installation, usage, architecture, data sources, limits
- [✓] `ARCHITECTURE.md` — Mermaid diagram + model spec tables
- [✓] `RUNBOOK.md` — exact commands verified
- [✓] `data/README.md` — MOSDAC/IBTrACS/ERA5 download instructions
- [✓] `docs/SIH_DEMO_GUIDE.md` — 4-minute presentation script
- [✓] `docs/JUDGE_QA.md` — 25 technical Q&As
- [✓] `run_cyclovision.bat` + `run_cyclovision.ps1` — one-command launcher
- [✓] `scripts/setup.ps1` + `scripts/setup.bat`

---

## Verified Test Results

| Test | Status | Result |
|---|---|---|
| `test_health` | ✅ PASSED | `status=ok, models_loaded=True, mode=operational, device=cuda` |
| `test_status` | ✅ PASSED | Model inventory returned correctly |
| `test_storms_catalog` | ✅ PASSED | AMPHAN + BIPARJOY in catalog |
| `test_demo_analysis_amphan` | ✅ PASSED | cyclone_detected=True, gradcam.heatmap_available=True |
| `test_upload_analysis_single_ir` | ✅ PASSED | `source_mode=UPLOAD_ANALYSIS`, class probs present |
| `test_upload_analysis_dual_channel` | ✅ PASSED | `wv_source=UPLOADED`, `wv_image_base64` returned |
| `test_upload_analysis_missing_ir` | ✅ PASSED | Clean 400/422 rejection |
| `test_upload_analysis_invalid_file` | ✅ PASSED | Clean 400 with `detail` |
| `npm run build` | ✅ PASSED | ~624 kB bundle in ~3.2s |

## Model Evaluation Results (outputs/evaluation_results.json)

| Model | Metric | Score |
|---|---|---|
| Detection | Accuracy | 1.000 |
| Detection | F1 | 1.000 |
| Detection | ROC-AUC | 1.000 |
| Intensity | Accuracy | 0.714 |
| Intensity | Macro F1 | 0.619 |
| RI | Accuracy | 1.000 |
| RI | F1 | 1.000 |
| RI | PR-AUC | 1.000 |

*All metrics on synthetic demonstration dataset — real-world accuracy pending genuine INSAT-3D/3DR + IBTrACS training.

---

## Independent Re-Verification Log (continue build)

Re-audited the completed project on 2026-09-17 and re-ran every component from a clean state.

| Check | Result |
|---|---|
| Environment (`torch 2.5.1+cu121`, `cuda=True`, `fastapi/pydantic/cv2/yaml/sklearn` present) | ✅ PASS |
| Backend `pytest backend/tests/test_api.py` (4 tests) | ✅ 4/4 PASSED (5.82s) |
| `GET /health` — `models_loaded=true, mode=operational, device=cuda, RTX 3050 6GB` | ✅ PASS |
| `GET /api/status` + `GET /api/storms` (AMPHAN + BIPARJOY) | ✅ PASS |
| `/api/demo/storm/demo_cyclone_amphan` (detect=0.9993, Grad-CAM base64 returned) | ✅ PASS |
| `/api/demo/storm/demo_biparjoy` (RI=0.42 → MODERATE) and `demo_non_cyclone` (detected=false) | ✅ PASS |
| Static demo imagery `/static/demo/...` | ✅ HTTP 200 |
| CORS from `http://localhost:5173` | ✅ headers present |
| Frontend `npm run build` | ✅ 623.80 kB bundle (4.05s) |
| Frontend dev server `http://localhost:5173` | ✅ HTTP 200 |
| Frontend `npm run lint` (oxlint) | ✅ 0 errors, 3 React-compiler warnings |
| ML evaluation `python -m ml.evaluation.evaluate` | ✅ all 3 models evaluated, saved to `outputs/evaluation_results.json` |

### Fixes applied during re-verification

- [✓] **oxlint native binding** — `npm run lint` crashed with "Cannot find native binding `@oxlint/binding-win32-x64-msvc`" (npm optional-dependency bug). Fixed permanently by pinning `@oxlint/binding-win32-x64-msvc@1.83.0` in `frontend/package.json` `devDependencies`. `npm run lint` now passes (0 errors).

### Upload Analysis Feature (2026-09-17)

Implemented `POST /api/analyse` + full frontend upload flow, then verified with genuine model inference:

| Check | Result |
|---|---|
| Backend `pytest backend/tests/test_api.py` (8 tests: health, status, storms, demo, 4× upload) | ✅ 8/8 PASSED |
| IR-only upload of Amphan `frame_5_ir.png` | ✅ 200, `source_mode=UPLOAD_ANALYSIS`, honest weak result (Non-Cyclonic) — real model behaviour without a genuine WV channel |
| IR+WV dual upload of Amphan `frame_5` | ✅ `cyclone_detected=True (0.9994)`, `Extremely Severe Cyclonic Storm`, RI `0.6528 HIGH`, `temporal_tendency=Intensifying`, both previews returned |
| Upload preprocessing parity vs trained demo sequences | ✅ PNG `/255` path differs <0.004 from demo `sequence.npy` — upload path is calibrated the same as training |
| Bad/garbage upload | ✅ HTTP 400 clean `detail` |
| Frontend `npm run build` | ✅ PASS |
| Frontend `npm run lint` | ✅ 0 errors (3 pre-existing React-compiler warnings) |

### Apple-Style Redesign + One-Image Intelligence Report (2026-09-17)

Rebuilt the frontend around a single-observation workflow and added the `insight_engine` report block.

| Check | Result |
|---|---|
| New engine `backend/app/services/insight_engine.py` | ✅ IMD system type + impact profile + RI-scaled 24/48/72 h outlook |
| Schema `AnalysisResponse` extended (`system_type`, `effects`, `outlook`) | ✅ Optional — backward compatible |
| Demo route returns report block | ✅ Amphan: 5 effects, 3 outlook points |
| Upload route returns report block | ✅ ESCS 104.5 kt → +24h ESCS, +48h SuCS, +72h SuCS (Intensifying) |
| `pytest backend/tests/test_api.py` (report assertions added) | ✅ 8/8 PASSED |
| Frontend `npm run build` (Apple-style light theme) | ✅ ~624 kB |
| Frontend `npm run lint` (oxlint) | ✅ 0 errors, 3 React-compiler warnings |
| Dev-server smoke test (`http://localhost:5173`) | ✅ HTTP 200 |

Single-image upload now returns: storm type, IMD classification, sustained winds, expected impacts, a rapid-intensification gauge, the 72-hour intensity outlook, Grad-CAM explainability and the NDMA/SDMA advisory. Effects/outlook are expert-system interpretations keyed to the model's own detected class and RI probability.

### Notes

- `git init` was attempted but Git is not installed on this machine — repository stays uninitialised; `.gitignore` already covers `.venv/`, `node_modules/`, `.env`, `outputs/logs/`, `data/raw/`.
- All servers were stopped after verification; ports 8000/5173 are free and ready for the one-command launcher.*