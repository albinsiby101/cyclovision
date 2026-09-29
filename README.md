# CycloVision — AI-Powered Cyclone Pattern Intelligence & Early Warning System

> **Smart India Hackathon 2026 | Problem Statement ID: SIH26070**  
> Theme: Disaster Management | Category: Software | Team: La Squadra

---

> ⚠️ **DISCLAIMER**: CycloVision is an AI decision-support prototype and must not replace official cyclone advisories issued by IMD or authorised disaster-management agencies.

---

## What is CycloVision?

CycloVision is an end-to-end AI prototype that processes multi-channel satellite imagery from INSAT-3D/3DR and outputs cyclone identification, intensity classification, and Rapid Intensification (RI) early warnings — all explained through Grad-CAM visual heatmaps and presented in a polished mission-control dashboard.

---

## Key Features

| Feature | Description |
|---|---|
| 🔍 **Cyclone Detection** | ResNet-18 CNN binary classifier (Cyclonic / Non-Cyclonic) |
| 📊 **Intensity Classification** | CNN + ConvLSTM temporal model across 7 IMD categories |
| 🔥 **Rapid Intensification (RI)** | Dedicated 24h early-warning module (≥ 30 kt / 24 h) |
| 🧠 **Grad-CAM Explainability** | Visual attention maps over cyclone eye & rainbands |
| 🗺️ **IBTrACS Storm Tracks** | North Indian Ocean historical + forecast trajectory table |
| 🚨 **NDMA Decision Support** | Automated action advisories with precautionary radii |
| 🌙 **Demo Mode** | Full run without downloading large satellite archives |
| ⚡ **GPU Accelerated** | Auto-detects CUDA; CPU fallback guaranteed |

---

## Architecture

```
INSAT-3D/3DR (MOSDAC)
  IR 10.8 μm + Water Vapour 6.8 μm
          ↓
  Preprocessing
  (crop · normalise · sequence assembly)
          ↓
    [Model 1] ResNet-18 Detector
    Cyclonic / Non-Cyclonic
          ↓
    [Model 2] CNN + ConvLSTM Intensity
    7 IMD Categories (Depression → Super)
          ↓
    [Model 3] Multimodal RI Focal Net
    Rapid Intensification (≥ 30 kt / 24 h)
          ↓
    Grad-CAM Heatmap Generator
          ↓
    Alert Engine (NDMA/SDMA Advisory)
          ↓
    FastAPI REST Backend  ←→  React Mission Control Dashboard
```

---

## Data Sources

| Source | Type | Access |
|---|---|---|
| INSAT-3D/3DR via MOSDAC | Satellite IR + WV imagery | https://www.mosdac.gov.in (free registration) |
| IBTrACS NI Basin | Cyclone best-track labels | https://www.ncei.noaa.gov/products/international-best-track-archive |
| ERA5 (optional) | Environmental reanalysis | https://cds.climate.copernicus.eu |
| TC PRIMED (optional) | AI-ready multi-source dataset | https://rammb-data.cira.colostate.edu/tcprimed |

---

## Repository Structure

```
CycloVision/
├── backend/              # FastAPI REST API
│   ├── app/
│   │   ├── main.py       # Entry point
│   │   ├── api/routes/   # Endpoints
│   │   ├── core/         # Configuration
│   │   ├── schemas/      # Pydantic v2 models
│   │   └── services/     # AI + Alert engines
│   ├── tests/            # Pytest suite
│   └── requirements.txt
├── frontend/             # React + Vite + Tailwind dashboard
│   └── src/
│       ├── App.jsx       # Main 6-tab dashboard
│       └── api.js        # API client with offline fallbacks
├── ml/
│   ├── configs/default.yaml    # Central configuration
│   ├── detection/              # ResNet-18 detector
│   ├── intensity/              # CNN + ConvLSTM model
│   ├── rapid_intensification/  # RI module + Focal Loss
│   ├── explainability/         # Grad-CAM
│   ├── preprocessing/          # Satellite data normalisation
│   ├── datasets/               # IBTrACS loader
│   ├── training/               # Training scripts
│   └── evaluation/             # Metrics evaluation
├── data/
│   ├── demo/             # Pre-generated demo sequences
│   └── raw/              # Place downloaded datasets here
├── models/               # Saved model checkpoints (.pt)
├── outputs/              # Predictions, Grad-CAM, plots, logs
├── docs/                 # SIH demo guide, judge Q&A
├── scripts/              # Setup and generation scripts
├── run_cyclovision.bat   # Windows quick-start
├── run_cyclovision.ps1   # PowerShell quick-start
└── .env.example          # Configuration template
```

---

## Installation

### Prerequisites
- Windows 10/11 with NVIDIA GPU (RTX recommended) *or* CPU
- Git

### Step 1 — Install `uv` (Python manager)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 2 — Install portable Node.js (if not installed)
Download from: https://nodejs.org/en/download/

### Step 3 — Run Setup
```powershell
cd C:\Users\albin\Documents\cyclo
.\scripts\setup.ps1
```

This will:
1. Create `.venv` with Python 3.11
2. Install PyTorch 2.5.1+cu121 and all backend requirements
3. Install frontend npm packages
4. Generate demo satellite sequences
5. Run smoke tests

---

## Running Locally

### One-command launcher (recommended)
```powershell
cd C:\Users\albin\Documents\cyclo
.\run_cyclovision.bat
```

### Manual startup (two terminals)

**Terminal 1 — Backend:**
```powershell
cd C:\Users\albin\Documents\cyclo
$env:PYTHONPATH = "C:\Users\albin\Documents\cyclo"
.\.venv\Scripts\python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 — Frontend:**
```powershell
cd C:\Users\albin\Documents\cyclo\frontend
npm run dev
```

| URL | Service |
|---|---|
| http://localhost:5173 | React Dashboard |
| http://localhost:8000 | FastAPI Backend |
| http://localhost:8000/docs | OpenAPI / Swagger UI |

---

## Demo Mode

CycloVision ships with pre-generated synthetic satellite sequences that realistically simulate:
- **Cyclone Amphan** (Bay of Bengal — Super Cyclone with Rapid Intensification)
- **Cyclone Biparjoy** (Arabian Sea — Extremely Severe Cyclonic Storm)
- **Non-Cyclonic Convection** (baseline oceanic cloud clusters)

Select between them from the header bar of the dashboard.

---

## Dataset Preparation (Real Data)

See `data/README.md` for step-by-step instructions on downloading and placing:
- INSAT-3D/3DR imagery from MOSDAC
- IBTrACS NI basin CSV
- ERA5 ERA5 reanalysis (optional)

---

## Training

```powershell
$env:PYTHONPATH = "C:\Users\albin\Documents\cyclo"

# Model 1: Cyclone Detection
.\.venv\Scripts\python -m ml.training.train_detection

# Model 2: Intensity Classification (CNN + ConvLSTM)
.\.venv\Scripts\python -m ml.training.train_intensity

# Model 3: Rapid Intensification Early Warning
.\.venv\Scripts\python -m ml.training.train_ri
```

---

## Evaluation

```powershell
$env:PYTHONPATH = "C:\Users\albin\Documents\cyclo"
.\.venv\Scripts\python -m ml.evaluation.evaluate
# Results saved to outputs/evaluation_results.json
```

---

## API Documentation

Full OpenAPI documentation available at http://localhost:8000/docs

### Key Endpoints
| Method | Path | Description |
|---|---|---|
| GET | `/health` | System health + GPU status |
| GET | `/api/status` | Backend + model status |
| GET | `/api/storms` | All tracked storms (IBTrACS) |
| GET | `/api/demo/storm/{key}` | Full AI analysis of demo storm |
| POST | `/api/analyse` | Analyse custom satellite sequence |

---

## Limitations

- All three models are trained on synthetic demonstrations; real-world performance requires training on genuine INSAT-3D/3DR + IBTrACS data.
- The frontend satellite images are algorithmically generated to simulate real satellite patterns.
- The suggested action radii are approximate and should not be used for operational emergency decisions.
- This system cannot replace the Dvorak technique or NWP model guidance used by IMD operational forecasters.

---

## Future Production Improvements

1. Integrate live MOSDAC INSAT-3D/3DR data stream (HDF5/BUFR ingestion)
2. Storm-split cross-validation to prevent data leakage
3. SMOTE/oversampling on tabular ERA5 features for RI model
4. NWP ensemble blending for 72-hour track prediction
5. Microwave satellite channel fusion (SSMIS/GMI) for inner-core structure
6. Pressure-wind relationships for Dvorak-style validation
7. Integration test suite with real IBTrACS label alignments
8. Docker containerisation for cloud deployment

---

## Meteorological Disclaimer

> CycloVision outputs are generated by machine learning models and should be used ONLY as supplementary decision-support information. Official cyclone warnings, watches, and advisories must be obtained from the India Meteorological Department (IMD) at https://mausam.imd.gov.in and the National Disaster Management Authority (NDMA) at https://ndma.gov.in.