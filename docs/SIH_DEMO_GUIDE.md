# CycloVision SIH Demo Guide
## 2-4 Minute Presentation Walkthrough

---

## Prerequisites (Before Demo Starts)

1. Open **two terminals** side by side:
   - **Terminal A**: Backend running at `http://localhost:8000`
   - **Terminal B**: Frontend dev server at `http://localhost:5173`
2. Open browser at `http://localhost:5173`
3. Ensure Wi-Fi is available (for Google Fonts to load)

---

## Demo Script (4 minutes)

### Slide 1 — Open Dashboard (10 seconds)
- Browser shows CycloVision Mission Control
- Point out the **DEMO MODE READY** banner (top) and sidebar navigation
- Say: *"This is our AI-powered cyclone intelligence platform, purpose-built for the North Indian Ocean basin using INSAT-3D and 3DR satellite imagery."*

### Slide 2 — Select Storm: Cyclone AMPHAN (20 seconds)
- Header bar → click **AMPHAN (Bay of Bengal)**
- Dashboard auto-refreshes with AI inference results
- Say: *"Cyclone Amphan (2020) underwent one of the most dramatic rapid intensification events in Bay of Bengal history — from 115 knots to Super Cyclone status in just 12 hours."*

### Slide 3 — Mission Control Overview (60 seconds)
- Point to **Cyclone Detection card**: *"Model 1 — Our ResNet-18 CNN processes the dual-channel IR and Water Vapour frame and classifies it as Cyclonic with 99.9% confidence."*
- Point to **Intensity card**: *"Model 2 — The CNN + ConvLSTM temporal model analyses a 6-frame sequence and classifies intensity across 7 IMD categories."*
- Point to **RI card (red, blinking)**: *"Model 3 — Our dedicated Rapid Intensification module flags an 88% RI risk — meaning the storm is projected to gain at least 30 knots within 24 hours."*
- Point to the **Intensity chart**: *"This shows the ConvLSTM sequence history. Notice wind speed and RI probability spiking simultaneously — exactly matching the real Amphan trajectory."*

### Slide 4 — Satellite Analysis + Grad-CAM (45 seconds)
- Click **Satellite & Grad-CAM** in sidebar
- Show the satellite frame
- Click **Grad-CAM Attention Map**
- Say: *"This is our explainability layer — Grad-CAM highlights the exact convective cloud structures that influenced the model's prediction. In operational forecasting, this lets IMD meteorologists verify the AI found the same eye wall and spiral rainbands a human analyst would look for."*
- Scrub through timeline frames to show temporal evolution

### Slide 5 — Storm Trajectory (30 seconds)
- Click **Storm Trajectory** in sidebar
- Point to IBTrACS observed track entries: *"These are real IBTrACS best-track coordinates."*
- Point to AI forecast rows in amber: *"And these are CycloVision's 24-hour AI position projections — clearly labelled as forecasts, not official warnings."*

### Slide 6 — NDMA Decision Support (30 seconds)
- Click **NDMA Decision Support** in sidebar
- Say: *"This is the actionable output layer for disaster managers. The system synthesises all three model outputs and generates an advisory: evacuation radius, maritime signal recommendation, and NDRF mobilisation guidance."*
- Read the RI protocol card: *"When RI is flagged, we recommend advancing evacuation timelines by 18 hours."*

### Slide 7 — Switch to Biparjoy (15 seconds)
- Header bar → click **BIPARJOY (Arabian Sea)**
- Dashboard updates with moderate RI risk (42%) and different intensity
- Say: *"Watch how the risk assessment changes dynamically — Biparjoy was an Extremely Severe storm but without the same rapid intensification profile."*

### Slide 8 — Non-Cyclonic Baseline (10 seconds)
- Click **Baseline Convection**
- Point to **NON-CYCLONIC** detection card
- Say: *"And here's the system correctly identifying disorganised oceanic convection — the model doesn't hallucinate a cyclone when there isn't one."*

---

## Key Technical Talking Points for Judges

1. **Why IR + Water Vapour?** IR identifies cold convective cloud tops; WV shows upper tropospheric moisture transport — together they capture both surface cyclone structure and atmospheric circulation environment.

2. **Why ConvLSTM?** Standard LSTM operates on flat vectors, losing spatial structure. ConvLSTM propagates hidden state as feature maps, preserving the geometric spiral organisation of rainbands over time.

3. **Why Focal Loss for RI?** RI events are 7-10% of observations. Standard cross-entropy would learn to predict "no RI" always and still be 90% accurate. Focal Loss (γ=2) penalises confident wrong predictions on the minority class by a quadratic factor.

4. **Why Grad-CAM?** IMD forecasters must trust the AI before using it. Grad-CAM gives a physical interpretation: if the model activates on the eye wall, forecasters know it's reasoning about the right structure, not statistical noise.