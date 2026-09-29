# CycloVision — Judge Q&A Preparation

---

## Foundational Architecture Questions

**Q: Why CNN?**
A: CNNs are the correct architectural choice for spatial pattern recognition in gridded satellite imagery. Cyclone signatures — spiral rainbands, central dense overcast (CDO), eye walls — are geometric structures that convolutional filters learn to detect through shared spatial weights. Transfer learning from ImageNet-pretrained ResNet-18 provides robust low-level feature detectors (edge, texture) that transfer well to meteorological imagery.

**Q: Why ConvLSTM over standard LSTM?**
A: Standard LSTM flattens its input to a 1D vector, destroying spatial relationships. ConvLSTM replaces the fully connected gates with convolutional operations, so the hidden state is itself a 2D feature map. This means the model remembers WHERE in the image convective towers were building — critical for tracking spiral band rotation and central pressure drop signatures over a 6-frame sequence.

**Q: Why not a standard Transformer?**
A: ViT-style Transformers require large amounts of training data (thousands of samples) to converge and are computationally heavier. For a prototype targeting the North Indian Ocean basin (limited labeled satellite sequences), CNN+ConvLSTM provides better inductive bias for spatial-temporal cyclone patterns with less data.

---

## Data Questions

**Q: What is IBTrACS?**
A: The International Best Track Archive for Climate Stewardship (IBTrACS) is the global authoritative repository of tropical cyclone track data maintained by NOAA/NCEI. It aggregates best-track records from multiple Regional Specialized Meteorological Centres (RSMCs), including IMD for the North Indian Ocean basin. We use it as our label source for storm positions, intensities, and timing.

**Q: What is INSAT-3D/3DR?**
A: INSAT-3D (2013) and INSAT-3DR (2016) are Indian geostationary meteorological satellites operated by ISRO and MOSDAC. They provide half-hourly to 15-minute rapid-scan imagery in multiple spectral channels. We use Thermal Infrared (10.8 μm) — which shows convective cloud-top temperatures — and Water Vapour (6.8 μm) — which shows upper tropospheric moisture circulation.

**Q: What is MOSDAC?**
A: Meteorological & Oceanographic Satellite Data Archival Centre, operated by Space Applications Centre (ISRO), Ahmedabad. It is the primary public access portal for INSAT-3D/3DR imagery. CycloVision is specifically designed around MOSDAC's data pipeline, making it natively compatible with IMD's operational satellite feed.

**Q: What is ERA5?**
A: ERA5 is the fifth generation global atmospheric reanalysis dataset from ECMWF, providing hourly estimates of a large number of atmospheric, land, and oceanic climate variables. For the RI model, ERA5 can supply vertical wind shear, ocean heat content proxies, and mid-level humidity — environmental variables that control whether a cyclone can rapidly intensify.

**Q: What is TC PRIMED?**
A: Tropical Cyclone PRecIpitation, Infrared, Microwave, and Environmental Dataset — an AI-ready multi-source dataset from NOAA/CSU. It combines geostationary IR, passive microwave brightness temperatures, and ERA5 environmental fields with best-track labels. Designed to enable exactly the kind of research CycloVision implements.

---

## ML Methodology Questions

**Q: What is Rapid Intensification (RI)?**
A: RI is formally defined as a sustained maximum wind speed increase of ≥ 30 knots (≈ 56 km/h) within a 24-hour period. This threshold was established by operational meteorologists and has been used consistently in research literature (Kaplan & DeMaria, 2003). RI is the single most dangerous cyclone behaviour for disaster preparedness because it compresses the warning lead time for coastal communities.

**Q: How is RI labelled in training?**
A: For real data training: using IBTrACS wind speed at time T and T+24h. If wind_T+24h − wind_T ≥ 30 knots, the observation at time T receives label 1 (RI=True). This requires IBTrACS observations at 6-hourly cadence, which IBTrACS provides for major basins.

**Q: How is class imbalance handled?**
A: Two mechanisms: (1) Focal Loss with γ=2.0, which geometrically down-weights the loss contribution from easy correct predictions and focuses training on the hard minority class examples. (2) Class-weighted positive weight (pos_weight ≈ 4.5) in the loss function, calibrated to the approximate 7-10% positive rate. We deliberately avoid raw SMOTE on satellite images because image-space interpolation between cyclone frames is meteorologically invalid.

**Q: How are training and test sets split?**
A: The correct method for real data is storm-wise split (also called cyclone-wise split): all observations from the same storm are kept in the same subset. This prevents the model from simply memorising the trajectory of a storm it has already partially seen. Random frame-wise splitting would create severe data leakage.

**Q: What prevents overfitting?**
A: (1) Dropout layers (0.2-0.3) in all classification heads; (2) L2 weight decay via AdamW (wd=1e-4); (3) Storm-wise train/val/test split; (4) BatchNorm in spatial encoder for stable gradient flow; (5) Early stopping on validation loss.

---

## Explainability Questions

**Q: What is Grad-CAM?**
A: Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017). It back-propagates the gradient of the target class score to the last convolutional layer and computes a weighted average of the activation maps — producing a heatmap that highlights which spatial regions drove the model's classification. For cyclone detection, this reliably activates on the eye wall and spiral rainbands.

**Q: Does Grad-CAM prove the model is making meteorologically correct decisions?**
A: No — Grad-CAM shows correlation, not causality. It indicates which image regions the model found discriminative, but a meteorologist must still verify that these align with physically meaningful structures. This is why we present it as "regions that most influenced the model prediction" rather than claiming the model understands cyclone physics.

---

## Deployment & Infrastructure Questions

**Q: Why FastAPI?**
A: FastAPI provides automatic OpenAPI documentation, native async support for non-blocking inference, Pydantic v2 validation that catches bad inputs before they reach the model, and high performance close to Node.js in benchmarks. It is the standard choice for Python ML API backends.

**Q: Why React?**
A: React's component model maps cleanly to the dashboard's tab structure. The recharts library provides interactive time-series visualisation out of the box. Tailwind CSS allows rapid iteration on the meteorological dark-mode UI without writing custom CSS.

**Q: Can the project run without GPU?**
A: Yes. The code calls `torch.cuda.is_available()` at startup and automatically routes all tensor operations to CPU if CUDA is unavailable. Inference is slower on CPU but functional. The startup log clearly prints `Device: CPU` or `Device: CUDA`.

**Q: How would IMD integrate this?**
A: Phase 1: Deploy as a parallel decision-support tool alongside existing Dvorak analysis, with IMD meteorologists rating its recommendations. Phase 2: Retrain on the full historical MOSDAC archive (2015-present) with storm-wise cross-validation. Phase 3: Connect to the live MOSDAC HDF5 data stream for near-real-time inference. The FastAPI REST architecture is designed for this integration.

**Q: What are the limitations?**
A: (1) Models currently trained on synthetic data — real performance requires genuine satellite + IBTrACS training. (2) No NWP ensemble blending for track prediction. (3) The RI threshold definition (30 kt/24h) is for the Atlantic basin; IMD may use a slightly different operational standard for the North Indian Ocean. (4) Microwave channels (SSMIS/GMI) are not yet integrated, limiting inner-core structural analysis. (5) No pressure-wind validation against Dvorak T-number estimates.