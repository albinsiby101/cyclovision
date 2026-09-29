# CycloVision Data Directory

This directory is where all real and processed satellite datasets are placed.

> **Note:** Demo mode works without any real datasets. The `data/demo/` folder contains pre-generated realistic demonstration sequences.

---

## Directory Structure

```
data/
├── raw/
│   ├── insat/          ← INSAT-3D/3DR satellite imagery (HDF5/BUFR/PNG)
│   ├── ibtracs/        ← IBTrACS best-track CSV
│   └── era5/           ← ERA5 NetCDF reanalysis files (optional)
├── interim/            ← Intermediate processing outputs
├── processed/          ← Final training-ready sequences (.npy arrays)
└── demo/               ← Pre-generated demo sequences (included)
    └── satellite_sequences/
        ├── demo_cyclone_amphan/     ← Super Cyclone Amphan (Bay of Bengal, 2020)
        ├── demo_biparjoy/           ← ESCS Biparjoy (Arabian Sea, 2023)
        └── demo_non_cyclone/        ← Non-cyclonic baseline convection
```

---

## Dataset 1: INSAT-3D/3DR Imagery

**Source:** MOSDAC — Meteorological & Oceanographic Satellite Data Archival Centre (ISRO)  
**URL:** https://www.mosdac.gov.in  
**Registration:** Free account required

**Channels to download:**
- `IMG_TIR1` — Thermal Infrared 1 (10.8 μm) — cyclone cloud-top temperatures
- `IMG_WV` — Water Vapour (6.8 μm) — upper tropospheric moisture

**Recommended search:**
- Date range: Cyclone season (April–December)
- Basin: Bay of Bengal (5–25°N, 80–100°E) and Arabian Sea (5–25°N, 55–75°E)
- Cadence: Half-hourly or 15-minute rapid scan

**Place downloaded files in:**
```
data/raw/insat/
```

---

## Dataset 2: IBTrACS Best-Track

**Source:** NOAA National Centers for Environmental Information (NCEI)  
**URL:** https://www.ncei.noaa.gov/products/international-best-track-archive  
**Direct CSV:** https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/IBTrACS.NI.list.v04r01.csv

**Download:**
```powershell
# PowerShell
Invoke-WebRequest -Uri "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/IBTrACS.NI.list.v04r01.csv" -OutFile "data\raw\ibtracs\IBTrACS.NI.list.v04r01.csv"
```

**Description:** Contains storm ID, name, timestamp (UTC), latitude, longitude, maximum sustained wind speed (knots), central pressure (hPa), and basin classification for all North Indian Ocean (NI) cyclones.

---

## Dataset 3: ERA5 Reanalysis (Optional)

**Source:** Copernicus Climate Data Store (ECMWF)  
**URL:** https://cds.climate.copernicus.eu  
**Registration:** Free account + API key required

**Variables relevant to RI prediction:**
- `u_component_of_wind`, `v_component_of_wind` at 200hPa and 850hPa (vertical wind shear)
- `sea_surface_temperature`
- `specific_humidity` at 500-700hPa
- `mean_sea_level_pressure`

**Place downloaded NetCDF files in:**
```
data/raw/era5/
```

---

## Dataset 4: TC PRIMED (Optional)

**Source:** Colorado State University / NOAA RAMMB  
**URL:** https://rammb-data.cira.colostate.edu/tcprimed  
**Description:** AI-ready dataset combining geostationary IR, passive microwave, ERA5, and best-track labels for global tropical cyclones. Supports plug-and-play pipeline integration.

---

## Preprocessing

After placing raw data, run:

```powershell
$env:PYTHONPATH = "C:\Users\albin\Documents\cyclo"
.\.venv\Scripts\python ml\preprocessing\preprocessor.py
```

This will:
1. Align INSAT timestamps with IBTrACS observations
2. Crop storm-centered 512×512 regions
3. Resize to 128×128
4. Normalise IR and WV channels
5. Assemble 6-frame temporal sequences
6. Export `.npy` arrays to `data/processed/`

---

## Credentials / API Keys

**Never commit credentials to the repository.**  
Place all API keys in `.env` (gitignored):

```
MOSDAC_USERNAME=your_username
MOSDAC_PASSWORD=your_password
CDS_API_KEY=your_era5_key
```