# Satellite IR Data — How to Build the Real Dataset

Drop geostationary IR frames in THIS folder, then run auto-labeling — no manual
labeling needed.

## Fastest path: Kaggle INSAT3D dataset (recommended)

1. kaggle.com -> dataset "INSAT3D Infrared & Raw Cyclone Imagery (2012-2021)"
   (sshuba m/insat3d-...), click **Download** (free account needed).
2. Extract the archive so you have a folder of cyclone images, e.g. into
   `data\satellite_raw\kaggle_insat3d\` (subfolders/storms are fine — the tool
   recurses).
3. Auto-generate the manifest (timestamps parsed from file names, storms matched
   from the IBTrACS cache, boxes centered on each storm):
   ```powershell
   $env:PYTHONPATH="C:\Users\albin\Documents\cyclo"
   & .venv\Scripts\python.exe backend\tools\ingest_kaggle_insat3d.py --root data\satellite_raw\kaggle_insat3d
   ```
4. Label + train:
   ```powershell
   & .venv\Scripts\python.exe backend\tools\build_image_dataset.py
   & .venv\Scripts\python.exe backend\tools\train_cnn.py --epochs 15
   ```
The dataset also ships a CSV of image-name -> intensity(knots); our intensity
labels come from the IBTrACS match instead, so no extra work is needed.

## Step 1 — Download frames from Zenodo

Search Zenodo for geostationary IR imagery covering the North Indian Ocean:

- Query: `Himawari infrared 2023 Bay of Bengal`
- Suggested collections / keywords: Himawari-8/9 AHI clean-IR (Band 13), INSAT-3D 10.7 um IR, Meteosat Vis/nir, or GPM/IMERG if you want rain fields.

Pick files that are:

1. Georeferenced (NetCDF, GeoTIFF, or full-disk PNG with known projection), and
2. Timestamped (ISO time in the filename or metadata).

Full-disk PNG tilesets from Himawari works too — for those, compute the bounding
box from the projection (full disk centered ~140.7E).

Download into this folder, e.g.:

```
data/satellite_raw/himawari_20230601_00.png
data/satellite_raw/himawari_20230601_12.png
```

## Step 2 — Create manifest.csv

```
file,iso_time,west,south,east,north
himawari_20230601_00.png,2023-06-01 00:00:00,45,-25,225,55
himawari_20230601_12.png,2023-06-01 12:00:00,45,-25,225,55
```

- `iso_time`  format `YYYY-MM-DD HH:MM:SS` (local or UTC, be consistent).
- `west/south/east/north` is the TRUE geographic extent of the image in degrees
  (for a full-disk Himawari image this is roughly west=45, south=-25, east=225,
  north=55).

Tip: name the files with the timestamp so you can fill the manifest in bulk.

## Step 3 — Auto-label and train

```powershell
$env:PYTHONPATH = "C:\Users\albin\Documents\cyclo"
& .venv\Scripts\python.exe backend\tools\build_image_dataset.py
& .venv\Scripts\python.exe backend\tools\train_cnn.py --epochs 15
# restart the backend; it loads the retrained checkpoints
```

What happens:

- Every frame is matched against `models/ibtracs_ni_storms.json` (IBTrACS cache).
- Storm centers present within +-6h of the frame's `iso_time` become
  `data/dataset/cyclone/` patches; random non-storm crops become
  `data/dataset/nocylone/`; all recorded in `labels.csv` with lat/lon/time/
  category so intensity and season context can be added later.

## How many frames?

For a meaningful detector retrain: at least ~30 cyclone patches and ~120
non-cyclone patches (i.e. ~15-40 frames with active storms + many clear-sky
frames). The intensity model additionally needs a spread of IMD categories.