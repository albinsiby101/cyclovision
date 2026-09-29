"""Build a labeled cyclone/non-cyclone image dataset from raw IR frames.

Frames come from `data/satellite_raw/`. Each needs geo-registration + timestamp.
Two supported layouts:

  1) A `manifest.csv` in that folder (recommended, plain-PNG friendly):
       file,iso_time,west,south,east,north
       himawari_20250601_00.png,2025-06-01 00:00:00,45,-25,225,55

  2) If a folder contains only PNG/JPEG frames, we attempt the legacy demo layout
     (data/demo/satellite_sequences/<storm>/frame_*_ir_enhanced.png) and treat
     none other than the demo set as labeled data. Practically you will use the
     manifest.

Labeling is AUTOMATIC: it cross-references models/ibtracs_ni_storms.json with each
frame's timestamp (within a time window) and crops 128x128 patches centered on
active storm positions -> cyclone/; random non-storm patches -> nocylone/.
Every patch carries its ground-truth lat/lon, time, and IMD category in labels.csv
so training can pass position/time context to the models later if desired.

Output:
  data/dataset/cyclone/*.png
  data/dataset/nocylone/*.png
  data/dataset/labels.csv     (file,label,lat,lon,iso_time,storm_id,category_knots)
"""

import csv
import os
import random
from datetime import datetime, timedelta

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = os.path.join(ROOT, "data", "satellite_raw")
CACHE = os.path.join(ROOT, "models", "ibtracs_ni_storms.json")
OUT = os.path.join(ROOT, "data", "dataset")
PATCH = 128
WINDOW_HOURS = 6      # storm position must be within this window of the frame time
K_RANDOM = 4          # non-storm patches sampled per frame
MIN_ACTIVE_DEG = 6.0  # "active" storm patch must be this far kept


def _load_manifest():
    path = os.path.join(RAW, "manifest.csv")
    if not os.path.exists(path):
        return None
    entries = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            entries.append({
                "file": row["file"].strip(),
                "iso_time": row["iso_time"].strip(),
                "west": float(row["west"]), "south": float(row["south"]),
                "east": float(row["east"]), "north": float(row["north"]),
            })
    return entries


def _load_storms():
    import json
    if not os.path.exists(CACHE):
        return {}
    with open(CACHE, "r", encoding="utf-8") as f:
        return json.load(f)


_TIME_FMTS = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
              "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M",
              "%Y-%m-%dT%H:%M:%S", "%d-%m-%Y %H:%M:%S"]


def _parse_dt(s):
    for fmt in _TIME_FMTS:
        try:
            return datetime.strptime(s[:19].strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"unrecognized timestamp: {s!r}")


def _frame_time(iso: str):
    return _parse_dt(iso)


def _find_active(storms, t):
    """Return [(storm_id, name, lat, lon, wind_knots)] with a track row near t."""
    out = []
    low = t - timedelta(hours=WINDOW_HOURS)
    high = t + timedelta(hours=WINDOW_HOURS)
    for sid, storm in storms.items():
        for row in storm.get("track", []):
            rt = row.get("iso_time", "")
            if not rt:
                continue
            try:
                rt_dt = _parse_dt(rt)
            except ValueError:
                continue
            if low <= rt_dt <= high:
                out.append((sid, storm.get("name", "?"), float(row["latitude"]),
                            float(row["longitude"]), float(row["wind_knots"])))
                break
    return out


def _geo_to_px(lon, lat, west, south, east, north, w, h):
    x = (lon - west) / (east - west) * (w - 1)
    y = (north - lat) / (north - south) * (h - 1)
    return int(round(x)), int(round(y))


def _imd_knots_band(wind):
    for name, lo in [("Super Cyclonic Storm", 120), ("Extremely Severe Cyclonic Storm", 90),
                     ("Very Severe Cyclonic Storm", 64), ("Severe Cyclonic Storm", 48),
                     ("Cyclonic Storm", 34), ("Deep Depression", 28), ("Depression", 17)]:
        if wind >= lo:
            return name
    return "Low"


def _patch(img, cx, cy, size=PATCH):
    h, w = img.shape[:2]
    x0 = max(0, cx - size // 2); y0 = max(0, cy - size // 2)
    x1 = min(w, x0 + size); y1 = min(h, y0 + size)
    crop = img[y0:y1, x0:x1]
    if crop.shape[0] < size or crop.shape[1] < size:
        paint = np.zeros((size, size), dtype=np.uint8)
        paint[:crop.shape[0], :crop.shape[1]] = crop
        crop = paint
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def main():
    os.makedirs(os.path.join(OUT, "cyclone"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "nocylone"), exist_ok=True)
    for sub in ("cyclone", "nocylone"):
        d = os.path.join(OUT, sub)
        for png in os.listdir(d):
            if png.lower().endswith(".png"):
                os.remove(os.path.join(d, png))
    storms = _load_storms()
    entries = _load_manifest()

    if entries is None:
        print("No data/satellite_raw/manifest.csv found.")
        print()
        print("Download IR satellite frames (Zenodo) and create manifest.csv in "
              + RAW + " with columns:")
        print("  file,iso_time,west,south,east,north")
        print("e.g. himawari_20230601_12.png,2023-06-01 12:00:00,45,-25,225,55")
        print("Each frame is auto-labelled against models/ibtracs_ni_storms.json — no manual labelling.")
        return

    rng = random.Random(42)
    rows = []
    n_cyc = n_no = 0
    for ent in entries:
        path = os.path.join(RAW, ent["file"])
        if not os.path.exists(path):
            print("skip missing", ent["file"])
            continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("skip undecodable", ent["file"])
            continue
        h, w = img.shape[:2]
        t = _frame_time(ent["iso_time"])
        active = _find_active(storms, t)

        centers = []
        for sid, name, lat, lon, wind in active:
            if not (ent["west"] <= lon <= ent["east"] and ent["south"] <= lat <= ent["north"]):
                continue
            cx, cy = _geo_to_px(lon, lat, ent["west"], ent["south"],
                                ent["east"], ent["north"], w, h)
            patch = _patch(img, cx, cy)
            fn = f"cyc_{sid}_{ent['file'].rsplit('.', 1)[0]}.png"
            cv2.imwrite(os.path.join(OUT, "cyclone", fn), patch)
            rows.append([fn, "cyclone", round(lat, 3), round(lon, 3),
                         ent["iso_time"], sid, wind, _imd_knots_band(wind)])
            n_cyc += 1
            centers.append((cx, cy))

        for _ in range(K_RANDOM):
            for _try in range(40):
                cx = rng.randint(PATCH // 2, w - PATCH // 2)
                cy = rng.randint(PATCH // 2, h - PATCH // 2)
                if centers and any(abs(cx - ax) < PATCH and abs(cy - ay) < PATCH for ax, ay in centers):
                    continue
                break
            patch = _patch(img, cx, cy)
            fn = f"nocyc_{ent['file'].rsplit('.', 1)[0]}_{cy}_{cx}.png"
            cv2.imwrite(os.path.join(OUT, "nocylone", fn), patch)
            rows.append([fn, "nocylone", "", "", ent["iso_time"], "", "", ""])
            n_no += 1

    with open(os.path.join(OUT, "labels.csv"), "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["file", "label", "lat", "lon", "iso_time", "storm_id", "wind_knots", "category"])
        wcsv.writerows(rows)

    print(f"Wrote {n_cyc} cyclone + {n_no} nocylone patches -> {OUT}")
    print(f"labels.csv rows: {len(rows)}")


if __name__ == "__main__":
    main()