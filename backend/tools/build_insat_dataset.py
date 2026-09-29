"""Build a labeled dataset from the Kaggle 'INSAT3D Infrared & Raw Cyclone
Imagery' archive + its intensity CSV.

The dataset ships a CSV (`img_name,label`) where the label is the storm's wind
intensity in KNOTS, encoded in the image file names themselves (25.jpg -> 25 kt).
The IR images are storm-centered crops, so no IBTrACS geo-matching is needed:

  * every dated/named image in the CSV becomes a cyclone patch (128x128, two
    channel [IR, blurred-IR] to match the detector), category assigned from the
    knots via the IMD band scale;
  * non-cyclone patches are pulled live from NASA GIBS: N random non-storm ocean
    boxes in the North Indian Ocean (real IR, current scene).

Usage:
    python backend/tools/build_insat_dataset.py \
        --images C:\\path\\to\\extracted_ir_cyclone_ds \
        --labels C:\\Users\\albin\\Downloads\\insat_3d_ds\\ -\\ Sheet.csv \\
        --nocylone 60
"""

import argparse
import csv
import glob
import os
import random
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "data", "dataset")
PATCH = 128

# IMD intensity bands: (name, min_knots) ordering low->high
_BANDS = [
    ("Low", 0), ("Depression", 17), ("Deep Depression", 28),
    ("Cyclonic Storm", 34), ("Severe Cyclonic Storm", 48),
    ("Very Severe Cyclonic Storm", 64), ("Extremely Severe Cyclonic Storm", 90),
    ("Super Cyclonic Storm", 120),
]


def _band(wind):
    name = "Low"
    for b, lo in _BANDS:
        if wind >= lo:
            name = b
    return name


def _two_channel(gray):
    gray = cv2.resize(gray, (PATCH, PATCH), interpolation=cv2.INTER_AREA)
    ir = gray.astype(np.float32) / 255.0
    wv = cv2.GaussianBlur(gray, (9, 9), 0).astype(np.float32) / 255.0
    return np.stack([ir, wv], axis=0)


def _find_file(images_root, name):
    for pat in (name, name.replace(".jpg", ".jpeg"), name.replace(".jpg", ".png"),
                name.lower()):
        hits = glob.glob(os.path.join(images_root, "**", pat), recursive=True)
        if hits:
            return hits[0]
    return None


def _add_synth_negatives(noc_dir, count, seed=7):
    """Bright, storm-like distractors labeled NON-cyclone so the detector learns to
    key on inner-core structure rather than raw brightness. Synthetic only."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(count):
        g = np.zeros((PATCH, PATCH), dtype=np.float32)
        kind = i % 4
        if kind == 0:  # concentric bright blob w/o eye (plain warm blob)
            cy, cx = rng.integers(40, 88), rng.integers(40, 88)
            rgy = rng.integers(16, 34)
            yy, xx = np.mgrid[0:PATCH, 0:PATCH]
            g[:] = rng.uniform(0.35, 0.6)
            g += 0.9 * np.exp(-(((yy - cy) / rgy) ** 2 + ((xx - cx) / rgy) ** 2))
        elif kind == 1:  # multiple separate bright blobs (convective but not one system)
            g[:] = rng.uniform(0.35, 0.55)
            for _ in range(rng.integers(3, 6)):
                cy, cx = rng.integers(20, 108), rng.integers(20, 108)
                r = rng.integers(8, 20)
                yy, xx = np.mgrid[0:PATCH, 0:PATCH]
                g += 0.75 * np.exp(-(((yy - cy) / r) ** 2 + ((xx - cx) / r) ** 2))
        elif kind == 2:  # smooth mid-bright gradient (featureless bright sea/cloud base)
            t = rng.uniform(20, 40)
            grad = np.linspace(rng.uniform(0.4, 0.5), rng.uniform(0.55, 0.75), PATCH,
                               dtype=np.float32)[:, None]
            g = np.tile(grad, (1, PATCH)) + rng.normal(0, t * 0.004, g.shape)
        else:  # structured low-frequency ripple (cloud bands), no core
            per = rng.uniform(10, 24)
            yy, xx = np.mgrid[0:PATCH, 0:PATCH].astype(np.float32)
            g[:] = rng.uniform(0.4, 0.55)
            g += 0.35 * np.sin(2 * np.pi * (xx + yy) / per + rng.uniform(0, 6))
            g += 0.12 * np.sin(2 * np.pi * xx / (per * 0.7))
        g = np.clip(g, 0, 1)
        fn = f"synth_neg_{i:03d}.png"
        cv2.imwrite(os.path.join(noc_dir, fn), (g * 255).astype(np.uint8))
        rows.append([fn, "nocylone", "", "", "", "", "", ""])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="root of the extracted insat3d image archive")
    ap.add_argument("--labels", required=True, help="path to the insat_3d_ds Sheet CSV (img_name,label)")
    ap.add_argument("--nocylone", type=int, default=60, help="how many live GIBS ocean patches to add as non-cyclone")
    ap.add_argument("--synth-negatives", type=int, default=70,
                    help="how many synthetic bright-ghost distractors to add as non-cyclone")
    args = ap.parse_args()

    if not os.path.isdir(args.images):
        sys.exit(f"images root not found: {args.images}")
    if not os.path.isfile(args.labels):
        sys.exit(f"labels CSV not found: {args.labels}")

    cyc_dir = os.path.join(OUT, "cyclone")
    noc_dir = os.path.join(OUT, "nocylone")
    os.makedirs(cyc_dir, exist_ok=True)
    os.makedirs(noc_dir, exist_ok=True)
    for sub in (cyc_dir, noc_dir):
        for png in os.listdir(sub):
            if png.lower().endswith(".png"):
                os.remove(os.path.join(sub, png))

    rows, cyc, missing = [], 0, []
    with open(args.labels, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            img_name = rec["img_name"].strip()
            try:
                knots = float(rec["label"].strip())
            except ValueError:
                continue
            fp = _find_file(args.images, img_name)
            if fp is None:
                missing.append(img_name)
                continue
            gray = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                missing.append(img_name)
                continue
            g = cv2.resize(gray, (PATCH, PATCH), interpolation=cv2.INTER_AREA)
            fn = f"insat_{int(knots)}_{img_name.rsplit('.', 1)[0]}.png"
            cv2.imwrite(os.path.join(cyc_dir, fn), g)
            rows.append([fn, "cyclone", "", "", "", "", knots, _band(knots)])
            cyc += 1

    # Non-cyclone: live GIBS ocean boxes (non-storm NIO waters)
    sys.path.insert(0, ROOT)
    from backend.app.services.satellite_service import fetch_extent, to_gray_ir

    rng = random.Random(7)
    noc = 0
    for i in range(args.nocylone):
        lon = rng.uniform(55.0, 99.0)
        lat = rng.uniform(-5.0, 24.0)
        try:
            res = fetch_extent(lon - 4, lat - 4, lon + 4, lat + 4)
        except Exception:
            continue
        gray, valid = to_gray_ir(res)
        if gray.size == 0 or np.mean(valid) < 0.3 or gray.std() < 0.06:
            continue  # too empty or featureless
        g = cv2.resize((gray * 255).astype(np.uint8), (PATCH, PATCH),
                       interpolation=cv2.INTER_AREA)
        fn = f"gibs_noc_{i:03d}.png"
        cv2.imwrite(os.path.join(noc_dir, fn), g)
        rows.append([fn, "nocylone", "", "", "", "", "", ""])
        noc += 1

    # Synthetic bright-ghost distractors (storm-like but non-cyclone)
    rows += _add_synth_negatives(noc_dir, args.synth_negatives)

    with open(os.path.join(OUT, "labels.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "lat", "lon", "iso_time", "storm_id", "wind_knots", "category"])
        w.writerows(rows)

    print(f"cyclone patches: {cyc}   (missing images: {len(missing)} -> {missing[:8]})")
    print(f"nocylone (GIBS) patches: {noc} + synthetic distractors: {args.synth_negatives}")
    print(f"total rows in labels.csv: {len(rows)}")
    print("now run:  python backend/tools/train_cnn.py --epochs 15")


if __name__ == "__main__":
    main()
