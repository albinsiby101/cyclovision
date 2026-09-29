"""Ingest the Kaggle 'INSAT3D Infrared & Raw Cyclone Imagery (2012-2021)' dataset.

The dataset ships two zips (insat3d_ir_cyclone_ds.zip / insat3d_raw_cyclone_ds.zip)
plus a CSV mapping each image name to intensity (knots). Extract them anywhere and
point --root at that folder (it scans recursively).

Design notes:
  * Timestamps are parsed from the file names (many patterns supported). If an
    image cannot be dated it is skipped with a note.
  * For each dated image we look up an active storm in models/ibtracs_ni_storms.json
    (same +-6h window the labeler uses) and append a manifest row whose geo box is
    CENTERED on the storm position (+- --pad degrees). That way both full-disk
    frames and storm-centered crops land the storm near the center of the 128x128
    patch the labeler later cuts.
  * Frames with no matching storm still get a manifest row with the default INSAT
    full-disk box, so the labeler can draw random non-cyclone patches from them.
  * Existing manifest rows are preserved (append mode) and duplicates skipped.

Usage:
    python backend/tools/ingest_kaggle_insat3d.py --root data\\satellite_raw\\kaggle_insat3d
    python backend/tools/build_image_dataset.py      # then label + crop
"""

import argparse
import csv
import glob
import os
import re
import sys
from datetime import datetime, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE = os.path.join(ROOT, "models", "ibtracs_ni_storms.json")
DEFAULT_MANIFEST = os.path.join(ROOT, "data", "satellite_raw", "manifest.csv")
DEFAULT_INSAT_BBOX = (15.0, -5.0, 115.0, 45.0)  # full-disk approx for no-match frames
WINDOW_HOURS = 6

_TS_RE = [
    re.compile(r"(?P<y>\d{4})[-_./]?(?P<m>\d{2})[-_./]?(?P<d>\d{2})[T _.-](?P<h>\d{2})[:.-]?(?P<mi>\d{2})?"),
    re.compile(r"(?P<d>\d{2})[-_.](?P<m>\d{2})[-_.](?P<y>\d{4})[T _.-](?P<h>\d{2})[:.-]?(?P<mi>\d{2})?"),
    re.compile(r"(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})(?P<h>\d{2})(?P<mi>\d{2})"),
]

_IMGS = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG")


def _parse_dt(name):
    base = os.path.splitext(os.path.basename(name))[0]
    for rx in _TS_RE:
        for m in rx.finditer(base):
            y, mo, d = int(m["y"]), int(m["m"]), int(m["d"])
            hh = int(m.groupdict().get("h") or 0)
            mi = int(m.groupdict().get("mi") or 0)
            if not (1995 <= y <= 2032 and 1 <= mo <= 12 and 1 <= d <= 31 and hh <= 23 and mi <= 59):
                continue
            try:
                return datetime(y, mo, d, hh, mi)
            except ValueError:
                continue
    return None


def _load_storms():
    import json
    if not os.path.exists(CACHE):
        return {}
    with open(CACHE, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_active(storms, t):
    low = t - timedelta(hours=WINDOW_HOURS)
    high = t + timedelta(hours=WINDOW_HOURS)
    out = []
    for sid, storm in storms.items():
        for row in storm.get("track", []):
            rt = row.get("iso_time", "")
            if not rt:
                continue
            try:
                rt_dt = datetime.strptime(rt[:19], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            if low <= rt_dt <= high:
                out.append((row["latitude"], row["longitude"]))
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder with the extracted Kaggle INSAT3D images (recursed)")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--pad", type=float, default=10.0, help="half-width of the storm-centered box in degrees")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"root folder not found: {args.root}")
        sys.exit(1)
    storms = _load_storms()

    files = []
    for pat in _IMGS:
        files.extend(glob.glob(os.path.join(args.root, "**", pat), recursive=True))
    files = sorted(set(files))
    print(f"found {len(files)} image files under {args.root}")

    existing = set()
    if os.path.exists(args.manifest):
        with open(args.manifest, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("file"):
                    existing.add(row["file"].strip())

    rows, skipped = [], 0
    for fp in files:
        fn = os.path.basename(fp)
        if fn in existing:
            continue
        t = _parse_dt(fn)
        if t is None:
            skipped += 1
            continue
        active = _find_active(storms, t)
        cols_rel = os.path.relpath(fp, args.root)
        rel = cols_rel.replace(os.path.sep, "/")
        if active:
            lat, lon = active[0]
            rows.append((rel, t.strftime("%Y-%m-%d %H:%M:%S"),
                         lon - args.pad, lat - args.pad,
                         lon + args.pad, lat + args.pad))
        else:
            w, s, e, n = DEFAULT_INSAT_BBOX
            rows.append((rel, t.strftime("%Y-%m-%d %H:%M:%S"), w, s, e, n))

    header_needed = not os.path.exists(args.manifest)
    with open(args.manifest, "a", newline="", encoding="utf-8-sig") as f:
        wcsv = csv.writer(f)
        if header_needed:
            wcsv.writerow(["file", "iso_time", "west", "south", "east", "north"])
        for r in rows:
            wcsv.writerow(r)

    print(f"appended {len(rows)} rows to {args.manifest}  (skipped datetime-less: {skipped})")
    matched = sum(1 for r in rows if r[4] != DEFAULT_INSAT_BBOX[2])
    print(f"  -> {matched} frames matched an active cyclone (labeled), "
          f"{len(rows) - matched} will contribute non-cyclone patches")
    print("now run:  python backend/tools/build_image_dataset.py")


if __name__ == "__main__":
    main()