"""Cyclone-activity climatology derived from the cached IBTrACS storm catalog.

Builds a monthly grid of storm-point frequency (counts of tracked 6-hourly storm
positions per 1-deg cell, lightly smoothed) and renders it as an RGBA PNG heat map
the frontend can drop onto a Leaflet map as an ImageOverlay spanning the globe.

Because the cache currently holds North Indian Ocean storms, the field is
India-oriented by construction; adding more basins to models/ibtracs_ni_storms.json
extends it automatically (fresh process / reload picks it up).
"""

import base64
import json
import os
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

_CACHE_REL = os.path.join("models", "ibtracs_ni_storms.json")
_LOCK = threading.Lock()
_GRID: Optional[np.ndarray] = None
_GRID_META: Optional[dict] = None

_COLORMAP = np.array([
    (0.12, 0.62, 1.00),   # 0.00 none (light blue)
    (0.20, 0.82, 0.40),   # 0.25 low (green)
    (0.95, 0.90, 0.08),   # 0.50 moderate (yellow)
    (0.97, 0.55, 0.15),   # 0.75 high (orange)
    (0.85, 0.15, 0.10),   # 1.00 severe (red)
], dtype=np.float32)


def _coarse_land_mask() -> np.ndarray:
    """True for land cells on the (90 rows x 180 cols) 1-deg grid.

    Rough polygons for the Indian-centred region; used only to keep colours on
    water where storms actually occur.
    """
    lat = 89.5 - np.arange(90)
    lon = -179.5 + np.arange(180)
    LL, LG = np.meshgrid(lon, lat)
    land = np.zeros((90, 180), dtype=bool)

    def in_poly(l, t, poly):
        x, y = l[..., None], t[..., None]
        px = np.array([p[0] for p in poly])[None, None, :]
        py = np.array([p[1] for p in poly])[None, None, :]
        inside = np.zeros_like(x, dtype=bool)
        j = len(poly) - 1
        for i in range(len(poly)):
            yi, yj = py[:, :, i], py[:, :, j]
            xi, xj = px[:, :, i], px[:, :, j]
            cond1 = (yi > y) != (yj > y)
            cond2 = x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
            inside[cond1 & cond2] = ~inside[cond1 & cond2]
            j = i
        return inside.squeeze()

    india = [(68, 5), (81, 7), (90, 18), (95, 24), (92, 28), (80, 31),
             (74, 35), (68, 25), (66, 20), (68, 12)]
    land |= in_poly(LG, LL, india)
    arabia = [(34, 12), (48, 12), (57, 17), (59, 25), (48, 32), (39, 34), (34, 27)]
    land |= in_poly(LG, LL, arabia)
    africa = [(25, 30), (30, 5), (45, 5), (45, -5), (25, -5)]
    land |= in_poly(LG, LL, africa)
    asia = [(60, 25), (100, 25), (100, 40), (60, 40)]
    land |= in_poly(LG, LL, asia)
    australia = [(114, -35), (153, -39), (142, -11), (132, -11), (123, -14)]
    land |= in_poly(LG, LL, australia)
    return land


def _load():
    global _GRID, _GRID_META
    if _GRID is not None:
        return _GRID, _GRID_META
    with _LOCK:
        if _GRID is not None:
            return _GRID, _GRID_META
        path = _CACHE_REL
        if not os.path.exists(path):
            _GRID, _GRID_META = None, {"land": _coarse_land_mask(),
                                       "lats": (89.5 - np.arange(90)).tolist(),
                                       "lons": (-179.5 + np.arange(180)).tolist()}
            return _GRID, _GRID_META
        grid = np.zeros((12, 90, 180), dtype=np.float32)
        with open(path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        for storm in cache.values():
            for row in storm.get("track", []):
                t = row.get("iso_time", "")
                lat, lon = row.get("latitude"), row.get("longitude")
                if lat is None or lon is None or not t:
                    continue
                try:
                    month = int(t[5:7])
                except (ValueError, IndexError):
                    continue
                i = int(np.clip(round(89.5 - float(lat)), 0, 89))
                j = int(np.clip(round(float(lon) + 179.5), 0, 179))
                grid[month - 1, i, j] += 1.0
        land = _coarse_land_mask()
        grid[:, land] = 0.0
        for m in range(12):
            s = np.clip(grid[m], 0, None)
            conv = np.zeros_like(s)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    conv += np.roll(s, (di, dj), axis=(0, 1))
            conv *= 1.0 / 9.0
            hi = np.percentile(conv, 99)
            grid[m] = np.clip(conv / hi if hi > 0 else conv, 0.0, 1.0)
        _GRID, _GRID_META = grid, {"land": land,
                                   "lats": (89.5 - np.arange(90)).tolist(),
                                   "lons": (-179.5 + np.arange(180)).tolist()}
        return _GRID, _GRID_META


def _idx(lat: float, lon: float) -> Tuple[int, int]:
    i = int(np.clip(round(89.5 - float(lat)), 0, 89))
    j = int(np.clip(round(float(lon) + 179.5), 0, 179))
    return i, j


def query(lat: float, lon: float, month: int) -> float:
    """Prone index (0-1) at a point for a given month (1-12)."""
    grid, _ = _load()
    if grid is None:
        return 0.0
    i, j = _idx(lat, lon)
    return float(grid[month - 1, i, j])


def render_png(month: int, width: int = 640, height: int = 320) -> dict:
    """RGBA heat map for a month: PNG base64 data-uri + geometry for ImageOverlay."""
    grid, meta = _load()
    if grid is None:
        raise RuntimeError("climatology unavailable: no storm cache")
    comp = grid[month - 1][::-1]                      # row 0 = north (lat 90)
    h, w = comp.shape
    r = comp.copy()
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for idx in range(_COLORMAP.shape[0] - 1):
        lo, hi = _COLORMAP[idx], _COLORMAP[idx + 1]
        t0, t1 = idx / 4.0, (idx + 1) / 4.0
        band = (r >= t0) & (r < t1 + 1e-9)
        tt = (r[band] - t0) / (t1 - t0 + 1e-12)
        rgb = (lo * (1 - tt[:, None]) + hi * tt[:, None]).astype(np.uint8)
        val = np.zeros((rgb.shape[0], 4), dtype=np.uint8)
        val[:, :3] = rgb
        val[:, 3] = 255
        rgba[band] = val
    rgba[comp <= 0.0, 3] = 0
    if width != w or height != h:
        rgba = cv2.resize(rgba, (width, height), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", rgba, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return {
        "image": "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode(),
        "bounds": [[-90, -180], [90, 180]],
        "res_deg": 1.0,
        "legend": [{"t": float(v), "color": "#%02x%02x%02x" % tuple(
            int(round(float(x) * 255)) for x in (_COLORMAP[min(int(v * 4), 4)][:3] if v <= 0.75 else _COLORMAP[-1][:3]))}
            for v in [0.0, 0.25, 0.5, 0.75, 1.0]],
    }