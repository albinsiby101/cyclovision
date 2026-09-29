"""Near-real-time satellite imagery access via NASA GIBS WMTS (EPSG:4326, 512px tiles).

Serves the *latest* available pass for a GPS bounding box ("best" endpoint), stitches
tiles into a single image, and returns georeferenced metadata so the vision chain knows
exactly what it looked at and when.

Tile math (verified against GIBS WMTSCapabilities.xml):
  top-left corner = (-180, +90); tile size 512.
  For zoom z: matrix width  W = 5*2^(z-2)  (z>=2),  W(0)=2, W(1)=3
              matrix height H = ceil(W/2)
  degrees per tile: dx = 360/W, dy = 180/H
  tile col = floor((lon+180)/dx), tile row = floor((90-lat)/dy)
"""

import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_GIBS = "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best"
_USER_AGENT = os.getenv("CYCLOVISION_GIBS_UA",
                        "CycloVision/0.1 (cyclovision-demo@example.com)")

# Layer id -> name (earthdata.nasa.gov/layers/...). All are thermal IR (BrightnessTemp
# Band 13 ~10.3um, gray ramp: brighter pixel = colder cloud top).
_LAYERS = {
    "himawari": "Himawari_AHI_Band13_Clean_Infrared",   # Indian Ocean / Asia-Pacific
    "goes_east": "GOES-East_ABI_Band13_Clean_Infrared", # Americas / Atlantic
    "goes_west": "GOES-West_ABI_Band13_Clean_Infrared", # Pacific
}


def layer_for_lon(lon: float) -> str:
    """Pick the geostationary disk that covers a longitude band."""
    if lon >= 60.0:
        return _LAYERS["himawari"]
    if lon < -60.0:
        return _LAYERS["goes_west"]
    return _LAYERS["goes_east"]


def _matrix_dims(z: int) -> Tuple[int, int]:
    if z <= 0:
        return 2, 1
    if z == 1:
        return 3, 2
    w = 5 * (2 ** (z - 2))
    return w, int(np.ceil(w / 2))


# The three Clean-IR (Band 13) layers each expose ONLY the '2km' TileMatrixSet on
# the EPSG4326 grid (verified against GIBS capabilities). Native instantiated
# resolution is matrix 5: w=40, h=20 tiles of 512px -> 9deg/tile ~ 1.9 km/px.
_NATIVE_TMS = "2km"
_NATIVE_Z = 5
_DEG_PER_TILE = 9.0  # 360 / 40 for the 2km set at z=5
_MATRIX = {"w": 40, "h": 20}


def _pick_zoom(_span: float) -> int:
    return _NATIVE_Z


def _deg_per_tile(z: int) -> float:
    return _DEG_PER_TILE


def _fetch_tile(layer: str, z: int, row: int, col: int,
                tms: str = _NATIVE_TMS) -> Optional[np.ndarray]:
    url = f"{_GIBS}/{layer}/default/{tms}/{z}/{row}/{col}.png"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            buf = resp.read()
    except Exception:
        return None
    arr = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)
    return arr


def fetch_extent(west: float, south: float, east: float, north: float,
                 max_dim: int = 2048) -> Dict:
    """Fetch the latest-available IR imagery covering a lat/lon box.

    Returns dict(image_np RGBA-stitched uint8, mask_np bool valid, bbox,
    cloud_time str/None, layer, zoom, tiles_fetched).
    """
    if west > east:
        west, east = east, west
    if north <= south:
        north, south = south, north
    west = max(-180.0, west); east = min(180.0, east)
    south = max(-90.0, south); north = min(90.0, north)

    lon = (west + east) / 2.0
    lat = (south + north) / 2.0
    layer = layer_for_lon(lon)

    lon_span = east - west
    lat_span = north - south
    z = _pick_zoom(max(lon_span, lat_span))
    w, h = _matrix_dims(z)
    dx = 360.0 / w
    dy = 180.0 / h

    c0 = int(np.floor((west + 180.0) / dx))
    c1 = int(np.floor((east + 180.0) / dx))
    r0 = int(np.floor((90.0 - north) / dy))
    r1 = int(np.floor((90.0 - south) / dy))
    c0 = max(0, min(w - 1, c0)); c1 = max(0, min(w - 1, c1))
    r0 = max(0, min(h - 1, r0)); r1 = max(0, min(h - 1, r1))

    cols = c1 - c0 + 1
    rows = r1 - r0 + 1
    if cols * rows > 16:
        raise ValueError(f"extent too large ({cols}x{rows} tiles); shrink the box")

    canvas = np.zeros((rows * 512, cols * 512, 4), dtype=np.uint8)
    mask = np.zeros((rows * 512, cols * 512), dtype=bool)
    fetched = 0
    targets = [(r0 + rr, c0 + cc, rr, cc) for rr in range(rows) for cc in range(cols)]
    with ThreadPoolExecutor(max_workers=min(8, len(targets))) as pool:
        results = list(pool.map(lambda t: (t, _fetch_tile(layer, z, t[0], t[1])), targets))
    for (rr_, cc_, rr, cc), tile in results:
        if tile is None:
            continue
        fetched += 1
        y0, x0 = rr * 512, cc * 512
        if tile.shape[0] != 512 or tile.shape[1] != 512:
            tile = cv2.resize(tile, (512, 512))
        if tile.ndim == 2:
            tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2RGBA)
        elif tile.shape[2] == 3:
            tile = cv2.cvtColor(tile, cv2.COLOR_RGB2RGBA)
        canvas[y0:y0 + 512, x0:x0 + 512] = tile
        mask[y0:y0 + 512, x0:x0 + 512] = tile[:, :, 3] > 0

    if fetched == 0 or mask.mean() < 0.10:
        raise RuntimeError(
            "No IR satellite coverage for this location (outside geostationary "
            "disk or server rejected request).")

    # Downsample if the stitch exceeds the model-friendly dimension.
    scale = max_dim / max(canvas.shape[0], canvas.shape[1])
    if scale < 1.0:
        new_shape = (int(round(canvas.shape[1] * scale)), int(round(canvas.shape[0] * scale)))
        canvas = cv2.resize(canvas, new_shape, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask.astype(np.uint8), new_shape, interpolation=cv2.INTER_AREA) > 0

    return {
        "image": canvas,  # (H, W, 4) or (H, W, 3); invalid region transparent/0
        "mask": mask,
        "bbox": {"west": west, "south": south, "east": east, "north": north},
        "layer": layer,
        "zoom": z,
        "tiles_fetched": fetched,
        "mean_fill": None,  # filled by prepare_ir_rgb() when needed
    }


def to_gray_ir(fetch: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Convert the GIBS stitch into a float32 0-1 IR frame + validity mask.

    Band 13 is a gray-scale ColdCloud warmth map: bright = cold cloud tops (the
    signature we classify on), dark = warm surface. Invalid (off-disk) pixels are
    filled with the median of the valid scene so stats/resampling stay sane.
    """
    img = fetch["image"]
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
    elif img.ndim == 3 and img.shape[2] == 3:
        alpha = np.ones(img.shape[:2], dtype=np.float32)
        rgb = img.astype(np.float32)
    else:
        alpha = np.ones(img.shape, dtype=np.float32)
        rgb = np.repeat(img[..., None], 3, axis=2)
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    valid = alpha > 0.05
    med = np.median(gray[valid]) if valid.any() else 0.0
    gray = np.where(valid, gray, med)
    return gray.astype(np.float32) / 255.0, valid