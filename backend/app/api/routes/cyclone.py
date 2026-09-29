"""
CycloVision API routes — IBTrACS RandomForest forecast chain with optional CNN vision.

GET  /health                      -> service health + model readiness
GET  /api/storms                  -> North Indian Ocean storm catalog (cached from IBTrACS)
GET  /api/demo/storm/{key}        -> full forecast for a cached storm's latest observation
POST /api/forecast                -> full forecast for an arbitrary observation (JSON)
POST /api/analyse                 -> CNN image analysis -> derived observation -> forecast (multipart)

The forecast chain runs on tabular observation features; when satellite imagery is
uploaded, the CNN stack first derives the current state (category -> wind/pressure),
which is then fed into the same RandomForest chain.
"""

import json
import os
import threading

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import TypeAdapter

from backend.app.schemas.cyclone import (
    ForecastResponse,
    HealthResponse,
    ObservationInput,
    StormSummary,
    StormTrackPoint,
)
from backend.app.services import forecast_service, vision_service, satellite_service, climatology_service

router = APIRouter()

_CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))),
                           "models", "ibtracs_ni_storms.json")
_cache_lock = threading.Lock()
_cache = None


def _load_cache():
    global _cache
    if _cache is not None:
        return _cache
    with _cache_lock:
        if _cache is not None:
            return _cache
        if not os.path.exists(_CACHE_PATH):
            _cache = {}
        else:
            with open(_CACHE_PATH, "r", encoding="utf-8") as f:
                _cache = json.load(f)
    return _cache


def _normalise(key: str) -> str:
    return key.replace("demo_cyclone_", "").replace("demo_", "")


def _category_bands():
    art = forecast_service._load()
    return list(art.get("cat_bands", [])) if art else []


def _band_name(cat_bands, wind):
    label = "Depression"
    for name, lo in cat_bands:
        if wind >= lo:
            label = name
    return label


def _imd_category(wind: float) -> str:
    return _band_name([("Depression", 17), ("Deep Depression", 28), ("Cyclonic Storm", 34),
                       ("Severe Cyclonic Storm", 48), ("Very Severe Cyclonic Storm", 64),
                       ("Extremely Severe Cyclonic Storm", 90), ("Super Cyclonic Storm", 120)], wind)


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    return HealthResponse(
        status="ok",
        application="CycloVision",
        version="1.0.0",
        models_loaded=forecast_service.models_ready(),
        mode="operational",
        device="cpu",
    )


def _obs_from_row(row: dict, name: str = None) -> ObservationInput:
    return ObservationInput(
        name=name,
        latitude=float(row["LAT"]),
        longitude=float(row["LON"]),
        wind_knots=float(row["WIND"]),
        pressure_hpa=float(row["PRES"]),
        storm_speed_knots=float(row["STORM_SPEED"]),
        storm_direction_deg=float(row["STORM_DIR"]),
        month=int(row["MONTH"]),
        hour=int(row["HOUR"]),
        obs_num=int(row["OBS_NUM"]),
    )


def _storm_summaries() -> list:
    cache = _load_cache()
    if not cache:
        return []
    bands = _category_bands()
    out = []
    for key, storm in cache.items():
        rows = storm.get("track", [])
        if not rows:
            continue
        last = rows[-1]
        current_wind = last["wind_knots"]
        points = []
        for r in rows:
            points.append(StormTrackPoint(
                iso_time=r["iso_time"], latitude=r["latitude"], longitude=r["longitude"],
                wind_knots=r["wind_knots"], category=_band_name(bands, r["wind_knots"]),
            ))
        try:
            obs = _obs_from_row(storm["last_row"], storm.get("name"))
            nxt = forecast_service.track_forecast(obs)
            _w = forecast_service.intensity_forecast(obs)["predicted_wind_knots"]
            points.append(StormTrackPoint(
                iso_time="+6h (AI)", latitude=nxt["latitude"], longitude=nxt["longitude"],
                wind_knots=_w, category=_imd_category(_w), is_forecast=True,
            ))
        except Exception:
            pass
        out.append(StormSummary(
            storm_id=str(storm.get("storm_id", key)),
            name=str(storm.get("name", key)).upper(),
            basin="NI",
            current_category=_band_name(bands, current_wind),
            current_wind_knots=current_wind,
            latitude=last["latitude"],
            longitude=last["longitude"],
            last_updated=last["iso_time"],
            track=points,
        ))
    return out


@router.get("/api/storms", response_model=list[StormSummary], tags=["catalog"])
def list_storms():
    return _storm_summaries()


@router.get("/api/demo/storm/{key}", response_model=ForecastResponse, tags=["catalog"])
def demo_storm(key: str):
    cache = _load_cache()
    entry = cache.get(key)
    if entry is None:
        target = _normalise(key)
        for k, v in cache.items():
            if _normalise(k) == target:
                entry, key = v, k
                break
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No cached storm for '{key}'")
    obs = _obs_from_row(entry["last_row"], entry.get("name"))
    try:
        payload = forecast_service.full_forecast(obs, source_mode="DEMO_FORECAST",
                                                 storm_id=str(entry.get("storm_id", key)),
                                                 name=str(entry.get("name", key)))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return TypeAdapter(ForecastResponse).validate_python(payload)


@router.post("/api/forecast", response_model=ForecastResponse, tags=["forecast"])
def forecast(obs: ObservationInput):
    try:
        payload = forecast_service.full_forecast(obs, source_mode="MODEL_FORECAST", name=obs.name)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return TypeAdapter(ForecastResponse).validate_python(payload)


def _decode_upload(file: UploadFile, field: str) -> np.ndarray:
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail=f"'{field}' is empty")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None or img.size == 0:
        raise HTTPException(status_code=400, detail=f"'{field}' could not be decoded as an image")
    return img


@router.post("/api/analyse", tags=["vision"], response_model=None)
async def analyse(
    ir_image: UploadFile = File(..., description="IR channel satellite image (required)"),
    wv_image: UploadFile = File(None, description="WV channel satellite image (optional)"),
    name: str = Form(None, max_length=64),
    latitude: float = Form(13.7),
    longitude: float = Form(86.3),
    storm_speed_knots: float = Form(15.0),
    storm_direction_deg: float = Form(280.0),
    month: int = Form(None, ge=1, le=12),
    hour: int = Form(None, ge=0, le=23),
):
    """
    CNN image analysis -> derived tabular observation -> RandomForest forecast chain.

    The IR channel (and optional WV) are fed to the ResNet-18 detector, the
    CNN+ConvLSTM intensity model and the RI early-warning model. The predicted
    IMD category maps to a wind/pressure observation, which the RF chain then
    extends into the 6 h track, formation probability and 24/48/72 h outlook.
    """
    if not vision_service.models_ready():
        raise HTTPException(status_code=503, detail="CNN vision checkpoints not found on disk")

    import datetime as _dt

    utc_now = _dt.datetime.now(_dt.timezone.utc)
    obs = ObservationInput(
        name=name,
        latitude=latitude,
        longitude=longitude,
        wind_knots=0.0,  # overridden by CNN-derived value inside analyse_observation
        pressure_hpa=1013.0,  # overridden by CNN-derived value
        storm_speed_knots=storm_speed_knots,
        storm_direction_deg=storm_direction_deg,
        month=month or utc_now.month,
        hour=hour if hour is not None else utc_now.hour,
        obs_num=0,
    )

    ir_img = _decode_upload(ir_image, "ir_image")
    wv_img = _decode_upload(wv_image, "wv_image") if wv_image is not None else None

    try:
        return vision_service.analyse_observation(obs, ir_img, wv_img)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


def _season_note(month: int) -> str:
    if month in (4, 5, 6):
        return "pre-monsoon season (Apr-Jun) — Arabian Sea & Bay of Bengal active"
    if month in (10, 11, 12):
        return "post-monsoon season (Oct-Dec) — Bay of Bengal peak activity"
    if month in (7, 8, 9):
        return "monsoon season (Jul-Sep) — subdued cyclogenesis"
    return "early season (Jan-Mar) — very low historical activity"


@router.get("/api/satellite/analyse", tags=["satellite"], response_model=None)
async def satellite_analyse(
    west: float = Query(..., ge=-180, le=180),
    south: float = Query(..., ge=-90, le=90),
    east: float = Query(..., ge=-180, le=180),
    north: float = Query(..., ge=-90, le=90),
    month: int = Query(None, ge=1, le=12),
    hour: int = Query(None, ge=0, le=23),
    storm_speed_knots: float = Query(15.0, ge=0, le=200),
    storm_direction_deg: float = Query(280.0, ge=0, le=360),
):
    """
    Fetch the latest-available thermal IR pass covering the GPS box, then run the
    CNN -> RF chain and report cyclone-proneness (climatology + imagery verdict).
    """
    import datetime as _dt

    if not vision_service.models_ready():
        raise HTTPException(status_code=503, detail="CNN vision checkpoints not found on disk")
    if north <= south or east <= west:
        raise HTTPException(status_code=400, detail="invalid box: require west<east and south<north")

    utc_now = _dt.datetime.now(_dt.timezone.utc)
    month = month or utc_now.month
    hour = hour if hour is not None else utc_now.hour

    try:
        fetch = satellite_service.fetch_extent(west, south, east, north)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    gray, _ = satellite_service.to_gray_ir(fetch)

    lat = (south + north) / 2.0
    lon = (west + east) / 2.0
    obs = ObservationInput(
        name="satellite-gps-box",
        latitude=lat,
        longitude=lon,
        wind_knots=0.0,
        pressure_hpa=1013.0,
        storm_speed_knots=storm_speed_knots,
        storm_direction_deg=storm_direction_deg,
        month=month,
        hour=hour,
        obs_num=0,
    )
    try:
        vision = vision_service.analyse_observation(obs, gray, None)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    prone = climatology_service.query(lat, lon, month)
    prone_label = "LOW" if prone < 0.3 else ("MODERATE" if prone < 0.6 else "HIGH")
    vision["satellite"] = {
        "bbox": fetch["bbox"],
        "center_lat": round(lat, 4),
        "center_lon": round(lon, 4),
        "layer": fetch["layer"],
        "zoom": fetch["zoom"],
        "tiles_fetched": fetch["tiles_fetched"],
        "frame_shape": list(fetch["image"].shape[:2]),
        "source": "NASA GIBS near-real-time (latest pass)",
        "utc_hint": utc_now.isoformat(),
        "month": month,
    }
    vision["prone"] = {
        "climatology_index": round(prone, 3),
        "level": prone_label,
        "season_note": _season_note(month),
        "basis": "IBTrACS tracked-storm frequency by month",
    }
    frame_b64 = _frame_preview(fetch["image"])
    if frame_b64:
        vision["satellite"]["frame_preview_b64"] = frame_b64
        vision["satellite"]["frame_composite"] = "IR gray stretch (bright = cold cloud tops)"
    return vision


def _frame_preview(rgba: np.ndarray, maxw: int = 320) -> str:
    import base64
    try:
        img = rgba.copy()
        if img.ndim == 3 and img.shape[2] >= 4:
            a = (img[:, :, 3:4].astype(np.float32) / 255.0)
            img = (img[:, :, :3].astype(np.float32) * a).astype(np.uint8)
        if img.max() <= 1.0 and img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        h, w = img.shape[:2]
        if w > maxw:
            img = cv2.resize(img, (maxw, int(round(h * maxw / w))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 80])
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    except Exception:
        return ""


@router.get("/api/climatology", tags=["satellite"], response_model=None)
def climatology(month: int = Query(6, ge=1, le=12)):
    try:
        payload = climatology_service.render_png(month)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    payload["month"] = month
    payload["season_note"] = _season_note(month)
    return payload