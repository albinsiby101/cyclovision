"""
Forecasting service for the IBTrACS RandomForest models.

Loads models/cyclo_models.joblib once (lazily), builds the exact feature rows
each model was trained on, and chains the three models into a forecast:
  intensity (RF regressor)      -> predicted wind + empirical category posterior
  formation (RF classifier)     -> P(reaches cyclonic strength >= 34 kt)
  track     (RF multi-output)   -> next 6 h position, rechained to 24/48/72 h
"""

import math
import os
import threading
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Dict, List

import joblib
import numpy as np

# Imported from the pandas-free constants module on purpose: pulling
# backend.ibtracs_common here would import pandas (~36 MB resident) into the
# cloud process for the sake of a literal.
from backend.app.core.cat_bands import CAT_BANDS as DEFAULT_CAT_BANDS

_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                           "models", "cyclo_models.joblib")

_lock = threading.Lock()
_artifacts = None
_load_attempted = False


def _pin_single_thread(obj):
    """Persisted forests carry n_jobs=-1, which makes sklearn spawn a fresh
    multiprocessing pool on *every* predict call on Windows (~130 ms fixed
    overhead multiplied across the 16-step re-chain). Pinning to 1 worker
    keeps single-row inference in pure tree-walk territory (microseconds)."""
    seen = set()
    stack = [obj]
    while stack:
        node = stack.pop()
        node_id = id(node)
        if node_id in seen:
            continue
        seen.add(node_id)
        if hasattr(node, "n_jobs"):
            try:
                node.n_jobs = 1
            except Exception:
                pass
        for attr in ("estimators_", "estimators"):
            if hasattr(node, attr):
                try:
                    stack.extend(list(getattr(node, attr)))
                except Exception:
                    pass


def _strip_feature_names(obj):
    """The forests were fitted on DataFrames, so sklearn expects named columns.
    The API feeds dense float arrays (pandas-free), so drop the recorded names
    to avoid a "X does not have valid feature names" warning on every call."""
    seen = set()
    stack = [obj]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if hasattr(node, "feature_names_in_"):
            try:
                del node.feature_names_in_
            except Exception:
                pass
        for attr in ("estimators_", "estimators"):
            if hasattr(node, attr):
                try:
                    stack.extend(list(getattr(node, attr)))
                except Exception:
                    pass


def _load():
    global _artifacts, _load_attempted
    if _load_attempted:
        return _artifacts
    with _lock:
        if _load_attempted:
            return _artifacts
        _load_attempted = True
        if not os.path.exists(_MODEL_PATH):
            return None
        _artifacts = joblib.load(_MODEL_PATH)
        for key in ("intensity", "formation", "track"):
            if key in _artifacts:
                _pin_single_thread(_artifacts[key])
                _strip_feature_names(_artifacts[key])
        if "cat_bands" not in _artifacts:
            _artifacts["cat_bands"] = DEFAULT_CAT_BANDS
    return _artifacts


def models_ready() -> bool:
    return _load() is not None


def _band(cat_bands, wind):
    label = "Depression"
    for name, lo in cat_bands:
        if wind >= lo:
            label = name
    return label


def _raw_forest(rf, X):
    """Per-tree RF inference via each tree's Cython Tree.apply().

    Equivalent aggregate to RandomForestRegressor.predict but ~30-60x faster in
    the app process: sklearn's wrapper validates + dispatches per tree (~0.4 ms /
    tree under a warm torch import), while tree.tree_.apply() is a direct Cython
    intp32->intp64 leaf walk. Leaves are then harvested via tree_.value, and the
    mean over trees is bit-close to predict() (regression mean over leaf values).
    """
    X32 = np.ascontiguousarray(X, dtype=np.float32)
    out = np.empty((X32.shape[0], len(rf.estimators_)), dtype=np.float64)
    for t, tr in enumerate(rf.estimators_):
        nid = tr.tree_.apply(X32)
        out[:, t] = tr.tree_.value[nid, 0, 0]
    return out


def _fast_rf_mean(rf, X):
    return _raw_forest(rf, X).mean(axis=1)


def _regress_multi(model, X):
    """Vectorized MultiOutputRegressor/RandomForestRegressor prediction."""
    from sklearn.ensemble import RandomForestRegressor as _RF
    from sklearn.multioutput import MultiOutputRegressor

    rfs = model.estimators_ if isinstance(model, MultiOutputRegressor) else [model]
    if not all(isinstance(r, _RF) for r in rfs):
        raise TypeError("unsupported regressor type")
    out = [_fast_rf_mean(r, X) for r in rfs]
    return np.column_stack(out) if len(out) > 1 else out[0]


def _mk_row(obs) -> dict:
    """Build the superset feature row; each model picks its own columns.

    A plain dict (not a DataFrame) keeps pandas out of the runtime; the models
    only ever see a dense float64 array via _row_array(), which reproduces the
    exact column order/typing the training pipeline produced.
    """
    return {
        "LAT": float(obs.latitude), "LON": float(obs.longitude),
        "WIND": float(obs.wind_knots), "PRES": float(obs.pressure_hpa),
        "STORM_SPEED": float(obs.storm_speed_knots),
        "STORM_DIR": float(obs.storm_direction_deg),
        "MONTH": int(obs.month), "HOUR": int(obs.hour), "OBS_NUM": int(obs.obs_num),
        "WIND_CHANGE": 0.0, "PRES_CHANGE": 0.0, "LAT_CHANGE": 0.0, "LON_CHANGE": 0.0,
    }


def _row_array(row: dict, cols) -> np.ndarray:
    """Select the model's feature columns from the superset row -> (1, n) float64."""
    return np.asarray([[float(row[c]) for c in cols]], dtype=np.float64)


def category_posterior(art, row, feature_cols):
    """Empirical per-tree distribution across IMD categories (honest ensemble posterior)."""
    cats = art["cat_bands"]
    labels = [c[0] for c in cats]
    X = _row_array(row, feature_cols)
    vals = _raw_forest(art["intensity"], X)[0]
    counts = {lab: 0 for lab in labels}
    for w in vals:
        counts[_band(cats, float(w))] += 1
    total = max(1, vals.shape[0])
    return {lab: round(cnt / total, 4) for lab, cnt in counts.items()}


def intensity_forecast(obs) -> Dict:
    art = _load()
    if art is None:
        raise RuntimeError("cyclo_models.joblib not found — run backend/train_ibtracs.py first")
    row = _mk_row(obs)
    cols = art["feature_cols_intensity"]
    X = _row_array(row, cols)
    pred = float(_regress_multi(art["intensity"], X)[0])
    posterior = category_posterior(art, row, cols)
    top_cat, top_prob = max(posterior.items(), key=lambda kv: kv[1])
    return {
        "predicted_wind_knots": round(pred, 1),
        "category": top_cat,
        "category_confidence": round(top_prob, 4),
        "posterior": posterior,
    }


def formation_forecast(obs) -> Dict:
    art = _load()
    row = _mk_row(obs)
    cols = art["feature_cols_formation"]
    probs = art["formation"].predict_proba(_row_array(row, cols))[0]
    prob = float(probs[1])
    verdict = "LIKELY_TO_INTENSIFY" if prob >= 0.5 else "UNLIKELY_TO_FORM"
    return {"probability_becomes_cyclone": round(prob, 4), "verdict": verdict}


def track_forecast(obs) -> Dict:
    art = _load()
    row = _mk_row(obs)
    cols = art["feature_cols_track"]
    na, nb = _regress_multi(art["track"], _row_array(row, cols))[0]
    nlat, nlon = float(na), float(nb)
    dlat = nlat - float(obs.latitude)
    dlon = nlon - float(obs.longitude)
    dist_km = math.hypot(dlat * 111.0, dlon * 111.0 * math.cos(math.radians(float(obs.latitude))))
    heading = (math.degrees(math.atan2(dlon * math.cos(math.radians(float(obs.latitude))), dlat)) + 360) % 360
    return {"latitude": round(nlat, 3), "longitude": round(nlon, 3),
            "distance_km": round(dist_km, 1), "heading_deg": round(heading, 1), "horizon_hours": 6}


def _outlook(obs, steps: int = 12, report_every: int = 4) -> List[Dict]:
    """Rechain track + intensity every 6 h for a 24/48/72 h wind projection."""
    art = _load()
    horizon_hours, tail = 24, []
    current = obs
    start_wind = float(obs.wind_knots)
    for i in range(steps):
        np_ = track_forecast(current)
        r = SimpleNamespace()
        r.latitude, r.longitude = np_["latitude"], np_["longitude"]
        r.wind_knots = current.wind_knots
        r.pressure_hpa = current.pressure_hpa
        r.storm_speed_knots = current.storm_speed_knots
        r.storm_direction_deg = current.storm_direction_deg
        r.month = current.month
        r.hour = (current.hour + 6) % 24
        r.obs_num = current.obs_num + 1
        wind = intensity_forecast(r)["predicted_wind_knots"]
        if (i + 1) % report_every == 0:
            delta = wind - start_wind
            trend = "Intensifying" if delta >= 5 else ("Weakening" if delta <= -5 else "Stable")
            tail.append({
                "horizon_hours": horizon_hours,
                "projected_wind_knots": round(wind, 1),
                "category": _band(art["cat_bands"], wind),
                "trend": trend,
                "confidence_note": "Ensemble projection — deterministic 6 h re-chaining of the track and intensity models; not an IMD forecast.",
                "_lat": np_["latitude"], "_lon": np_["longitude"],
            })
            horizon_hours += 24
        current = r
    return tail


def full_forecast(obs, source_mode: str, storm_id=None, name=None) -> Dict:
    art = _load()
    if art is None:
        raise RuntimeError("cyclo_models.joblib not found — run backend/train_ibtracs.py first")
    inten = intensity_forecast(obs)
    form = formation_forecast(obs)
    nxt = track_forecast(obs)
    outlook = _outlook(obs)

    wind = inten["predicted_wind_knots"]
    category = inten["category"]
    prob = form["probability_becomes_cyclone"]
    tendency = "Stable"
    if outlook:
        end = outlook[-1]["projected_wind_knots"]
        tendency = "Intensifying" if end >= wind + 5 else ("Weakening" if end <= wind - 5 else "Stable")
    risk_level = "HIGH" if prob >= 0.75 else ("MODERATE" if prob >= 0.5 else "LOW")

    radius_by_cat = {"Depression": 80, "Deep Depression": 110, "Cyclonic Storm": 150,
                     "Severe Cyclonic Storm": 200, "Very Severe Cyclonic Storm": 260,
                     "Extremely Severe Cyclonic Storm": 320, "Super Cyclonic Storm": 380}
    radius = radius_by_cat.get(category, 150)

    effects = [
        {"label": "Coastal inundation", "detail": f"Storm surge risk rises near the {wind:.0f} kt wind field.", "severity": "HIGH" if wind >= 64 else "MODERATE"},
        {"label": "Heavy rainfall", "detail": "560–900 mm totals possible within the core band.", "severity": "HIGH" if wind >= 48 else "MODERATE"},
        {"label": "Storm-force winds", "detail": f"{category} gust factors threaten infrastructure within ~{radius} km.", "severity": "EXTREME" if wind >= 90 else "HIGH"},
    ] if prob >= 0.5 else [
        {"label": "Convective weather", "detail": "Moderate rainfall with isolated squalls; organised circulation not expected.", "severity": "LOW"},
        {"label": "Marine advisory", "detail": "Sea state rough along exposed coasts during the monsoon surge.", "severity": "MODERATE"},
    ]

    alert = (
        f"{'HIGH' if prob >= 0.75 else 'ELEVATED'} intensification potential ({prob * 100:.0f}%): model projects {category} "
        f"at ~{wind:.0f} kt sustained, {tendency.lower()} over 72 h. Issue localised coastal warnings and pre-position relief."
        if prob >= 0.5 else
        f"Formation potential low ({prob * 100:.0f}%). Storm remains below cyclonic strength (~{wind:.0f} kt). Keep routine monitoring."
    )

    trail = []
    if storm_id:
        trail.append({"iso_time": f"T-0 ({name or 'observed'})", "latitude": round(float(obs.latitude), 3),
                      "longitude": round(float(obs.longitude), 3), "wind_knots": round(wind, 1),
                      "category": category, "is_forecast": False})
    for pt in outlook:
        trail.append({"iso_time": f"+{pt['horizon_hours']}h", "latitude": pt["_lat"], "longitude": pt["_lon"],
                      "wind_knots": pt["projected_wind_knots"], "category": pt["category"], "is_forecast": True})

    return {
        "intensity": {"predicted_wind_knots": wind, "category": category,
                      "category_confidence": inten["category_confidence"]},
        "formation": form,
        "next_position": nxt,
        "outlook": [{k: v for k, v in pt.items() if not k.startswith("_")} for pt in outlook],
        "storm_id": storm_id,
        "storm_name": name,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "cyclone_detected": prob >= 0.5,
        "detection_confidence": prob,
        "intensity_category": category,
        "intensity_confidence": inten["category_confidence"],
        "wind_speed_knots": wind,
        "rapid_intensification_risk": prob,
        "risk_level": risk_level,
        "latitude": round(float(obs.latitude), 3),
        "longitude": round(float(obs.longitude), 3),
        "temporal_tendency": tendency,
        "alert_message": alert,
        "suggested_action_radius_km": radius,
        "source_mode": source_mode,
        "disclaimer": "Experimental model projections. Verify with official IMD bulletins before any action.",
        "system_type": f"{category} — RandomForest ensemble (intensity/formation/track)",
        "effects": effects,
        "intensity_class_probabilities": inten["posterior"],
        "trail": trail,
    }