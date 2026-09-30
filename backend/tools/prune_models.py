"""
Shrink the persisted RandomForest ensemble for the free cloud tier.

The saved forests carry 300 trees each (intensity, formation and both track
heads) and cost ~219 MB resident once loaded, which is what pushes the
512 MB Render instance into OOM on the image-upload path. The app already does
its own per-tree inference (`_raw_forest` averages over `estimators_`), so
truncating the tree list is a drop-in change.

Usage:  python backend/tools/prune_models.py [n_trees]
Writes models/cyclo_models.joblib (backup kept as *.full.bak) and prints the
prediction drift so the accuracy cost is visible.
"""
import os
import shutil
import sys
from types import SimpleNamespace

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

MODEL_PATH = os.path.join(ROOT, "models", "cyclo_models.joblib")
N_TREES = int(sys.argv[1]) if len(sys.argv) > 1 else 60

SAMPLES = [
    # (lat, lon, wind, pres, speed, dir, month, hour, obs_num)
    (13.7, 86.3, 41.0, 985.0, 15.0, 280.0, 5, 12, 0),
    (19.0, 72.5, 65.0, 960.0, 22.0, 265.0, 6, 6, 0),
    (10.5, 79.5, 28.0, 998.0, 10.0, 300.0, 7, 18, 0),
    (21.5, 68.0, 95.0, 940.0, 28.0, 250.0, 8, 9, 0),
    (8.2, 77.5, 18.0, 998.0, 8.0, 310.0, 9, 3, 0),
    (15.8, 82.0, 120.0, 920.0, 32.0, 240.0, 10, 15, 0),
]


def sample_obs(row):
    lat, lon, wind, pres, speed, direction, month, hour, obs_num = row
    return SimpleNamespace(latitude=lat, longitude=lon, wind_knots=wind, pressure_hpa=pres,
                           storm_speed_knots=speed, storm_direction_deg=direction,
                           month=month, hour=hour, obs_num=obs_num)


def predict_with(bundle):
    """Mirror the app's inference path (per-tree apply + mean)."""
    from backend.app.services import forecast_service as fs

    intensity, formation, track = bundle["intensity"], bundle["formation"], bundle["track"]
    out = []
    for row in SAMPLES:
        obs = sample_obs(row)
        feature_row = fs._mk_row(obs)

        X = fs._row_array(feature_row, bundle["feature_cols_intensity"])
        wind = float(fs._fast_rf_mean(intensity, X)[0])

        Xf = fs._row_array(feature_row, bundle["feature_cols_formation"])
        prob = float(formation.predict_proba(Xf)[0][1])

        Xt = fs._row_array(feature_row, bundle["feature_cols_track"])
        nlat, nlon = [float(v) for v in fs._regress_multi(track, Xt)[0]]

        out.append((wind, prob, nlat, nlon))
    return out


def count_trees(obj):
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.multioutput import MultiOutputRegressor

    if isinstance(obj, (RandomForestRegressor, RandomForestClassifier)):
        return len(obj.estimators_)
    if isinstance(obj, MultiOutputRegressor):
        return sum(len(e.estimators_) for e in obj.estimators_)
    return 0


def prune(obj, n):
    """Truncate every persisted forest to its first n trees."""
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.multioutput import MultiOutputRegressor

    if isinstance(obj, (RandomForestRegressor, RandomForestClassifier)):
        if len(obj.estimators_) > n:
            obj.estimators_ = obj.estimators_[:n]
            obj.n_estimators = n
        return obj
    if isinstance(obj, MultiOutputRegressor):
        for est in obj.estimators_:
            prune(est, n)
        return obj
    return obj


def main():
    print("loading", MODEL_PATH)
    bundle = joblib.load(MODEL_PATH)
    before = predict_with(bundle)
    print("trees before:", {k: count_trees(bundle[k]) for k in ("intensity", "formation", "track")})

    for key in ("intensity", "formation", "track"):
        prune(bundle[key], N_TREES)
    after = predict_with(bundle)
    print("trees after :", {k: count_trees(bundle[k]) for k in ("intensity", "formation", "track")})

    print("\n%-34s %10s %10s %8s" % ("sample (lat,lon,wind,pres)", "wind_kt", "form_prob", "d_wind"))
    max_wind = max_form = 0.0
    for row, (wb, fb, _, _), (wa, fa, _, _) in zip(SAMPLES, before, after):
        print("%-34s %6.1f ->%6.1f  %5.3f->%5.3f  %+7.2f" % (
            ",".join(str(v) for v in row[:4]), wb, wa, fb, fa, wa - wb))
        max_wind = max(max_wind, abs(wa - wb))
        max_form = max(max_form, abs(fa - fb))
    print("\nmax |d wind| = %.2f kt   max |d formation prob| = %.4f" % (max_wind, max_form))

    backup = MODEL_PATH + ".full.bak"
    if not os.path.exists(backup):
        shutil.copy2(MODEL_PATH, backup)
        print("backup ->", backup)
    joblib.dump(bundle, MODEL_PATH, compress=9)
    print("wrote %s (%.1f MB)" % (MODEL_PATH, os.path.getsize(MODEL_PATH) / 1e6))


if __name__ == "__main__":
    main()
