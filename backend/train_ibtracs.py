# Mirror of the Colab notebook pipeline (clean -> features -> three RandomForests).
# Saves the exact artifact the API loads; the Colab-produced cyclo_models.joblib
# is drop-in interchangeable. Optional: python backend/train_ibtracs.py
# (already-ran models are reused; bikeshed-free numpy fits avoid sklearn name warnings).

import json
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

from backend import ibtracs_common

ROOT = ibtracs_common.project_root()
DATA = os.path.join(ROOT, "data", "ibtracs.csv")
MODEL_PATH = os.path.join(ROOT, "models", "cyclo_models.joblib")
CACHE_PATH = os.path.join(ROOT, "models", "ibtracs_ni_storms.json")


def _fit_intensity(df, feature_cols, target="WIND", seed_storms=42):
    unique_storms = df["SID"].unique()
    s_tr, s_te = train_test_split(unique_storms, test_size=0.2, random_state=seed_storms)
    tr, te = df[df["SID"].isin(s_tr)], df[df["SID"].isin(s_te)]
    model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(tr[feature_cols].to_numpy(), tr[target].to_numpy())
    preds = model.predict(te[feature_cols].to_numpy())
    return model, mean_absolute_error(te[target], preds), r2_score(te[target], preds)


def _fit_formation(early, feature_cols, target):
    s_tr, s_te = train_test_split(early["SID"].unique(), test_size=0.2, random_state=42)
    tr, te = early[early["SID"].isin(s_tr)], early[early["SID"].isin(s_te)]
    model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1, class_weight="balanced")
    model.fit(tr[feature_cols].to_numpy(), tr[target].to_numpy())
    preds = model.predict(te[feature_cols].to_numpy())
    return model, accuracy_score(te[target], preds), classification_report(te[target], preds, zero_division=0)


def _fit_track(track_df, feature_cols, targets):
    s_tr, s_te = train_test_split(track_df["SID"].unique(), test_size=0.2, random_state=42)
    tr, te = track_df[track_df["SID"].isin(s_tr)], track_df[track_df["SID"].isin(s_te)]
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1))
    model.fit(tr[feature_cols].to_numpy(), tr[targets].to_numpy())
    p = model.predict(te[feature_cols].to_numpy())
    lat_mae = mean_absolute_error(te["NEXT_LAT"], p[:, 0])
    lon_mae = mean_absolute_error(te["NEXT_LON"], p[:, 1])
    mean_lat = np.deg2rad(te["LAT"].mean())
    return model, lat_mae, lon_mae, lat_mae * 111.0, lon_mae * 111.0 * np.cos(mean_lat)


def main():
    df = ibtracs_common.clean_ibtracs(DATA)
    df = ibtracs_common.add_sequence_features(df)
    print(f"N rows after cleaning: {df.shape[0]} | storms: {df['SID'].nunique()}")

    f_intensity = ["LAT", "LON", "PRES", "STORM_SPEED", "STORM_DIR",
                   "MONTH", "HOUR", "OBS_NUM", "PRES_CHANGE", "LAT_CHANGE", "LON_CHANGE"]
    model_i, mae_i, r2_i = _fit_intensity(df, f_intensity)
    print(f"Intensity  MAE: {mae_i:.2f} kt | R2: {r2_i:.3f}")

    storm_max = df["SID"].map(dict(df.groupby("SID")["WIND"].max()))
    df["BECOMES_CYCLONE"] = storm_max.ge(34).astype(int)
    early = df[df["OBS_NUM"] <= 2].copy()
    f_formation = ["LAT", "LON", "WIND", "PRES", "STORM_SPEED", "STORM_DIR", "MONTH", "HOUR"]
    model_f, acc_f, rep_f = _fit_formation(early, f_formation, "BECOMES_CYCLONE")
    first_report_line = rep_f.strip().split("\n")[-1]
    print(f"Formation acc: {acc_f:.3f} | {first_report_line}")

    df["NEXT_LAT"] = df.groupby("SID")["LAT"].shift(-1)
    df["NEXT_LON"] = df.groupby("SID")["LON"].shift(-1)
    track_df = df.dropna(subset=["NEXT_LAT", "NEXT_LON"]).copy()
    f_track = ["LAT", "LON", "WIND", "PRES", "STORM_SPEED", "STORM_DIR",
               "MONTH", "OBS_NUM", "LAT_CHANGE", "LON_CHANGE", "WIND_CHANGE", "PRES_CHANGE"]
    model_t, lmae, dmae, lkm, dkm = _fit_track(track_df, f_track, ["NEXT_LAT", "NEXT_LON"])
    print(f"Track   MAE lat {lmae:.2f} deg / lon {dmae:.2f} deg  ({lkm:.1f} / {dkm:.1f} km)")

    art = {
        "intensity": model_i, "feature_cols_intensity": f_intensity,
        "formation": model_f, "feature_cols_formation": f_formation,
        "track": model_t, "feature_cols_track": f_track,
        "pres_wind_corr": float(df[["WIND", "PRES"]].corr().iloc[0, 1]),
        "cat_bands": ibtracs_common.CAT_BANDS,
        "trained_rows": int(df.shape[0]), "trained_storms": int(df["SID"].nunique()),
    }
    joblib.dump(art, MODEL_PATH)
    print("saved", MODEL_PATH)

    cache = ibtracs_common.build_storm_cache(
        df, ["AMPHAN", "BIPAR", "FANI", "TAUKT", "YAAS", "TITLI", "PHAILIN"])
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)
    print("saved", CACHE_PATH, "->", list(cache.keys()))


if __name__ == "__main__":
    main()