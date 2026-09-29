"""
Shared IBTrACS cleaning / feature engineering / storm-cache helpers.

Keeps the Colab pipeline and the API-side cache builder on identical logic.
"""

import os

import numpy as np
import pandas as pd


def clean_ibtracs(csv_path: str) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, skiprows=[1], low_memory=False)
    cols = ["SID", "SEASON", "BASIN", "NAME", "ISO_TIME", "NATURE",
            "LAT", "LON", "NEWDELHI_WIND", "NEWDELHI_PRES", "STORM_SPEED", "STORM_DIR"]
    df = raw[cols].copy()
    df = df[df["BASIN"] == "NI"].copy()
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"])
    for c in ["LAT", "LON", "NEWDELHI_WIND", "NEWDELHI_PRES", "STORM_SPEED", "STORM_DIR"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.rename(columns={"NEWDELHI_WIND": "WIND", "NEWDELHI_PRES": "PRES"})
    df = df.dropna(subset=["WIND"])
    df = df.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)
    df["PRES"] = df.groupby("SID")["PRES"].transform(lambda x: x.interpolate(limit_direction="both"))
    df = df.dropna(subset=["STORM_SPEED", "STORM_DIR"])
    df["PRES"] = df["PRES"].fillna(df.groupby(pd.cut(df["WIND"], bins=10))["PRES"].transform("median"))
    df["PRES"] = df["PRES"].fillna(df["PRES"].median())
    return df


def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MONTH"] = df["ISO_TIME"].dt.month
    df["HOUR"] = df["ISO_TIME"].dt.hour
    df["OBS_NUM"] = df.groupby("SID").cumcount()
    df["WIND_CHANGE"] = df.groupby("SID")["WIND"].diff().fillna(0)
    df["PRES_CHANGE"] = df.groupby("SID")["PRES"].diff().fillna(0)
    df["LAT_CHANGE"] = df.groupby("SID")["LAT"].diff().fillna(0)
    df["LON_CHANGE"] = df.groupby("SID")["LON"].diff().fillna(0)
    return df


def build_storm_cache(df: pd.DataFrame, names: list) -> dict:
    """Per-storm demo entries, truncated at peak wind (front end 'current' = peak)."""

    def _wrangle(g: pd.DataFrame) -> dict:
        g = g.sort_values("ISO_TIME")
        g = add_sequence_features(g)
        peak_idx = int(g["WIND"].idxmax())
        g = g.loc[:peak_idx]
        src_cols = ["LAT", "LON", "WIND", "PRES", "STORM_SPEED", "STORM_DIR",
                    "MONTH", "HOUR", "OBS_NUM", "WIND_CHANGE", "PRES_CHANGE",
                    "LAT_CHANGE", "LON_CHANGE"]
        last = g.iloc[-1]
        return {
            "storm_id": g["SID"].iloc[-1],
            "name": str(g["NAME"].iloc[-1]).strip(),
            "last_row": {c: (float(last[c]) if pd.notna(last[c]) else None) for c in src_cols},
            "track": [{"iso_time": r.ISO_TIME.strftime("%Y-%m-%d %H:%M:%S"),
                       "latitude": float(r.LAT), "longitude": float(r.LON),
                       "wind_knots": float(r.WIND),
                       "pressure_hpa": None if pd.isna(r.PRES) else float(r.PRES)}
                      for r in g.itertuples()],
        }

    by_name = {nm.strip().upper(): g for nm, g in df.groupby(df["NAME"].str.strip().str.upper())}
    cache = {}
    for frag in names:
        hit = [nm for nm in by_name if nm.startswith(frag)]
        if hit:
            name = sorted(hit)[0]
            storm = _wrangle(by_name[name])
            cache[f"demo_cyclone_{name.lower().replace(' ', '_')}"] = storm
    weak = df[df["SID"].map(dict(df.groupby("SID")["WIND"].max())).lt(34)]
    if len(weak):
        sid = weak["SID"].mode().iloc[0]
        cache["demo_non_cyclone"] = _wrangle(df[df["SID"] == sid])
    return cache


CAT_BANDS = [("Depression", 17), ("Deep Depression", 28), ("Cyclonic Storm", 34),
             ("Severe Cyclonic Storm", 48), ("Very Severe Cyclonic Storm", 64),
             ("Extremely Severe Cyclonic Storm", 90), ("Super Cyclonic Storm", 120)]


def imd_category(wind) -> str:
    label = CAT_BANDS[0][0]
    for name, lo in CAT_BANDS:
        if wind >= lo:
            label = name
    return label


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))