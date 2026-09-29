"""
CycloVision Decision-Support Insight Engine

Deterministic expert-system content derived from REAL model outputs
(detection flag, intensity class posterior, rapid-intensification risk).

Produces three report blocks:
  1. Formal IMD system type (plain-language classification)
  2. Impact / effects profile keyed to the detected intensity class
  3. 24 / 48 / 72 hour intensity outlook projected from the RI risk

This engine NEVER invents model results — it interprets the model's own
predictions using IMD North Indian Ocean domain rules, and every outlook
point carries an explicit confidence note.
"""

from typing import Dict, List, Any, Optional

IMD_CATEGORIES = [
    "Depression",
    "Deep Depression",
    "Cyclonic Storm",
    "Severe Cyclonic Storm",
    "Very Severe Cyclonic Storm",
    "Extremely Severe Cyclonic Storm",
    "Super Cyclonic Storm",
]

# Wind thresholds (knots) used for category projection from forecast wind.
CATEGORY_WIND_KT = [
    (17, "Depression"),
    (28, "Deep Depression"),
    (34, "Cyclonic Storm"),
    (48, "Severe Cyclonic Storm"),
    (64, "Very Severe Cyclonic Storm"),
    (90, "Extremely Severe Cyclonic Storm"),
    (120, "Super Cyclonic Storm"),
]

CATEGORY_PROFILE: Dict[str, Dict[str, Any]] = {
    "Depression": {
        "full_type": "Depression (D)",
        "wind_kt": "17-27 kt",
        "wind_kmh": "31-49 km/h",
        "pressure": "~998 hPa",
        "surge": "0.5-0.7 m above astronomical tide",
        "rainfall": "8-12 cm/day, intermittent",
        "severity": "LOW",
        "effects": [
            {"label": "Winds", "detail": "Sustained 17-27 kt (31-49 km/h) with gusty squalls over sea areas.", "severity": "LOW"},
            {"label": "Sea state", "detail": "Rough to very rough seas; some swell in the coastal belt.", "severity": "LOW"},
            {"label": "Rainfall", "detail": "Light to moderate rain, isolated districts above 8 cm.", "severity": "LOW"},
            {"label": "Maritime", "detail": "Trawlers and small craft advised to remain close to shore.", "severity": "LOW"},
        ],
    },
    "Deep Depression": {
        "full_type": "Deep Depression (DD)",
        "wind_kt": "28-33 kt",
        "wind_kmh": "50-61 km/h",
        "pressure": "~990 hPa",
        "surge": "0.7-1.2 m above astronomical tide",
        "rainfall": "12-20 cm/day",
        "severity": "MODERATE",
        "effects": [
            {"label": "Winds", "detail": "Sustained 28-33 kt (50-61 km/h); strong gusts along coast.", "severity": "MODERATE"},
            {"label": "Sea state", "detail": "High seas; waves 3-4 m, hazardous for small fishermen.", "severity": "MODERATE"},
            {"label": "Rainfall", "detail": "Heavy rain in isolated coastal districts; localised waterlogging.", "severity": "MODERATE"},
            {"label": "Maritime", "detail": "Fishermen advised not to venture into the deep sea.", "severity": "MODERATE"},
        ],
    },
    "Cyclonic Storm": {
        "full_type": "Cyclonic Storm (CS)",
        "wind_kt": "34-47 kt",
        "wind_kmh": "62-88 km/h",
        "pressure": "~982 hPa",
        "surge": "1.2-2.0 m above astronomical tide",
        "rainfall": "20-35 cm/day",
        "severity": "MODERATE",
        "effects": [
            {"label": "Winds", "detail": "Sustained 34-47 kt (62-88 km/h).", "severity": "MODERATE"},
            {"label": "Storm surge", "detail": "1.2-2.0 m surge may inundate low-lying coastal pockets.", "severity": "MODERATE"},
            {"label": "Rainfall", "detail": "Heavy 20-35 cm/day; flash flooding in urban drains and rivulets.", "severity": "HIGH"},
            {"label": "Structural", "detail": "Minor damage to huts, trees and power poles likely.", "severity": "MODERATE"},
            {"label": "Maritime", "detail": "Ports hoist Distant Cautionary Signal; coastal shipping suspended.", "severity": "MODERATE"},
        ],
    },
    "Severe Cyclonic Storm": {
        "full_type": "Severe Cyclonic Storm (SCS)",
        "wind_kt": "48-63 kt",
        "wind_kmh": "89-118 km/h",
        "pressure": "~972 hPa",
        "surge": "2.0-3.5 m above astronomical tide",
        "rainfall": "25-45 cm/day",
        "severity": "HIGH",
        "effects": [
            {"label": "Winds", "detail": "Sustained 48-63 kt (89-118 km/h); damaging gusts.", "severity": "HIGH"},
            {"label": "Storm surge", "detail": "2.0-3.5 m surge; inundation of coastal districts up to 5 km inland.", "severity": "HIGH"},
            {"label": "Rainfall", "detail": "Very heavy 25-45 cm/day; rivers may flow above danger level.", "severity": "HIGH"},
            {"label": "Structural", "detail": "Partial damage to kutcha houses, uprooting of trees, disruption of power.", "severity": "HIGH"},
            {"label": "Maritime", "detail": "Ports hoist Signal No. 8; total suspension of coastal traffic.", "severity": "HIGH"},
        ],
    },
    "Very Severe Cyclonic Storm": {
        "full_type": "Very Severe Cyclonic Storm (VSCS)",
        "wind_kt": "64-89 kt",
        "wind_kmh": "119-166 km/h",
        "pressure": "~960 hPa",
        "surge": "3.0-5.0 m above astronomical tide",
        "rainfall": "30-60 cm/day",
        "severity": "HIGH",
        "effects": [
            {"label": "Winds", "detail": "Sustained 64-89 kt (119-166 km/h); destructive gusts up to 100 kt.", "severity": "HIGH"},
            {"label": "Storm surge", "detail": "3.0-5.0 m surge; extensive flooding of coastal plains and harbours.", "severity": "EXTREME"},
            {"label": "Rainfall", "detail": "Extremely heavy 30-60 cm/day; major riverine flooding risk.", "severity": "HIGH"},
            {"label": "Structural", "detail": "Major damage — roofs, mobile homes, standing crops flattened; communication outage.", "severity": "EXTREME"},
            {"label": "Maritime", "detail": "Great danger signal; evacuation along vulnerable coastal stretches.", "severity": "HIGH"},
        ],
    },
    "Extremely Severe Cyclonic Storm": {
        "full_type": "Extremely Severe Cyclonic Storm (ESCS)",
        "wind_kt": "90-119 kt",
        "wind_kmh": "167-221 km/h",
        "pressure": "~945 hPa",
        "surge": "5.0-7.5 m above astronomical tide",
        "rainfall": "40-80 cm/day",
        "severity": "EXTREME",
        "effects": [
            {"label": "Winds", "detail": "Sustained 90-119 kt (167-221 km/h) — violent storm force.", "severity": "EXTREME"},
            {"label": "Storm surge", "detail": "5.0-7.5 m surge with storm tide penetration far inland.", "severity": "EXTREME"},
            {"label": "Rainfall", "detail": "Catastrophic 40-80 cm/day rainfall; extensive flash floods.", "severity": "EXTREME"},
            {"label": "Structural", "detail": "Widespread devastation of structures, power grid and transport networks.", "severity": "EXTREME"},
            {"label": "Maritime", "detail": "All port operations suspended; mass coastal evacuation triggered.", "severity": "EXTREME"},
        ],
    },
    "Super Cyclonic Storm": {
        "full_type": "Super Cyclonic Storm (SuCS)",
        "wind_kt": ">= 120 kt",
        "wind_kmh": ">= 222 km/h",
        "pressure": "< 930 hPa",
        "surge": "7.5-10+ m above astronomical tide",
        "rainfall": "> 60 cm/day",
        "severity": "EXTREME",
        "effects": [
            {"label": "Winds", "detail": "Sustained >= 120 kt (>= 222 km/h) — catastrophic, long-lived storm force.", "severity": "EXTREME"},
            {"label": "Storm surge", "detail": "7.5-10+ m devastating storm tide over coastal plains.", "severity": "EXTREME"},
            {"label": "Rainfall", "detail": "Extreme rainfall beyond 60 cm/day; apocalyptic flash flooding.", "severity": "EXTREME"},
            {"label": "Structural", "detail": "Total destruction of housing stock, lifelines and ports over wide area.", "severity": "EXTREME"},
            {"label": "Maritime", "detail": "Highest alert; mass evacuation, war-room coordination by NDMA/SDMA.", "severity": "EXTREME"},
        ],
    },
}


def _category_for_wind(knots: float) -> str:
    cat = "Depression"
    for threshold, name in CATEGORY_WIND_KT:
        if knots >= threshold:
            cat = name
    return cat


def system_type(category: str) -> str:
    """Formal IMD classification string for display."""
    profile = CATEGORY_PROFILE.get(category)
    if profile is None:
        return category
    return profile["full_type"]


def build_effects(category: str) -> List[Dict[str, str]]:
    """Impact / effects profile keyed to the intensity class."""
    profile = CATEGORY_PROFILE.get(category)
    if profile is None:
        return []
    return list(profile["effects"])


def build_outlook(
    wind_knots: float,
    category: str,
    ri_risk: float,
    detected: bool = True,
) -> List[Dict[str, Any]]:
    """
    Deterministic 24/48/72h intensity outlook projected from the RI risk.
    Delta steps are scaled by the RI band; every point is surface-level
    interpretation (intensifying / stable / weakening) with a confidence note.
    """
    if ri_risk >= 0.65:
        deltas = [8, 16, 24]
        base_note = "RI model agreement >= 65% — rapid intensification regime."
    elif ri_risk >= 0.35:
        deltas = [3, 5, 8]
        base_note = "RI risk moderate — gradual intensification expected."
    else:
        deltas = [-4, -6, -8]
        base_note = "RI risk low — gradual weakening/transition anticipated."

    if not detected:
        deltas = [d * 0.5 for d in deltas]
        base_note = "Detection confidence low — outlook applies only if system consolidates."

    outlook: List[Dict[str, Any]] = []
    for horizon, delta in zip((24, 48, 72), deltas):
        projected = min(165.0, max(18.0, round(float(wind_knots) + delta, 1)))
        trend = "Intensifying" if delta > 0 else ("Stable" if delta == 0 else "Weakening")
        outlook.append({
            "horizon_hours": horizon,
            "projected_wind_knots": projected,
            "category": _category_for_wind(projected),
            "trend": trend,
            "confidence_note": base_note,
        })
    return outlook


def build_report(
    wv_source: Optional[str],
    cyclone_detected: bool,
    detection_confidence: float,
    intensity_category: str,
    intensity_confidence: float,
    wind_knots: float,
    ri_risk: float,
    risk_level: str,
    temporal_tendency: str,
) -> Dict[str, Any]:
    """
    Assemble the full decision-support report block from verified model outputs.
    """
    return {
        "system_type": system_type(intensity_category),
        "effects": build_effects(intensity_category),
        "outlook": build_outlook(
            wind_knots=float(wind_knots),
            category=intensity_category,
            ri_risk=float(ri_risk),
            detected=bool(cyclone_detected),
        ),
    }