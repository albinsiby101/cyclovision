"""
Automated Pytest Suite for the IBTrACS RandomForest CycloVision backend.
Tests health, storms catalog, demo forecasts, and the observation forecast endpoint.
"""

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)

AMPLE_OBS = {
    "name": "AMPHAN",
    "latitude": 13.7,
    "longitude": 86.3,
    "wind_knots": 130.0,
    "pressure_hpa": 920.0,
    "storm_speed_knots": 12.0,
    "storm_direction_deg": 290.0,
    "month": 5,
    "hour": 12,
    "obs_num": 0,
}


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["application"] == "CycloVision"
    assert "models_loaded" in data
    assert data["mode"] in ("operational", "demo")


def test_storms_catalog():
    res = client.get("/api/storms")
    assert res.status_code == 200
    storms = res.json()
    assert len(storms) >= 2
    assert any(s["name"] == "AMPHAN" for s in storms)
    assert any(s["name"] == "BIPARJOY" for s in storms)
    for s in storms:
        assert s["track"], f"storm {s['name']} has an empty track"
        assert all(pt["is_forecast"] is False for pt in s["track"][:-1])
        for pt in s["track"]:
            assert "latitude" in pt and "longitude" in pt and "wind_knots" in pt


def test_demo_forecast_amphan():
    res = client.get("/api/demo/storm/demo_cyclone_amphan")
    assert res.status_code == 200
    data = res.json()
    assert data["cyclone_detected"] is True
    assert 0 < data["detection_confidence"] <= 1
    assert data["intensity_category"]
    assert data["intensity"]["predicted_wind_knots"] > 0
    assert 0 <= data["rapid_intensification_risk"] <= 1
    assert data["risk_level"] in ["LOW", "MODERATE", "HIGH"]
    assert len(data["effects"]) >= 3
    assert len(data["outlook"]) == 3
    assert data["outlook"][0]["horizon_hours"] == 24
    assert data["next_position"]["horizon_hours"] == 6


def test_demo_forecast_biparjoy_key_normalisation():
    res = client.get("/api/demo/storm/demo_biparjoy")
    assert res.status_code == 200
    data = res.json()
    assert data["storm_id"]
    assert data["intensity_category"]


def test_forecast_observation():
    res = client.post("/api/forecast", json=AMPLE_OBS)
    assert res.status_code == 200
    data = res.json()
    assert data["source_mode"] == "MODEL_FORECAST"
    assert data["storm_name"] == "AMPHAN"
    assert data["intensity"]["predicted_wind_knots"] > 0
    assert data["formation"]["verdict"] in ("LIKELY_TO_INTENSIFY", "LIKELY_TO_FORM", "UNLIKELY")
    assert data["next_position"]["distance_km"] > 0
    assert len(data["outlook"]) == 3
    assert data["intensity_class_probabilities"]
    assert sum(data["intensity_class_probabilities"].values()) > 0.99
    for pt in data["outlook"]:
        assert pt["trend"] in ("Intensifying", "Stable", "Weakening")
    assert len(data["trail"]) == 3
    assert all(pt["is_forecast"] for pt in data["trail"])


def test_forecast_weak_observation_non_cyclone():
    weak = dict(AMPLE_OBS)
    weak.update({"name": "BASELINE", "wind_knots": 15.0, "pressure_hpa": 1005.0, "obs_num": 0})
    res = client.post("/api/forecast", json=weak)
    assert res.status_code == 200
    data = res.json()
    assert data["intensity"]["predicted_wind_knots"] < 34


def test_forecast_validation_bad_pressure():
    bad = dict(AMPLE_OBS)
    bad["pressure_hpa"] = 9999
    res = client.post("/api/forecast", json=bad)
    assert res.status_code == 422


def test_forecast_missing_fields():
    res = client.post("/api/forecast", json={"name": "NO_VALUES"})
    assert res.status_code == 422


_DEMO_DIR = "data/demo/satellite_sequences/demo_cyclone_amphan"


def test_analyse_satellite_image():
    import os

    ir_path = os.path.join(_DEMO_DIR, "frame_0_ir.png")
    wv_path = os.path.join(_DEMO_DIR, "frame_0_wv.png")
    if not os.path.exists(ir_path):
        return  # demo fixtures absent -> skip gracefully
    with open(ir_path, "rb") as ir:
        with open(wv_path, "rb") as wv:
            res = client.post(
                "/api/analyse",
                files={"ir_image": ("ir.png", ir, "image/png"),
                       "wv_image": ("wv.png", wv, "image/png")},
                data={"name": "AMPHAN_IMG", "latitude": "13.7", "longitude": "86.3",
                      "storm_speed_knots": "12", "storm_direction_deg": "290"},
            )
    assert res.status_code == 200
    data = res.json()
    assert data["source_mode"] == "UPLOAD_ANALYSIS"
    assert data["storm_name"] == "AMPHAN_IMG"
    assert "detection" in data and "cyclone_detected" in data["detection"]
    assert 0 <= data["detection_confidence"] <= 1
    assert data["intensity_category"]
    assert data["wind_speed_knots"] > 0
    assert "intensity_cnn" in data
    assert "ri" in data
    assert "gradcam" in data and "heatmap_available" in data["gradcam"]
    # ONNX Runtime cloud build does not ship PyTorch/Grad-CAM: heatmap must be
    # cleanly flagged unavailable (prediction verdict itself is unchanged).
    assert data["gradcam"]["heatmap_available"] is False
    assert data["gradcam"]["overlay_image_base64"] is None
    assert data["wv_source"] == "UPLOADED"
    assert 0 <= data["rapid_intensification_risk"] <= 1
    assert len(data["outlook"]) == 3
    assert data["next_position"]["distance_km"] > 0
    # CNN headline fields override RF values
    assert data["intensity"]["category"] == data["intensity_category"]
    assert data["intensity_class_probabilities"] == data["intensity_cnn"]["class_probabilities"]


def test_analyse_satellite_ir_only_uses_proxy():
    import os

    ir_path = os.path.join(_DEMO_DIR, "frame_0_ir.png")
    if not os.path.exists(ir_path):
        return
    with open(ir_path, "rb") as ir:
        res = client.post(
            "/api/analyse",
            files={"ir_image": ("ir.png", ir, "image/png")},
            data={"name": "IR_ONLY"},
        )
    assert res.status_code == 200
    data = res.json()
    assert data["source_mode"] == "UPLOAD_ANALYSIS"
    assert data["wv_source"] == "PROXY_FROM_IR"


def test_analyse_rejects_empty_image():
    res = client.post("/api/analyse", files={"ir_image": ("ir.png", b"", "image/png")})
    assert res.status_code == 400


def test_analyse_gate_rejects_non_satellite_noise():
    import io

    import cv2
    import numpy as np

    checker = np.kron(np.tile(np.eye(8), (4, 4)) * 255, np.ones((8, 8), np.uint8)).astype(np.uint8)
    ok, buf = cv2.imencode(".png", checker)
    assert ok
    res = client.post(
        "/api/analyse",
        files={"ir_image": ("noise.png", io.BytesIO(buf.tobytes()), "image/png")},
        data={"name": "RANDOM"},
    )
    assert res.status_code == 200
    assert res.json()["input_verified"] is False
    assert res.json()["cyclone_detected"] is None
    assert res.json()["intensity_category"] == "Inconclusive"
    assert res.json()["wind_speed_knots"] is None
    assert res.json()["risk_level"] == "INCONCLUSIVE"
    assert res.json()["formation"]["verdict"] == "INCONCLUSIVE"
    assert res.json()["outlook"] == []