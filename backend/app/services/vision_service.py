"""
CycloVision Hybrid Vision Service

Runs the CNN vision stack (ResNet-18 detection, CNN+ConvLSTM intensity, RI risk)
on uploaded satellite imagery, derives a *tabular storm observation* from the
imagery (category -> wind/pressure via IMD bands), and pushes that observation
through the IBTrACS RandomForest forecast chain so the response carries both the
CNN-derived current state and the RF track/formation/outlook.

The CNNs run through onnxruntime (exported from the PyTorch checkpoints by
backend/tools/export_onnx.py). PyTorch is NOT a cloud runtime dependency - the
lightweight build keeps the free-tier memory footprint small. Grad-CAM is a
PyTorch-only nicety and is intentionally omitted in this build (the response
flags heatmap_available=False; detection/intensity/RI verdicts are unchanged).
"""

import os
from typing import Dict, Optional

import cv2
import numpy as np

from ml.preprocessing.preprocessor import SatellitePreprocessor
from backend.app.schemas.cyclone import ObservationInput
from backend.app.services import forecast_service

# IMD category -> (wind band [min, max) in knots, representative pressure hPa)
# Representative pressure is the midpoint of the band used during IBTrACS training.
CATEGORY_BANDS_KNOTS = {
    "Depression": (17, 28),
    "Deep Depression": (28, 34),
    "Cyclonic Storm": (34, 48),
    "Severe Cyclonic Storm": (48, 64),
    "Very Severe Cyclonic Storm": (64, 90),
    "Extremely Severe Cyclonic Storm": (90, 120),
    "Super Cyclonic Storm": (120, 200),
}
CATEGORY_PRESSURE_HPA = {
    "Depression": 1002.0,
    "Deep Depression": 998.0,
    "Cyclonic Storm": 990.0,
    "Severe Cyclonic Storm": 978.0,
    "Very Severe Cyclonic Storm": 962.0,
    "Extremely Severe Cyclonic Storm": 942.0,
    "Super Cyclonic Storm": 920.0,
}
_WIND_RANGES_KMH = ["31-49", "50-61", "62-88", "89-117", "118-166", "167-221", ">= 222"]


class VisionChain:
    def __init__(self):
        self.preprocessor = None
        self.detector = None
        self.intensity_model = None
        self.ri_model = None
        self.models_loaded = False
        self._load()

    def _load(self):
        try:
            import onnxruntime as ort
            det_path = "models/onnx/detector.onnx"
            int_path = "models/onnx/intensity.onnx"
            ri_path = "models/onnx/ri.onnx"
            opts = ort.SessionOptions()
            # Memory-conscious settings for the 512 MB cloud tier: the CPU arena
            # and the mem-pattern planner hold on to buffers that a handful of
            # 128x128 inferences never need, and extra op threads multiply the
            # per-thread arenas.
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.enable_cpu_mem_arena = False
            opts.enable_mem_pattern = False
            self.detector = self._session(det_path, opts) if os.path.exists(det_path) else None
            self.intensity_model = self._session(int_path, opts) if os.path.exists(int_path) else None
            self.ri_model = self._session(ri_path, opts) if os.path.exists(ri_path) else None
            self.preprocessor = SatellitePreprocessor(image_size=(128, 128), sequence_length=6)
            self.models_loaded = (self.detector is not None
                                  and self.intensity_model is not None
                                  and self.ri_model is not None)
        except Exception as e:  # noqa: BLE001 - degraded fallback is intentional
            print(f"[WARN] Vision chain load failed ({e})")
            self.models_loaded = False

    @staticmethod
    def _session(path: str, opts) -> object:
        import onnxruntime as ort
        return ort.InferenceSession(path, sess_options=opts,
                                    providers=["CPUExecutionProvider"])

    def _as_grayscale(self, img: np.ndarray) -> np.ndarray:
        img = np.asarray(img)
        if img is None or img.size == 0:
            raise ValueError("Empty image array received.")
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = img[:, :, :3]
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                img = img[:, :, 0]
        if img.dtype == np.uint16 and img.max() > 255:
            img = (img.astype(np.float32) / 256.0)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        return img.astype(np.float32)

    def _colorfulness(self, rgb: np.ndarray) -> float:
        """Per-pixel mean channel spread. True thermal IR (gray/false-color ramp)
        scores near zero; arbitrary photographs score tens-to-hundreds."""
        if rgb is None or rgb.ndim != 3 or rgb.shape[2] < 3:
            return 0.0
        f = np.asarray(rgb)[..., :3].astype(np.float32)
        spread = (np.abs(f[..., 0] - f[..., 1]) +
                  np.abs(f[..., 1] - f[..., 2]) +
                  np.abs(f[..., 0] - f[..., 2])) / 3.0
        return float(spread.mean())

    def _palette_probe(self, rgb: np.ndarray) -> Dict:
        """For COLORED inputs: measure the saturated-hue diversity.

        Smooth false-color IR palettes are a near-1D hue ramp (few hue clusters,
        large hue spread). Natural photographs contain many scattered hue clusters.
        Returns {'ok': bool, 'hue_bins': int, 'hue_spread': float}."""
        if rgb is None or rgb.ndim != 3 or rgb.shape[2] < 3:
            return {"ok": True, "hue_bins": 0, "hue_spread": 0.0}
        rgb8 = np.asarray(rgb)[..., :3].astype(np.uint8)
        if max(rgb8.shape[:2]) > 96:  # cheap downscale - hue structure is scale-invariant
            rgb8 = cv2.resize(rgb8, (min(rgb8.shape[1], 96), min(rgb8.shape[0], 96)),
                              interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(rgb8, cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = hsv[..., 0], hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
        sel = (s > 0.18) & (v > 0.18)
        if sel.sum() < 4:
            return {"ok": True, "hue_bins": 0, "hue_spread": 0.0}  # effectively gray
        hist, _ = np.histogram(h[sel], bins=180, range=(0, 180))
        hist = hist / hist.sum()
        bins = int((hist > 0.01).sum())
        spread = float(np.abs(h[sel] - h[sel].mean()).mean())
        ok = (bins <= 12) and (spread >= 12.0)  # smooth color-ramp, not a photograph
        return {"ok": ok, "hue_bins": bins, "hue_spread": round(spread, 1)}

    def _plausibility(self, ir_norm: np.ndarray, rgb: Optional[np.ndarray] = None) -> Dict:
        """Reject inputs that do not resemble INSAT-3D thermal satellite imagery.

        Real IR frames (verified against the demo/training distribution) have a dark
        ocean background punctuated by a smaller bright cloudy region and smooth,
        low-texture gradients. The thresholds are deliberately permissive on texture
        (compression noise, coastline edges and enhancement palettes legitimately push
        real imagery to ~0.08-0.12) while still catching:
          - pure / heavily textured noise          (texture > 0.12)
          - extreme periodic patterns              (checker / bar charts: mid-frac ~ 0)
          - solid or near-constant images          (variance ~ 0)
          - fully saturated bright scenes          (no dark ocean backdrop)
        """
        if np.isnan(ir_norm).any() or np.isinf(ir_norm).any():
            return {"plausible": False, "score": 0.0,
                    "note": "Input contains NaN/Inf values — not a valid image."}

        gx = np.abs(np.diff(ir_norm, axis=1))
        gy = np.abs(np.diff(ir_norm, axis=0))
        texture = float((gx.mean() + gy.mean()) / 2.0)
        bright = float((ir_norm > 0.85).mean())
        dark = float((ir_norm < 0.25).mean())
        mid = float(((ir_norm >= 0.25) & (ir_norm <= 0.85)).mean())
        std = float(ir_norm.std())

        ok = True
        reasons = []
        if texture > 0.12:
            ok, reasons = False, reasons + [f"high-frequency texture ({texture:.3f})"]
        if mid < 0.05:
            ok, reasons = False, reasons + [f"no gradual cloud-shade gradient (mid-range {mid:.2f})"]
        if bright > 0.25:
            ok, reasons = False, reasons + [f"excessive bright pixels ({(bright * 100):.0f}%) — no dark ocean backdrop"]
        if std < 0.02:
            ok, reasons = False, reasons + ["near-constant image (zero variance)"]
        colorfulness = self._colorfulness(rgb)
        palette = None
        if colorfulness > 35.0:
            palette = self._palette_probe(rgb)
            if not palette["ok"]:
                ok, reasons = False, reasons + [
                    f"colored photograph - mixed hues (colorfulness {colorfulness:.0f}/255, "
                    f"{palette['hue_bins']} hue clusters) - thermal IR is gray-scale or a smooth palette"]

        score = max(0.0, min(1.0, 1.0 - texture * 4.0))  # roughness-aware confidence fudge
        note = ("Looks like thermal satellite imagery." if ok
                else "Input does NOT resemble thermal IR satellite imagery: " + "; ".join(reasons[:3]) + ".")
        result = {"plausible": ok, "score": round(score, 3), "note": note,
                  "texture": round(texture, 4), "bright_frac": round(bright, 3),
                  "dark_frac": round(dark, 3), "mid_frac": round(mid, 3),
                  "colorfulness": round(colorfulness, 1)}
        if palette is not None:
            result["palette"] = palette
        return result

    def analyze_images(self, ir_img: np.ndarray, wv_img: Optional[np.ndarray] = None,
                       sequence_length: int = 6) -> Dict:
        """Run the CNN stack on a single (or pair of) satellite frame(s).

        Returns the full vision block: detection, intensity, ri, gradcam, previews,
        plus the *derived tabular observation* for the RF forecast chain.
        """
        if not self.models_loaded:
            raise RuntimeError("CNN vision models not loaded - cannot analyse imagery.")

        # Normalize any already-normalized [0,1] float source back to 8-bit pixel
        # range; normalize_channel treats uint8 as pixels (/255) but float [0,1] as
        # Kelvin temperatures, which would silently black-out the frame.
        def _to_pixel8(arr: np.ndarray) -> np.ndarray:
            arr = np.asarray(arr)
            if arr.dtype != np.uint8 and arr.max() <= 1.0 and arr.min() >= 0.0:
                arr = (arr * 255.0)
            return arr.astype(np.float32)

        ir_pix8 = _to_pixel8(ir_img)
        ir_gray = self._as_grayscale(ir_pix8)
        ir_norm = np.asarray(self.preprocessor.normalize_channel(ir_gray, "ir"), dtype=np.float32)

        if wv_img is not None:
            wv_gray = self._as_grayscale(_to_pixel8(wv_img))
            wv_norm = np.asarray(self.preprocessor.normalize_channel(wv_gray, "wv"), dtype=np.float32)
            wv_source = "UPLOADED"
        else:
            ir_pix = self._as_grayscale(_to_pixel8(ir_img))
            wv_smooth = cv2.GaussianBlur(ir_pix, (9, 9), 0)
            wv_norm = np.asarray(self.preprocessor.normalize_channel(wv_smooth, "wv"), dtype=np.float32)
            wv_source = "PROXY_FROM_IR"

        frame = np.stack([ir_norm, wv_norm], axis=0).astype(np.float32)          # [2,128,128]
        seq = np.repeat(frame[np.newaxis, ...], sequence_length, axis=0)          # [6,2,128,128]
        seq_in = seq[np.newaxis, ...].astype(np.float32)                          # [1,6,2,128,128]
        # RI tabular context (normalised wind/pressure proxies, matching training scale)
        tabular = np.array([[0.60, 0.40, 0.60, 0.98]], dtype=np.float32)

        det_logits = self.detector.run(None, {"input": seq_in[:, -1]})[0][0]
        det_probs = _softmax(det_logits)
        detected = bool(np.argmax(det_probs) == 1)
        det_conf = float(det_probs[1] if detected else det_probs[0])

        int_logits = self.intensity_model.run(None, {"input": seq_in})[0][0]
        int_probs = _softmax(int_logits)
        cat_idx = int(np.argmax(int_probs))
        cat_name = list(CATEGORY_BANDS_KNOTS)[cat_idx]
        int_conf = float(int_probs[cat_idx])

        ri_logit = self.ri_model.run(None, {"seq": seq_in, "tabular": tabular})[0]
        ri_prob = float(1.0 / (1.0 + np.exp(-ri_logit.reshape(-1)[0])))

        gradcam = {
            "heatmap_available": False,
            "target_class": "Cyclonic Pattern (Eye / Rainbands)",
            "overlay_image_base64": None,
            "heatmap_image_base64": None,
            "explanation": ("Grad-CAM heatmap requires the PyTorch runtime; the lightweight "
                            "cloud build (ONNX Runtime) omits it. The detection/intensity/RI "
                            "verdicts are unaffected."),
        }

        plausibility = self._plausibility(ir_norm, rgb=ir_pix8)

        # Derived tabular observation (what the RF chain consumes)
        lo, hi = CATEGORY_BANDS_KNOTS[cat_name]
        wind = round((lo + min(hi, lo + 17)) / 2.0, 1)
        pressure = CATEGORY_PRESSURE_HPA[cat_name]

        return {
            "detection": {
                "cyclone_detected": detected,
                "confidence": round(det_conf, 4),
                "class_name": "Cyclonic" if detected else "Non-Cyclonic",
                "class_probabilities": {
                    "Non-Cyclonic": round(float(det_probs[0]), 4),
                    "Cyclonic": round(float(det_probs[1]), 4),
                },
                "source_mode": "MODEL_PREDICTION",
            },
            "intensity": {
                "category": cat_name,
                "category_index": cat_idx,
                "confidence": round(int_conf, 4),
                "wind_speed_range_knots": f"{lo}-{hi - 1} kt",
                "wind_speed_range_kmh": _WIND_RANGES_KMH[cat_idx],
                "class_probabilities": {
                    name: round(float(int_probs[i]), 4) for i, name in enumerate(CATEGORY_BANDS_KNOTS)
                },
                "source_mode": "MODEL_PREDICTION",
            },
            "ri": {
                "rapid_intensification_risk": round(ri_prob, 4),
                "risk_level": "HIGH" if ri_prob >= 0.65 else ("MODERATE" if ri_prob >= 0.35 else "LOW"),
                "forecast_window_hours": 24,
                "threshold_definition": ">= 30 knots sustained wind speed increase within 24 hours",
                "source_mode": "MODEL_PREDICTION",
            },
            "gradcam": gradcam,
            "wv_source": wv_source,
            "input_frames": 1,
            "sequence_build": f"static-replicate-x{sequence_length}",
            "plausibility": plausibility,
            "observed": {  # the tabular observation DERIVED from the imagery
                "wind_knots": wind,
                "category": cat_name,
                "wind_speed_range_knots": f"{lo}-{hi - 1} kt",
                "pressure_hpa": pressure,
                "detected": detected,
            },
        }


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


_chain = None


def get_vision_chain() -> VisionChain:
    global _chain
    if _chain is None:
        _chain = VisionChain()
    return _chain


def models_ready() -> bool:
    """Non-loading readiness check: models exist on disk."""
    return (os.path.exists("models/onnx/detector.onnx")
            and os.path.exists("models/onnx/intensity.onnx")
            and os.path.exists("models/onnx/ri.onnx"))


def analyse_observation(obs: ObservationInput, ir_img: np.ndarray,
                        wv_img: Optional[np.ndarray] = None) -> Dict:
    """CNN image analysis -> derived observation -> RF forecast chain. Merged response."""
    chain = get_vision_chain()
    vision = chain.analyze_images(ir_img, wv_img)
    derived = vision["observed"]

    # Overlay the CNN-derived state onto the caller-supplied observation so the
    # RF chain predicts from *what the imagery shows* (pressure/coords/motion stay
    # from the caller when meaningful).
    merged = ObservationInput(
        name=obs.name,
        latitude=obs.latitude,
        longitude=obs.longitude,
        wind_knots=derived["wind_knots"] if derived["wind_knots"] else obs.wind_knots,
        pressure_hpa=derived["pressure_hpa"] if derived["pressure_hpa"] else obs.pressure_hpa,
        storm_speed_knots=obs.storm_speed_knots,
        storm_direction_deg=obs.storm_direction_deg,
        month=obs.month,
        hour=obs.hour,
        obs_num=obs.obs_num,
    )

    rf = forecast_service.full_forecast(merged, source_mode="UPLOAD_ANALYSIS", name=obs.name)
    det = vision["detection"]
    inten = vision["intensity"]
    ri = vision["ri"]
    plausible = vision["plausibility"]["plausible"]

    # CNN-derived current state drives the headline; RF drives track/formation/outlook.
    rf["cyclone_detected"] = det["cyclone_detected"]
    rf["detection_confidence"] = det["confidence"]
    rf["intensity_category"] = inten["category"]
    rf["intensity_confidence"] = inten["confidence"]
    rf["wind_speed_knots"] = derived["wind_knots"]
    rf["rapid_intensification_risk"] = ri["rapid_intensification_risk"]
    rf["risk_level"] = ri["risk_level"]
    rf["intensity_class_probabilities"] = inten["class_probabilities"]
    rf["intensity"] = {
        "predicted_wind_knots": derived["wind_knots"],
        "category": inten["category"],
        "category_confidence": inten["confidence"],
    }
    rf["system_type"] = ("CNN (ResNet-18 detection + CNN-ConvLSTM intensity) -> "
                         "RandomForest ensemble (track/formation)")
    rf["source_mode"] = "UPLOAD_ANALYSIS"

    if not plausible:
        # The input does not look like thermal satellite imagery — do NOT present
        # a confident cyclone verdict. Flag inconclusive instead.
        rf["input_plausible"] = False
        rf["input_note"] = vision["plausibility"]["note"]
        rf["input_verified"] = False
        rf["verification_note"] = vision["plausibility"]["note"]
        rf["detection"] = {
            "cyclone_detected": None,
            "confidence": round(1.0 - vision["plausibility"]["score"], 4),
            "class_name": "Inconclusive",
            "class_probabilities": {k: 0.0 for k in det["class_probabilities"]},
            "source_mode": "PLAUSIBILITY_GATE",
        }
        rf["cyclone_detected"] = None
        rf["intensity_category"] = "Inconclusive"
        rf["intensity_confidence"] = None
        rf["intensity"] = {
            "predicted_wind_knots": None,
            "category": "Inconclusive",
            "category_confidence": None,
        }
        rf["wind_speed_knots"] = None
        rf["risk_level"] = "INCONCLUSIVE"
        rf["rapid_intensification_risk"] = None
        rf["alert_message"] = ("Rejected input: the uploaded image does not resemble thermal IR satellite "
                               "imagery and was not classified. " + vision["plausibility"]["note"])
        rf["intensity_class_probabilities"] = {c: 0.0 for c in det["class_probabilities"]}
        rf["formation"] = {"probability_becomes_cyclone": None, "verdict": "INCONCLUSIVE"}
        rf["outlook"] = []
        rf["trail"] = []
    else:
        rf["detection"] = det
        rf["input_plausible"] = True
        rf["input_note"] = vision["plausibility"]["note"]
        rf["input_verified"] = True
        rf["verification_note"] = vision["plausibility"]["note"]

    rf.update({
        "intensity_cnn": inten,
        "ri": ri,
        "gradcam": vision["gradcam"],
        "wv_source": vision["wv_source"],
        "input_frames": vision["input_frames"],
        "sequence_build": vision["sequence_build"],
        "plausibility": vision["plausibility"],
    })
    return rf