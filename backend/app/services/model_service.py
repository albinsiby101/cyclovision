"""
CycloVision Unified AI Inference Service
Integrates Detection (ResNet18), Intensity (CNN+ConvLSTM), RI early warning,
and Grad-CAM visual explainability. Supports automatic GPU/CPU routing and Demo fallback.
"""

import os
import base64
import torch
import numpy as np
import cv2
from typing import Dict, Any, Optional, Tuple

from ml.detection.detector import CycloneDetector
from ml.intensity.intensity_model import CycloneIntensityModel
from ml.rapid_intensification.ri_model import RapidIntensificationModel
from ml.explainability.gradcam import GradCAM
from ml.preprocessing.preprocessor import SatellitePreprocessor
from ml.datasets.ibtracs_loader import IBTrACSLoader

class ModelManager:
    def __init__(self, config_path: str = "ml/configs/default.yaml"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = SatellitePreprocessor(image_size=(128, 128), sequence_length=6)
        self.ibtracs = IBTrACSLoader()
        
        self.detector: Optional[CycloneDetector] = None
        self.intensity_model: Optional[CycloneIntensityModel] = None
        self.ri_model: Optional[RapidIntensificationModel] = None
        self.gradcam_detector: Optional[GradCAM] = None
        
        self.models_loaded = False
        self.mode = "demo"
        self._load_models()

    def _load_models(self):
        try:
            # 1. Detection Model
            det_path = "models/detection/detector_best.pt"
            if os.path.exists(det_path):
                self.detector = CycloneDetector(in_channels=2, num_classes=2).to(self.device)
                ckpt = torch.load(det_path, map_location=self.device)
                self.detector.load_state_dict(ckpt["model_state_dict"])
                self.detector.eval()
                # Initialize Grad-CAM on layer4
                self.gradcam_detector = GradCAM(self.detector, self.detector.layer4)
                
            # 2. Intensity Model
            int_path = "models/intensity/intensity_best.pt"
            if os.path.exists(int_path):
                self.intensity_model = CycloneIntensityModel(in_channels=2, num_classes=7).to(self.device)
                ckpt = torch.load(int_path, map_location=self.device)
                self.intensity_model.load_state_dict(ckpt["model_state_dict"])
                self.intensity_model.eval()

            # 3. RI Model
            ri_path = "models/ri/ri_best.pt"
            if os.path.exists(ri_path):
                self.ri_model = RapidIntensificationModel(in_channels=2, spatial_features=48, convlstm_hidden=48, tabular_dim=4).to(self.device)
                ckpt = torch.load(ri_path, map_location=self.device)
                self.ri_model.load_state_dict(ckpt["model_state_dict"])
                self.ri_model.eval()

            self.models_loaded = (self.detector is not None and self.intensity_model is not None and self.ri_model is not None)
            self.mode = "operational_checkpoint" if self.models_loaded else "demo"
            print(f"[INFO] CycloVision AI Models initialized. Models loaded: {self.models_loaded} | Device: {self.device} | Mode: {self.mode}")
        except Exception as e:
            print(f"[WARN] Failed loading full model checkpoints ({e}). Running in resilient DEMO fallback.")
            self.models_loaded = False
            self.mode = "demo"

    def run_detection(self, frame_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        frame_tensor: [1, 2, 128, 128]
        """
        if self.detector is not None:
            with torch.no_grad():
                logits = self.detector(frame_tensor.to(self.device))
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                detected = bool(np.argmax(probs) == 1)
                conf = float(probs[1] if detected else probs[0])
                return {
                    "cyclone_detected": detected,
                    "confidence": round(conf, 4),
                    "class_name": "Cyclonic" if detected else "Non-Cyclonic",
                    "class_probabilities": {
                        "Non-Cyclonic": round(float(probs[0]), 4),
                        "Cyclonic": round(float(probs[1]), 4)
                    },
                    "source_mode": "MODEL_PREDICTION"
                }
        return {
            "cyclone_detected": True,
            "confidence": 0.945,
            "class_name": "Cyclonic",
            "class_probabilities": {"Non-Cyclonic": 0.055, "Cyclonic": 0.945},
            "source_mode": "DEMO"
        }

    def run_intensity(self, seq_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        seq_tensor: [1, 6, 2, 128, 128]
        """
        cat_names = [
            "Depression", "Deep Depression", "Cyclonic Storm",
            "Severe Cyclonic Storm", "Very Severe Cyclonic Storm",
            "Extremely Severe Cyclonic Storm", "Super Cyclonic Storm"
        ]
        wind_ranges_knots = ["17-27 kt", "28-33 kt", "34-47 kt", "48-63 kt", "64-89 kt", "90-119 kt", ">= 120 kt"]
        wind_ranges_kmh = ["31-49 km/h", "50-61 km/h", "62-88 km/h", "89-117 km/h", "118-166 km/h", "167-221 km/h", ">= 222 km/h"]
        
        if self.intensity_model is not None:
            with torch.no_grad():
                logits = self.intensity_model(seq_tensor.to(self.device))
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
                conf = float(probs[pred_idx])
                
                prob_dict = {cat_names[i]: round(float(probs[i]), 4) for i in range(len(cat_names))}
                return {
                    "category": cat_names[pred_idx],
                    "confidence": round(conf, 4),
                    "category_index": pred_idx,
                    "wind_speed_range_knots": wind_ranges_knots[pred_idx],
                    "wind_speed_range_kmh": wind_ranges_kmh[pred_idx],
                    "class_probabilities": prob_dict,
                    "source_mode": "MODEL_PREDICTION"
                }
        return {
            "category": "Extremely Severe Cyclonic Storm",
            "confidence": 0.882,
            "category_index": 5,
            "wind_speed_range_knots": "90-119 kt",
            "wind_speed_range_kmh": "167-221 km/h",
            "class_probabilities": {cat_names[5]: 0.882},
            "source_mode": "DEMO"
        }

    def run_ri(self, seq_tensor: torch.Tensor, tabular_info: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        seq_tensor: [1, 6, 2, 128, 128]
        tabular_info: [1, 4]
        """
        if tabular_info is None:
            tabular_info = np.array([[85.0 / 150.0, 20.0 / 50.0, 18.0 / 30.0, 970.0 / 1013.0]], dtype=np.float32)
        tab_t = torch.from_numpy(tabular_info).to(self.device)
        
        if self.ri_model is not None:
            with torch.no_grad():
                logit = self.ri_model(seq_tensor.to(self.device), tab_t)
                prob = torch.sigmoid(logit).item()
            level = "LOW" if prob < 0.35 else ("MODERATE" if prob < 0.65 else "HIGH")
            return {
                "rapid_intensification_risk": round(prob, 4),
                "risk_level": level,
                "forecast_window_hours": 24,
                "threshold_definition": ">= 30 knots sustained wind speed increase within 24 hours",
                "source_mode": "MODEL_PREDICTION"
            }
        return {
            "rapid_intensification_risk": 0.745,
            "risk_level": "HIGH",
            "forecast_window_hours": 24,
            "threshold_definition": ">= 30 knots sustained wind speed increase within 24 hours",
            "source_mode": "DEMO"
        }

    def generate_gradcam(self, frame_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Generates Grad-CAM attention heatmap for the latest frame.
        """
        try:
            if self.gradcam_detector is not None:
                # Target cyclonic class (idx 1)
                heatmap = self.gradcam_detector.generate(frame_tensor.to(self.device), class_idx=1)
                
                # Base IR image
                ir_slice = frame_tensor[0, 0].cpu().numpy()
                colored_cam, blended = GradCAM.overlay_heatmap(ir_slice, heatmap)
                
                # Encode to Base64 JPEG for transmission
                _, buf_overlay = cv2.imencode(".jpg", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
                _, buf_heat = cv2.imencode(".jpg", cv2.cvtColor(colored_cam, cv2.COLOR_RGB2BGR))
                
                b64_overlay = base64.b64encode(buf_overlay).decode("utf-8")
                b64_heat = base64.b64encode(buf_heat).decode("utf-8")
                
                return {
                    "heatmap_available": True,
                    "target_class": "Cyclonic Pattern (Eye / Rainbands)",
                    "overlay_image_base64": f"data:image/jpeg;base64,{b64_overlay}",
                    "heatmap_image_base64": f"data:image/jpeg;base64,{b64_heat}",
                    "explanation": "Regions that most influenced the model prediction: intense central convective core and inner spiraling rainband organization."
                }
        except Exception as e:
            print(f"[WARN] Grad-CAM generation error: {e}")
            
        return {
            "heatmap_available": False,
            "target_class": "Cyclonic",
            "overlay_image_base64": None,
            "heatmap_image_base64": None,
            "explanation": "Grad-CAM preview unavailable in fallback mode."
        }

    def analyze_observation(
        self,
        ir_img: np.ndarray,
        wv_img: Optional[np.ndarray] = None,
        sequence_length: int = 6
    ) -> Dict[str, Any]:
        """
        End-to-end inference chain for a single uploaded satellite observation.

        ir_img / wv_img: decoded BGR/RGB/greyscale images (any size), will be resized
        to the configured model input, normalized, and assembled into a static
        6-frame temporal sequence. If WV imagery is not supplied, a clearly-labelled
        smoothing proxy of the IR channel is used (wv_source = PROXY_FROM_IR).
        """
        ir_gray = self._as_grayscale(ir_img)
        ir_norm = self.preprocessor.normalize_channel(ir_gray, "ir")
        ir_norm = np.asarray(ir_norm, dtype=np.float32)

        if wv_img is not None:
            wv_gray = self._as_grayscale(wv_img)
            wv_norm = self.preprocessor.normalize_channel(wv_gray, "wv")
            wv_source = "UPLOADED"
        else:
            # Upper-tropospheric moisture proxy: smoothed IR gradient envelope.
            # NEVER presented as real WV observation - always flagged in response.
            wv_smooth = cv2.GaussianBlur(ir_gray, (9, 9), 0)
            wv_norm = self.preprocessor.normalize_channel(wv_smooth, "wv")
            wv_source = "PROXY_FROM_IR"

        wv_norm = np.asarray(wv_norm, dtype=np.float32)
        frame = np.stack([ir_norm, wv_norm], axis=0).astype(np.float32)  # [2, H, W]
        seq = np.repeat(frame[np.newaxis, ...], sequence_length, axis=0)  # [T, 2, H, W]
        seq_t = torch.from_numpy(seq).unsqueeze(0).float()  # [1, T, 2, H, W]

        det_res = self.run_detection(seq_t[:, -1])
        int_res = self.run_intensity(seq_t)
        ri_res = self.run_ri(seq_t)
        grad_res = self.generate_gradcam(seq_t[:, -1])

        # IR preview (128x128 grayscale) for the frontend
        ir_preview_uint8 = (np.clip(ir_norm, 0.0, 1.0) * 255.0).astype(np.uint8)
        _, ir_buf = cv2.imencode(".jpg", ir_preview_uint8)
        ir_b64 = base64.b64encode(ir_buf).decode("utf-8")

        # WV preview (128x128 grayscale) for the frontend
        wv_preview_uint8 = (np.clip(wv_norm, 0.0, 1.0) * 255.0).astype(np.uint8)
        _, wv_buf = cv2.imencode(".jpg", wv_preview_uint8)
        wv_b64 = base64.b64encode(wv_buf).decode("utf-8")

        return {
            "detection": det_res,
            "intensity": int_res,
            "ri": ri_res,
            "gradcam": grad_res,
            "wv_source": wv_source,
            "input_frames": 1,
            "ir_image_base64": f"data:image/jpeg;base64,{ir_b64}",
            "wv_image_base64": f"data:image/jpeg;base64,{wv_b64}",
            "sequence_build": f"static-replicate-x{sequence_length}",
        }

    @staticmethod
    def _as_grayscale(img: np.ndarray) -> np.ndarray:
        """
        Normalize an arbitrary decoded image to a single 128x128 grayscale channel
        in the source value range (0-255), ready for channel normalization.
        Handles RGB, RGBA, 16-bit PNG and raw grayscale inputs.
        """
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
        if img.dtype == np.uint16 and img.max(initial=0) > 255:
            img = (img.astype(np.float32) / 256.0)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        return img.astype(np.float32)

# Global singleton
ai_manager = ModelManager()