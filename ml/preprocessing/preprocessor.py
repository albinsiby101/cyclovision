"""
CycloVision Multi-Source Satellite & Environmental Preprocessing Engine
Handles INSAT-3D/3DR IR & WV channel ingestion, missing frame imputation,
normalization, storm-centered cropping, and temporal sequence assembly.
"""

import os
import glob
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image

class SatellitePreprocessor:
    def __init__(
        self,
        image_size: Tuple[int, int] = (128, 128),
        sequence_length: int = 6,
        ir_min_max: Tuple[float, float] = (180.0, 310.0), # Brightness temperature in Kelvin
        wv_min_max: Tuple[float, float] = (200.0, 270.0)
    ):
        self.image_size = tuple(image_size)
        self.sequence_length = sequence_length
        self.ir_min, self.ir_max = ir_min_max
        self.wv_min, self.wv_max = wv_min_max

    def normalize_channel(self, arr: np.ndarray, ctype: str = "ir") -> np.ndarray:
        """
        Normalize temperature / pixel values into [0.0, 1.0] range.
        For IR, lower brightness temp represents higher/colder convective cloud tops.
        Inverted or standard normalisation is preserved consistently.
        """
        arr = arr.astype(np.float32)
        if arr.max() > 1.0 and arr.max() <= 255.0 and arr.min() >= 0:
            # If standard 8-bit image representation
            return np.clip(arr / 255.0, 0.0, 1.0)
        
        min_val = self.ir_min if ctype == "ir" else self.wv_min
        max_val = self.ir_max if ctype == "ir" else self.wv_max
        norm = (arr - min_val) / (max_val - min_val + 1e-6)
        return np.clip(norm, 0.0, 1.0)

    def crop_storm_center(
        self,
        image: np.ndarray,
        center: Optional[Tuple[int, int]] = None,
        crop_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Extract storm-centered region around cyclone eye / central dense overcast (CDO).
        """
        h, w = image.shape[:2]
        if center is None:
            cx, cy = w // 2, h // 2
        else:
            cx, cy = center
            
        target_h, target_w = crop_size if crop_size else self.image_size
        x1 = max(0, cx - target_w // 2)
        y1 = max(0, cy - target_h // 2)
        x2 = min(w, x1 + target_w)
        y2 = min(h, y1 + target_h)
        
        # Adjust if near boundary
        x1 = max(0, x2 - target_w)
        y1 = max(0, y2 - target_h)
        
        cropped = image[y1:y2, x1:x2]
        if cropped.shape[:2] != (target_h, target_w):
            cropped = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return cropped

    def combine_channels(self, ir_channel: np.ndarray, wv_channel: np.ndarray) -> np.ndarray:
        """
        Stack IR (10.8 um) and Water Vapour (6.8 um) into 2-channel tensor [H, W, 2].
        """
        if ir_channel.shape[:2] != self.image_size:
            ir_channel = cv2.resize(ir_channel, self.image_size, interpolation=cv2.INTER_AREA)
        if wv_channel.shape[:2] != self.image_size:
            wv_channel = cv2.resize(wv_channel, self.image_size, interpolation=cv2.INTER_AREA)
            
        ir_norm = self.normalize_channel(ir_channel, "ir")
        wv_norm = self.normalize_channel(wv_channel, "wv")
        
        if ir_norm.ndim == 2:
            ir_norm = np.expand_dims(ir_norm, axis=-1)
        if wv_norm.ndim == 2:
            wv_norm = np.expand_dims(wv_norm, axis=-1)
            
        # Combine into [H, W, 2]
        combined = np.concatenate([ir_norm, wv_norm], axis=-1)
        return combined

    def assemble_temporal_sequence(
        self,
        frames: List[np.ndarray],
        target_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Assemble sequence of frames into [T, C, H, W] tensor for ConvLSTM.
        Handles missing frames via linear or forward temporal interpolation.
        """
        length = target_length or self.sequence_length
        if len(frames) == 0:
            raise ValueError("Frame list is empty. Cannot construct temporal sequence.")
            
        # Temporal padding or subsampling
        if len(frames) < length:
            # Replicate last frame or linear interpolation
            pad_count = length - len(frames)
            padded = list(frames) + [frames[-1]] * pad_count
            seq = np.stack(padded, axis=0) # [T, H, W, C]
        elif len(frames) > length:
            # Sample evenly
            indices = np.linspace(0, len(frames) - 1, length, dtype=int)
            seq = np.stack([frames[i] for i in indices], axis=0)
        else:
            seq = np.stack(frames, axis=0)
            
        # Transpose from [T, H, W, C] to [T, C, H, W]
        if seq.ndim == 4 and seq.shape[-1] in (1, 2, 3):
            seq = np.transpose(seq, (0, 3, 1, 2))
            
        return seq.astype(np.float32)

    def load_frame_from_disk(self, file_path: str) -> np.ndarray:
        """
        Robust loader supporting PNG, JPG, TIFF, NPY, NPZ.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Satellite frame not found: {file_path}")
            
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".npy", ".npz"]:
            data = np.load(file_path)
            if isinstance(data, np.lib.npyio.NpzFile):
                data = data[data.files[0]]
            return data
        else:
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to read image at {file_path}")
            if img.ndim == 3 and img.shape[2] == 3:
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img