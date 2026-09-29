"""
Generates synthetic realistic satellite test samples for Demo Mode:
- Tropical Cyclone Biparjoy (Arabian Sea, June 2023)
- Tropical Cyclone Amphan (Bay of Bengal, May 2020 - Rapid Intensification)
- Non-cyclonic oceanic convective baseline
Produces paired IR (10.8 um) and Water Vapour (6.8 um) sequences in [T, C, H, W] format,
along with sample PNGs for frontend preview.
"""

import os
import numpy as np
import cv2
from PIL import Image

def generate_cyclonic_cloud_frame(
    t_idx: int,
    total_t: int = 6,
    intensity_scale: float = 1.0,
    has_eye: bool = True,
    size: int = 128
) -> np.ndarray:
    """
    Generates realistic 2-channel [H, W, 2] array:
    Channel 0: IR (cold convective cloud tops, spiral bands)
    Channel 1: Water Vapour (upper tropospheric moisture envelope)
    """
    y, x = np.ogrid[:size, :size]
    cx = size // 2 + int(np.sin(t_idx * 0.4) * 4)
    cy = size // 2 - int((t_idx / total_t) * 8)
    
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    theta = np.arctan2(y - cy, x - cx)
    
    # Logarithmic spiral rainbands
    spiral_arms = np.sin(3.5 * theta - 0.22 * r + (t_idx * 0.35))
    spiral_mask = np.clip(spiral_arms * 0.4 + 0.6, 0.0, 1.0)
    
    # Central Dense Overcast (CDO) disk
    cdo_radius = 24.0 + 8.0 * intensity_scale
    cdo_envelope = np.exp(-(r**2) / (2 * (cdo_radius**2)))
    
    # Eye formation for severe cyclones
    eye_mask = 1.0
    if has_eye and intensity_scale > 0.5:
        eye_radius = 5.0 + 3.0 * (1.0 - intensity_scale * 0.5)
        eye_mask = 1.0 - 0.85 * np.exp(-(r**2) / (2 * (eye_radius**2)))
        
    # IR Channel (Higher intensity = colder cloud tops = high brightness contrast)
    ir = (cdo_envelope * 0.75 + spiral_mask * 0.25) * eye_mask
    ir += np.random.normal(0, 0.03, (size, size))
    ir = np.clip(ir * (0.6 + 0.4 * intensity_scale), 0.0, 1.0)
    
    # Water Vapour Channel (Smoother, broader moisture circulation)
    wv_envelope = np.exp(-(r**2) / (2 * ((cdo_radius * 1.5)**2)))
    wv = (wv_envelope * 0.7 + spiral_mask * 0.3)
    wv += np.random.normal(0, 0.02, (size, size))
    wv = np.clip(wv * (0.5 + 0.5 * intensity_scale), 0.0, 1.0)
    
    # Combine into [size, size, 2]
    frame = np.stack([ir, wv], axis=-1).astype(np.float32)
    return frame

def generate_non_cyclone_frame(t_idx: int, size: int = 128) -> np.ndarray:
    """Disorganized, non-cyclonic convective cloud clusters"""
    y, x = np.ogrid[:size, :size]
    # Disorganized patches
    ir = np.zeros((size, size), dtype=np.float32)
    for _ in range(3):
        px, py = np.random.randint(20, size - 20, 2)
        pr = np.random.randint(12, 28)
        dist = np.sqrt((x - px)**2 + (y - py)**2)
        ir += np.exp(-(dist**2) / (2 * (pr**2))) * np.random.uniform(0.3, 0.6)
    ir = np.clip(ir + np.random.normal(0, 0.04, (size, size)), 0.0, 1.0)
    wv = cv2.GaussianBlur(ir, (15, 15), 0) * 0.8
    return np.stack([ir, wv], axis=-1).astype(np.float32)

def save_sample_artifacts(base_dir: str, name: str, frames: list):
    out_dir = os.path.join(base_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    
    # Save full 6-frame sequence as numpy array [T, C, H, W]
    seq_np = np.stack(frames, axis=0) # [T, H, W, 2]
    seq_np = np.transpose(seq_np, (0, 3, 1, 2)) # [T, 2, H, W]
    np.save(os.path.join(out_dir, "sequence.npy"), seq_np)
    
    # Save PNG images for IR and WV of each frame for direct web inspection
    for i, frame in enumerate(frames):
        ir_img = (frame[:, :, 0] * 255).astype(np.uint8)
        wv_img = (frame[:, :, 1] * 255).astype(np.uint8)
        
        # False color enhancement (IR Storm Enhancer)
        ir_enhanced = cv2.applyColorMap(ir_img, cv2.COLORMAP_MAGMA)
        wv_enhanced = cv2.applyColorMap(wv_img, cv2.COLORMAP_VIRIDIS)
        
        cv2.imwrite(os.path.join(out_dir, f"frame_{i}_ir.png"), ir_img)
        cv2.imwrite(os.path.join(out_dir, f"frame_{i}_wv.png"), wv_img)
        cv2.imwrite(os.path.join(out_dir, f"frame_{i}_ir_enhanced.png"), ir_enhanced)
        cv2.imwrite(os.path.join(out_dir, f"frame_{i}_wv_enhanced.png"), wv_enhanced)
        
    print(f"Generated {name}: shape {seq_np.shape} in {out_dir}")

if __name__ == "__main__":
    demo_root = "data/demo/satellite_sequences"
    
    # 1. Cyclone Biparjoy (Arabian Sea - Very/Extremely Severe)
    biparjoy_frames = [
        generate_cyclonic_cloud_frame(i, total_t=6, intensity_scale=0.75 + i * 0.04, has_eye=True)
        for i in range(6)
    ]
    save_sample_artifacts(demo_root, "demo_biparjoy", biparjoy_frames)
    
    # 2. Cyclone Amphan (Bay of Bengal - Rapid Intensification Event)
    amphan_frames = [
        generate_cyclonic_cloud_frame(i, total_t=6, intensity_scale=0.55 + i * 0.08, has_eye=True)
        for i in range(6)
    ]
    save_sample_artifacts(demo_root, "demo_cyclone_amphan", amphan_frames)
    
    # 3. Non-Cyclonic Oceanic Convection
    non_cyclone_frames = [
        generate_non_cyclone_frame(i) for i in range(6)
    ]
    save_sample_artifacts(demo_root, "demo_non_cyclone", non_cyclone_frames)
    print("Demo satellite sequences successfully generated.")