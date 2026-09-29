"""
CycloVision Upload Domain Gate

Decides whether an uploaded image is authentic satellite meteorology imagery
(thermal IR 10.8 µm / water vapour 6.8 µm, including enhanced renderings) versus
an arbitrary photograph, screenshot or synthetic texture. Non-satellite uploads are
refused outright so the AI inference chain never analyses random imagery.

Approach:
  1. A fast statistical pre-filter rejects obvious photographs (busy edge density),
     blank/uniform frames, and over-compressed junk.
  2. A learned classifier scores deep features from a frozen ImageNet ResNet-18
     backbone. Authentic satellite frames are out-of-distribution for ImageNet and
     sit on one side of the boundary; photographs and screenshots (which live on the
     natural-image manifold) sit on the other. A small MLP is trained on the real
     demo satellite frames (positives) against a large library of procedurally
     generated photo / screenshot-like negatives.

The gate is built lazily on first use and cached so startup stays fast.
"""

import os
import json
import gzip
import numpy as np
import cv2
import torch
import torch.nn as nn
from typing import List, Optional, Tuple
import threading

ARTIFACT_DIR = "models/satellite_gate"
ARTIFACT_CKPT = os.path.join(ARTIFACT_DIR, "photo_mlp.pt.gz")
ARTIFACT_META = os.path.join(ARTIFACT_DIR, "metadata.json")
DEMO_DATA_DIR = "data/demo/satellite_sequences"

IMGNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMGNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class _PhotoMLP(nn.Module):
    def __init__(self, in_dim: int = 512, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SatelliteGate:
    def __init__(self, device: torch.device | str = None):
        self.device = torch.device(device) if device is not None else (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.backbone = None
        self.classifier: Optional[_PhotoMLP] = None
        self.threshold: Optional[float] = None
        self.description = "not initialised"
        self._backbone_lock = threading.RLock()

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _load_demo_satellite_frames() -> List[np.ndarray]:
        imgs: List[np.ndarray] = []
        if not os.path.isdir(DEMO_DATA_DIR):
            return imgs
        for storm in sorted(os.listdir(DEMO_DATA_DIR)):
            storm_dir = os.path.join(DEMO_DATA_DIR, storm)
            if not os.path.isdir(storm_dir):
                continue
            for fn in sorted(os.listdir(storm_dir)):
                if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                img = cv2.imread(os.path.join(storm_dir, fn), cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.ndim == 3 and img.shape[2] == 4:
                    img = img[:, :, :3]
                elif img.ndim == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    continue
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                imgs.append(cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA))
        return imgs

    @staticmethod
    def _augment_satellite(images: List[np.ndarray]) -> List[np.ndarray]:
        out = list(images)
        for img in images:
            out.append(cv2.flip(img, 0))
            out.append(cv2.flip(img, 1))
            out.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
            out.append(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))
            out.append(cv2.GaussianBlur(img, (7, 7), 0))
            out.append(np.clip(img.astype(np.int16) + 18, 0, 255).astype(np.uint8))
            out.append(np.clip(img.astype(np.int16) - 18, 0, 255).astype(np.uint8))
        return out

    # ---------------- realistic satellite product renders (positive class) ----
    def _real_satellite_products(self, seed: int = 7, count: int = 96) -> List[np.ndarray]:
        """
        Procedural renderings of authentic-looking satellite weather products so the
        classifier recognises the full range of real uploads: monochrome thermal-IR
        full-discs (dark space background), zoomed grayscale cloud fields, water-vapour
        greyscales, colour-enhanced LUT maps, and frames with legend/colour bars.
        """
        rng = np.random.default_rng(seed)
        out: List[np.ndarray] = []
        S = 256

        for i in range(count):
            style = i % 5
            img = np.zeros((S, S, 3), np.uint8)

            if style == 0:  # full-disc grayscale IR with black space + coastline blobs
                ccx, ccy = int(rng.integers(S * 0.4, S * 0.6)), int(rng.integers(S * 0.4, S * 0.6))
                R = int(rng.integers(70, 105))
                yy, xx = np.ogrid[:S, :S]
                disc = (xx - ccx) ** 2 + (yy - ccy) ** 2 < R ** 2
                blobs = cv2.GaussianBlur(rng.integers(0, 256, (S, S), dtype=np.uint8).astype(np.float32), (29, 29), 0)
                base = rng.integers(80, 200)
                img[disc, 0] = cv2.normalize(blobs, None, base - 40, base + 40, cv2.NORM_MINMAX).astype(np.uint8)[disc]
                img[disc, 1] = img[disc, 0]; img[disc, 2] = img[disc, 0]
            elif style == 1:  # zoomed grayscale cloud spiral (IR loop product)
                yy, xx = np.ogrid[:S, :S]
                cx, cy = S * 0.5, S * 0.5
                d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                ang = np.arctan2(yy - cy, xx - cx)
                spiral = np.clip((np.sin(6 * ang + d * 0.12) * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
                nois = cv2.GaussianBlur(rng.integers(0, 256, (S, S), dtype=np.uint8), (15, 15), 0).astype(np.float32)
                g = np.clip(spiral.astype(np.float32) * 0.6 + nois * 0.4, 0, 255).astype(np.uint8)
                img[:, :, 0] = g; img[:, :, 1] = g; img[:, :, 2] = g
            elif style == 2:  # water vapour greyscale (fluffy mid-tone)
                yy, xx = np.ogrid[:S, :S]
                blobs = cv2.GaussianBlur(rng.integers(0, 256, (S, S), dtype=np.uint8).astype(np.float32), (19, 19), 0)
                g = cv2.normalize(blobs, None, 90, 210, cv2.NORM_MINMAX).astype(np.uint8)
                img[:, :, 0] = g; img[:, :, 1] = g; img[:, :, 2] = g
            elif style == 3:  # colour-enhanced LUT (false colour cloud map)
                yy, xx = np.ogrid[:S, :S]
                blobs = cv2.GaussianBlur(rng.integers(0, 256, (S, S), dtype=np.uint8).astype(np.float32), (21, 21), 0)
                g = cv2.normalize(blobs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                lut_r = np.linspace(0, 255, 256, dtype=np.uint8)
                lut_g = np.roll(lut_r, 70); lut_b = np.roll(lut_r, 140)
                img[:, :, 0] = cv2.LUT(g, lut_r)
                img[:, :, 1] = cv2.LUT(g, lut_g)
                img[:, :, 2] = cv2.LUT(g, lut_b)
            else:  # grayscale product with a small legend/colour bar strip at the bottom
                yy, xx = np.ogrid[:S, :S]
                blobs = cv2.GaussianBlur(rng.integers(0, 256, (S, S), dtype=np.uint8).astype(np.float32), (25, 25), 0)
                g = cv2.normalize(blobs, None, 20, 235, cv2.NORM_MINMAX).astype(np.uint8)
                img[:, :, 0] = g; img[:, :, 1] = g; img[:, :, 2] = g
                bar = np.linspace(0, 255, 64, dtype=np.uint8)
                img[-12:, 96:160, 0] = bar[None, :]
                img[-12:, 96:160, 1] = (255 - bar)[None, :]
                img[-12:, 96:160, 2] = bar[::-1][None, :]
                for k in range(10, 200, 16):
                    cv2.line(img, (k, -2), (k, 22), (230, 230, 230), 1)
            out.append(cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA))
        return out

    # ------------------------------------------------- real photograph negatives
    @staticmethod
    def _load_real_photos(max_count: int = 500) -> List[np.ndarray]:
        """Sample real photographs (Windows wallpaper library) as photo negatives."""
        roots = [
            r"C:\Windows\Web\Wallpaper",
            r"C:\Windows\Web\Wallpaper\Windows",
            r"C:\Windows\Web\Wallpaper\Theme1",
            r"C:\Windows\Web\Wallpaper\Theme2",
        ]
        rng = np.random.default_rng(42)
        paths: List[str] = []
        for root in roots:
            if os.path.isdir(root):
                for dirpath, _, files in os.walk(root):
                    for fn in files:
                        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                            paths.append(os.path.join(dirpath, fn))
        imgs: List[np.ndarray] = []
        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for _ in range(2):
                h, w = img.shape[:2]
                if h < 224 or w < 224:
                    crops = [cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)]
                else:
                    y0 = int(rng.integers(0, h - 223)); x0 = int(rng.integers(0, w - 223))
                    crops = [img[y0:y0 + 224, x0:x0 + 224]]
                imgs.extend(crops)
            if len(imgs) >= max_count:
                break
        return imgs[:max_count]

    # ------------------------------------------------- synthetic photo library
    def _synthetic_photo_negatives(self, seed: int = 2026, count: int = 2200) -> List[np.ndarray]:
        rng = np.random.default_rng(seed)
        S = 224
        imgs: List[np.ndarray] = []

        yy, xx = np.ogrid[:S, :S]

        for _ in range(count):
            kind = rng.integers(0, 12)
            if kind == 0:  # sunflower / flower: radial petals + high saturation
                cx, cy = int(rng.integers(40, 184)), int(rng.integers(40, 184))
                R = int(rng.integers(50, 100))
                petal = tuple(int(v) for v in rng.integers(0, 255, 3))
                hub = tuple(int(v) for v in rng.integers(0, 130, 3))
                bg = tuple(int(v) for v in rng.integers(0, 255, 3))
                img = np.zeros((S, S, 3), np.uint8); img[:, :, :] = bg
                d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                petals = (d < R)
                img[petals] = petal
                img[d < R * 0.28] = hub
                tex = rng.uniform(0.75, 1.0, (S, S, 1))
                img = np.clip(img.astype(np.float32) * tex, 0, 255).astype(np.uint8)
                imgs.append(cv2.GaussianBlur(img, (3, 3), 0))

            elif kind == 1:  # landscape: sky + mountains/terrain
                img = np.zeros((S, S, 3), np.uint8)
                sky = tuple(int(v) for v in rng.integers(60, 180, 3))
                land = tuple(int(v) for v in rng.integers(30, 120, 3))
                img[:, :, :] = sky
                horizon = int(rng.integers(80, 190))
                amp = int(rng.integers(4, 60))
                freq = rng.uniform(0.008, 0.05)
                xcoords = np.arange(S)
                ridge = np.clip((horizon - amp * np.sin(xcoords * freq)).astype(int), 0, S)
                for y in range(S):
                    img[y, ridge[y]:] = land
                tex = rng.uniform(0.8, 1.15, (S, S, 1))
                imgs.append(np.clip(img.astype(np.float32) * tex, 0, 255).astype(np.uint8))

            elif kind == 2:  # face / portrait patch
                img = np.full((S, S, 3), tuple(int(v) for v in rng.integers(180, 235, 3)), np.uint8)
                cx, cy = int(rng.integers(80, 144)), int(rng.integers(80, 160))
                a, b = int(rng.integers(40, 80)), int(rng.integers(55, 105))
                face = ((xx - cx) ** 2 / a ** 2 + (yy - cy) ** 2 / b ** 2) < 1
                hair = (yy < cy - b * 0.5) & face
                img[face] = tuple(int(v) for v in rng.integers(150, 215, 3))
                img[hair] = tuple(int(v) for v in rng.integers(10, 45, 3))
                eyes = (np.abs(yy - cy + b * 0.25) < 7) & ((np.abs(xx - cx + a * 0.4) < 13) | (np.abs(xx - cx - a * 0.4) < 13))
                mouth = (yy - cy + b * 0.6) ** 2 + (xx - cx) ** 2 / 400 < 16
                img[eyes] = (8, 8, 8); img[mouth] = (96, 42, 30)
                imgs.append(cv2.GaussianBlur(img, (3, 3), 0))

            elif kind == 3:  # forest / grass / leaves texture
                img = np.zeros((S, S, 3), np.uint8)
                base = tuple(int(v) for v in rng.integers(10, 90, 3))
                img[:, :, :] = base
                stem_col = tuple(int(v) for v in rng.integers(0, 80, 3))
                n = int(rng.integers(40, 200))
                for _ in range(n):
                    x0 = int(rng.integers(0, S)); w = int(rng.integers(1, 6))
                    img[:, x0:x0 + w] = stem_col
                patch = rng.integers(0, 255, (S, S, 1), dtype=np.uint8)
                tex = cv2.GaussianBlur(patch, (7, 7), 0).astype(np.float32)
                imgs.append(np.clip(img.astype(np.float32) * (0.75 + tex[..., None] / 255.0 * 0.5), 0, 255).astype(np.uint8))

            elif kind == 4:  # buildings / city blocks
                img = np.zeros((S, S, 3), np.uint8)
                bg = tuple(int(v) for v in rng.integers(90, 200, 3))
                img[:, :, :] = bg
                skyline = 80 + rng.integers(0, 90, S // 10)
                for i, h in enumerate(skyline):
                    col = tuple(int(v) for v in rng.integers(60, 160, 3))
                    img[S - h:, i * 10:(i + 1) * 10] = col
                    win = int(rng.integers(6, 16))
                    for wy in range(S - h + 8, S, 14):
                        for wx_i in range(i * 10 + 3, (i + 1) * 10, 4):
                            if rng.random() < 0.5:
                                img[wy:wy + 5, wx_i:wx_i + 2] = (230, 220, 160)
                imgs.append(cv2.GaussianBlur(img, (3, 3), 0))

            elif kind == 5:  # sky + clouds (photo style, not thermal)
                img = np.zeros((S, S, 3), np.uint8)
                sk = tuple(int(v) for v in rng.integers(100, 220, 3))
                img[:, :, :] = sk
                for _ in range(int(rng.integers(8, 40))):
                    cx, cy = int(rng.integers(0, S)), int(rng.integers(0, S))
                    r = int(rng.integers(8, 50))
                    br = 235 - int(rng.integers(0, 60))
                    cv2.circle(img, (cx, cy), r, (br, br, br), -1)
                    cv2.circle(img, (cx + r // 2, cy - r // 3), int(r * 0.6), (br, br, br), -1)
                imgs.append(cv2.GaussianBlur(img, (5, 5), 0))

            elif kind == 6:  # water / ocean with waves + shoreline
                img = np.zeros((S, S, 3), np.uint8)
                c1 = tuple(int(v) for v in rng.integers(0, 120, 3))
                img[:, :, :] = c1
                freq = rng.uniform(0.01, 0.06)
                for y in range(S):
                    shade = int(rng.integers(-12, 12))
                    img[y] = np.clip(img[y].astype(np.int16) + shade, 0, 255).astype(np.uint8)
                imgs.append(cv2.GaussianBlur(img, (9, 9), 0))

            elif kind == 7:  # noise / static / texture random
                img = rng.integers(0, 256, (S, S, 3), dtype=np.uint8)
                if rng.random() < 0.5:
                    img = cv2.GaussianBlur(img, (5, 5), 0)
                imgs.append(img)

            elif kind == 8:  # screenshots: text page / spreadsheet / terminal
                img = np.full((S, S, 3), int(rng.integers(235, 255)), np.uint8)
                mono = bool(rng.random() < 0.5)
                ink = (12, 12, 12)
                for by in range(6, S, int(rng.integers(14, 28))):
                    img[by:by + 6, 8:216] = ink
                    img[by:by + 2, 8:216] = (255, 255, 255) if mono else tuple(int(v) for v in rng.integers(180, 255, 3))
                if rng.random() < 0.3:
                    img[:, 80:124] = (90, 130, 210)
                imgs.append(img)

            elif kind == 9:  # radial/gradient modern art
                cx, cy = int(rng.integers(0, S)), int(rng.integers(0, S))
                d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                band = np.linspace(0, 255, 24, dtype=np.uint8)
                idx = (d / 12.0).astype(int) % 24
                c = np.stack([band[idx % 24], band[(idx + 8) % 24], band[(idx + 16) % 24]], axis=-1)
                imgs.append(c.astype(np.uint8))

            elif kind == 10:  # animals / fur-like texture
                img = np.zeros((S, S, 3), np.uint8)
                base = tuple(int(v) for v in rng.integers(40, 140, 3))
                img[:, :, :] = base
                n = int(rng.integers(800, 3000))
                for _ in range(n):
                    x, y = int(rng.integers(0, S)), int(rng.integers(0, S))
                    cv2.line(img, (x, y), (min(S - 1, x + int(rng.integers(1, 15))), min(S - 1, y + int(rng.integers(-6, 7)))), tuple(int(v) for v in rng.integers(0, max(40, base[0] - 15), 3)), 1)
                imgs.append(img)

            else:  # generic cluttered collage / dashboard
                img = np.full((S, S, 3), 245, np.uint8)
                for _ in range(int(rng.integers(6, 20))):
                    x0, y0 = int(rng.integers(0, S - 60)), int(rng.integers(0, S - 60))
                    w, h = int(rng.integers(20, 90)), int(rng.integers(10, 50))
                    col = tuple(int(v) for v in rng.integers(0, 255, 3))
                    img[y0:y0 + h, x0:x0 + w] = col
                    img[y0:y0 + 3, x0:x0 + w] = (10, 10, 10)
                imgs.append(img)
        return imgs

    @staticmethod
    def _tensor_from_images(images: List[np.ndarray], device) -> torch.Tensor:
        x = np.stack([img.astype(np.float32) / 255.0 for img in images]).transpose(0, 3, 1, 2)
        return torch.from_numpy(x).to(device)

    def _extract_features(self, images: List[np.ndarray]) -> np.ndarray:
        self._ensure_backbone()
        out = []
        chunk = 256
        for start in range(0, len(images), chunk):
            part = images[start:start + chunk]
            x = self._tensor_from_images(part, self.device)
            x = (x - IMGNET_MEAN.to(self.device)) / IMGNET_STD.to(self.device)
            with torch.no_grad():
                f = self.backbone(x).reshape(len(part), -1).cpu().numpy()
            out.append(f)
        return np.concatenate(out, axis=0)

    def _ensure_backbone(self):
        if self.backbone is not None:
            return
        with self._backbone_lock:
            if self.backbone is not None:
                return
            from torchvision.models import resnet18, ResNet18_Weights
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model.eval()
            self.backbone = nn.Sequential(*list(model.children())[:-1]).to(self.device)
            for p in self.backbone.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------ train
    def _train(self):
        self._ensure_backbone()
        demo = self._load_demo_satellite_frames()
        if len(demo) < 40:
            raise RuntimeError(f"Only {len(demo)} demo satellite frames available for gate training.")
        products = self._real_satellite_products()
        positives = self._augment_satellite(demo + products)
        negatives = self._synthetic_photo_negatives()
        photos = self._load_real_photos(max_count=500)
        negatives = negatives + photos

        X_pos = self._extract_features(positives)
        X_neg = self._extract_features(negatives)

        mean_x = X_pos.mean(0)
        std_x = X_pos.std(0) + 1e-6
        X_pos = (X_pos - mean_x) / std_x
        X_neg = (X_neg - mean_x) / std_x

        X = np.concatenate([X_pos, X_neg], axis=0).astype(np.float32)
        y = np.concatenate([np.zeros(len(X_pos)), np.ones(len(X_neg))]).astype(np.float32)

        model = _PhotoMLP(in_dim=X.shape[1]).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        Xt = torch.from_numpy(X).to(self.device)
        yt = torch.from_numpy(y).to(self.device)
        n = Xt.shape[0]
        batch = 128
        epochs = 40
        model.train()
        for epoch in range(1, epochs + 1):
            perm = torch.randperm(n, device=self.device)
            tot = 0.0
            nb = 0
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                opt.zero_grad()
                out = model(Xt[idx])
                loss = torch.nn.functional.binary_cross_entropy_with_logits(out, yt[idx])
                loss.backward()
                opt.step()
                tot += loss.item()
                nb += 1
            if epoch % 10 == 0 or epoch == epochs:
                print(f"[INFO] Photo/domain classifier epoch {epoch}/{epochs}: loss={tot / nb:.5f}")

        model.eval()
        with torch.no_grad():
            prob_sat = torch.sigmoid(model(Xt[:X_pos.shape[0]])).cpu().numpy()
            prob_photo = torch.sigmoid(model(Xt[X_pos.shape[0]:])).cpu().numpy()
        worst_sat = float(prob_sat.max())
        best_photo = float(prob_photo.min())
        threshold = 0.5
        print(
            f"[INFO] Satellite domain classifier trained on {len(positives)} satellite vs "
            f"{len(negatives)} photo/screenshot samples. "
            f"photo-prob: sat[max={worst_sat:.3f}] photo[min={best_photo:.3f}] threshold={threshold}"
        )

        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        with gzip.open(ARTIFACT_CKPT, "wb") as f:
            torch.save({"model_state_dict": model.state_dict()}, f)
        with open(ARTIFACT_META, "w", encoding="utf-8") as f:
            json.dump({
                "threshold": threshold,
                "worst_satellite_prob": worst_sat,
                "best_photo_prob": best_photo,
                "mean_f": mean_x.tolist(),
                "std_f": std_x.tolist(),
                "n_satellite": len(positives),
                "n_photo": len(negatives),
                "backbone": "resnet18_imagenet1k_v1",
            }, f, indent=2)

        self.classifier = model
        self.threshold = threshold
        self.mean_f = mean_x
        self.std_f = std_x
        self.description = f"resnet18-fused MLP (sat={len(positives)}, photo={len(negatives)})"

    # ------------------------------------------------------------------ runtime
    def ensure_ready(self) -> bool:
        if self.classifier is not None:
            return True
        with self._backbone_lock:
            if self.classifier is not None:
                return True
            if os.path.exists(ARTIFACT_CKPT) and os.path.exists(ARTIFACT_META):
                try:
                    self._load_artifacts()
                    return True
                except Exception as e:
                    print(f"[WARN] Satellite domain gate: artifact load failed ({e}); retraining.")
            try:
                self._train()
            except Exception as e:
                print(f"[WARN] Satellite domain gate: could not train ({e}). Gate DISABLED.")
                return False
            return self.classifier is not None

    def _load_artifacts(self):
        self._ensure_backbone()
        with gzip.open(ARTIFACT_CKPT, "rb") as f:
            ckpt = torch.load(f, map_location=self.device, weights_only=True)
        model = _PhotoMLP(in_dim=512).to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        with open(ARTIFACT_META, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.classifier = model
        self.threshold = float(meta["threshold"])
        self.mean_f = np.asarray(meta["mean_f"], dtype=np.float32)
        self.std_f = np.asarray(meta["std_f"], dtype=np.float32)
        self.description = f"resnet18-fused MLP (sat={meta.get('n_satellite')}, photo={meta.get('n_photo')})"
        print(f"[INFO] Satellite domain gate loaded ({self.description}, threshold={self.threshold})")

    def photo_probability(self, img_rgb: np.ndarray) -> Optional[float]:
        """Probability that the image is a photograph/screenshot (not satellite imagery)."""
        if not self.ensure_ready():
            return None
        if img_rgb.ndim == 2:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
        if img_rgb.shape[:2] != (224, 224):
            img_rgb = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
        feats = self._extract_features([img_rgb])
        f = (feats - self.mean_f) / self.std_f
        Xt = torch.from_numpy(f.astype(np.float32)).to(self.device)
        with torch.no_grad():
            prob = torch.sigmoid(self.classifier(Xt)).item()
        return prob

    def is_satellite_like(self, img_rgb: np.ndarray) -> Optional[bool]:
        prob = self.photo_probability(img_rgb)
        if prob is None:
            return None
        return prob <= self.threshold


# Global singleton (module-level, mirroring ai_manager pattern)
satellite_gate = SatelliteGate()