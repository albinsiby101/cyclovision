"""Retrain the CNN vision stack on the auto-labeled dataset.

Reads data/dataset/labels.csv + images (built by build_image_dataset.py), then:

  - Trains the ResNet-18 detector on {cyclone, nocylone} patches.
  - Trains the CNN-ConvLSTM intensity model on cyclone patches (7 IMD categories).
  - Writes checkpoints in the exact format VisionChain._load() expects:
        {'model_state_dict': state_dict}

Usage:
    python backend/tools/train_cnn.py --epochs 12 [--batch 32] [--device cpu]
"""

import argparse
import csv
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET = os.path.join(ROOT, "data", "dataset")
LABELS = os.path.join(DATASET, "labels.csv")
CATS = ["Depression", "Deep Depression", "Cyclonic Storm", "Severe Cyclonic Storm",
        "Very Severe Cyclonic Storm", "Extremely Severe Cyclonic Storm", "Super Cyclonic Storm"]
CAT_TO_IDX = {c: i for i, c in enumerate(CATS)}


def _read_labels():
    rows = []
    with open(LABELS, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def _read_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    ir = img.astype(np.float32) / 255.0
    wv = cv2.GaussianBlur(img, (9, 9), 0).astype(np.float32) / 255.0
    return np.stack([ir, wv], axis=0)  # [2,128,128]


class PatchSet(Dataset):
    def __init__(self, soup, _unused, augment=True):
        self.items = soup
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        row, arr = self.items[i]
        x = torch.from_numpy(arr).float()
        y = torch.tensor(1 if row["label"] == "cyclone" else 0, dtype=torch.long)
        if self.augment:
            if random.random() < 0.5:
                x = torch.flip(x, dims=[2])
            if random.random() < 0.5:
                x = torch.flip(x, dims=[1])
        return x, y


class IntenseSet(Dataset):
    def __init__(self, rows):
        self.items = [(r, _read_img(os.path.join(DATASET, "cyclone", r["file"]))) for r in rows]
        self.augment = True

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        row, arr = self.items[i]
        x = torch.from_numpy(arr).float().unsqueeze(0)  # [1, C, H, W] -> [T, C, H, W]
        y = torch.tensor(CAT_TO_IDX.get(row["category"], 0), dtype=torch.long)
        if self.augment:
            if random.random() < 0.5:
                x = torch.flip(x, dims=[3])
        return x, y


def _ckpt(model_state):
    return {"model_state_dict": {k: v.cpu() for k, v in model_state.items()}}


def train_detector(epochs, batch, device):
    from ml.detection.detector import CycloneDetector

    rows = _read_labels()
    cyc = [r for r in rows if r["label"] == "cyclone"]
    nocyc = [r for r in rows if r["label"] == "nocylone"]
    if not cyc or not nocyc:
        raise SystemExit("dataset needs both cyclone and nocylone patches")

    soup = []
    for r in rows:
        img = _read_img(os.path.join(DATASET, r["label"], r["file"]))
        if img is not None:
            soup.append((r, img))
    ds = PatchSet(soup, "cyclone")
    n = len(ds)
    tr, va = random_split(ds, [int(0.85 * n), n - int(0.85 * n)],
                          generator=torch.Generator().manual_seed(0))
    dl = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=0)
    dv = DataLoader(va, batch_size=batch, num_workers=0)

    model = CycloneDetector(in_channels=2, num_classes=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    best_va, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
        model.eval()
        acc = 0.0; tot = 0
        with torch.no_grad():
            for x, y in dv:
                p = model(x.to(device)).argmax(1).cpu()
                acc += (p == y).sum().item(); tot += y.numel()
        acc /= max(1, tot)
        print(f"[detector] epoch {ep:02d}  loss={loss.item():.3f}  val_acc={acc:.3f}")
        if acc > best_va:
            best_va, best_state = acc, model.state_dict()

    out = os.path.join(ROOT, "models", "detection", "detector_best.pt")
    torch.save(_ckpt(best_state), out)
    print(f"detector -> {out}  (val_acc={best_va:.3f})")


def train_intensity(epochs, batch, device):
    from ml.intensity.intensity_model import CycloneIntensityModel

    rows = [r for r in _read_labels() if r["label"] == "cyclone" and r["category"] in CAT_TO_IDX]
    if len(rows) < 8:
        print(f"intensity: only {len(rows)} cyclone patches with category — skipping "
              "(detector checkpoint still written)")
        return
    ds = IntenseSet(rows)
    n = len(ds)
    tr, va = random_split(ds, [int(0.85 * n), n - int(0.85 * n)],
                          generator=torch.Generator().manual_seed(1))
    dl = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=0)
    dv = DataLoader(va, batch_size=batch, num_workers=0)

    model = CycloneIntensityModel(in_channels=2, num_classes=7).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    best_va, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
        model.eval()
        acc = 0.0; tot = 0
        with torch.no_grad():
            for x, y in dv:
                p = model(x.to(device)).argmax(1).cpu()
                acc += (p == y).sum().item(); tot += y.numel()
        acc /= max(1, tot)
        print(f"[intensity] epoch {ep:02d}  loss={loss.item():.3f}  val_acc={acc:.3f}")
        if acc > best_va:
            best_va, best_state = acc, model.state_dict()

    out = os.path.join(ROOT, "models", "intensity", "intensity_best.pt")
    torch.save(_ckpt(best_state), out)
    print(f"intensity -> {out}  (val_acc={best_va:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if not os.path.exists(LABELS):
        raise SystemExit("No data/dataset/labels.csv — run build_image_dataset.py first.")
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available()
                          else "cpu")
    print("device:", device)
    train_detector(args.epochs, args.batch, device)
    train_intensity(args.epochs, args.batch, device)
    print("Done. Restart the backend; vision will load the retrained checkpoints on startup.")


if __name__ == "__main__":
    main()