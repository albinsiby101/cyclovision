"""
Training script for Cyclone Detection (Model 1)
Trains ResNet18 on multi-spectral satellite imagery (IR + Water Vapour).
Saves checkpoint to models/detection/detector_best.pt and logs metrics.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import yaml
from ml.detection.detector import CycloneDetector
from scripts.generate_demo_data import generate_cyclonic_cloud_frame, generate_non_cyclone_frame

class SyntheticSatelliteDataset(Dataset):
    def __init__(self, size: int = 200):
        self.samples = []
        for i in range(size):
            is_cyclone = (i % 2 == 0)
            if is_cyclone:
                frame = generate_cyclonic_cloud_frame(i % 6, intensity_scale=np.random.uniform(0.3, 1.2))
                label = 1
            else:
                frame = generate_non_cyclone_frame(i % 6)
                label = 0
            # [H, W, 2] -> [2, H, W]
            tensor = torch.from_numpy(np.transpose(frame, (2, 0, 1))).float()
            self.samples.append((tensor, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def train_detection():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training Detection Model on Device: {device}")
    
    os.makedirs("models/detection", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    
    dataset = SyntheticSatelliteDataset(size=240)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
    
    model = CycloneDetector(in_channels=2, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
    
    best_acc = 0.0
    for epoch in range(1, 11):
        model.train()
        total_loss, correct = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            
        train_acc = correct / train_size
        
        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_correct += (logits.argmax(dim=1) == y).sum().item()
        val_acc = val_correct / val_size
        print(f"Epoch {epoch:02d} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")
        
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "in_channels": 2,
                "num_classes": 2
            }, "models/detection/detector_best.pt")
            
    print(f"[SUCCESS] Best Detection Model saved with Val Acc: {best_acc:.3f}")

if __name__ == "__main__":
    train_detection()