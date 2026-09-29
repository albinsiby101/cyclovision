"""
Training script for Rapid Intensification (RI) Model (Model 3)
Uses Focal Loss to combat the severe class imbalance of rare RI events (~7-10%).
Saves checkpoint to models/ri/ri_best.pt.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from ml.rapid_intensification.ri_model import RapidIntensificationModel, FocalLoss
from scripts.generate_demo_data import generate_cyclonic_cloud_frame

class SyntheticRIDataset(Dataset):
    def __init__(self, size: int = 150, seq_len: int = 6):
        self.samples = []
        for i in range(size):
            # Imbalance: ~20% positive RI, 80% negative
            is_ri = (i % 5 == 0)
            target = 1.0 if is_ri else 0.0
            
            # If RI, cloud tops expand rapidly, scale spikes
            base_scale = 0.5 if is_ri else 0.4
            scale_step = 0.08 if is_ri else 0.01
            frames = []
            for t in range(seq_len):
                f = generate_cyclonic_cloud_frame(t, total_t=seq_len, intensity_scale=base_scale + t * scale_step, has_eye=True)
                frames.append(np.transpose(f, (2, 0, 1)))
            seq_np = np.stack(frames, axis=0) # [T, 2, H, W]
            
            # Tabular features: [current_wind_kt/150, 12h_trend/50, lat/30, pressure_norm]
            current_wind = 55.0 + (30.0 if is_ri else 0.0)
            trend = 25.0 if is_ri else 5.0
            tab = np.array([current_wind / 150.0, trend / 50.0, 15.0 / 30.0, 970.0 / 1013.0], dtype=np.float32)
            
            self.samples.append((
                torch.from_numpy(seq_np).float(),
                torch.from_numpy(tab).float(),
                torch.tensor([target], dtype=torch.float32)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def train_ri():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training Rapid Intensification Model on Device: {device}")
    
    os.makedirs("models/ri", exist_ok=True)
    dataset = SyntheticRIDataset(size=150, seq_len=6)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
    
    model = RapidIntensificationModel(in_channels=2, spatial_features=48, convlstm_hidden=48, tabular_dim=4).to(device)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
    
    best_loss = 999.0
    for epoch in range(1, 11):
        model.train()
        total_loss = 0.0
        for seq, tab, y in train_loader:
            seq, tab, y = seq.to(device), tab.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(seq, tab)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq, tab, y in val_loader:
                seq, tab, y = seq.to(device), tab.to(device), y.to(device)
                logits = model(seq, tab)
                loss = criterion(logits, y)
                val_loss += loss.item() * len(y)
        val_loss /= val_size
        print(f"RI Epoch {epoch:02d} | Val Focal Loss: {val_loss:.4f}")
        
        if val_loss <= best_loss:
            best_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss
            }, "models/ri/ri_best.pt")
            
    print(f"[SUCCESS] Best RI Model saved with Val Loss: {best_loss:.4f}")

if __name__ == "__main__":
    train_ri()