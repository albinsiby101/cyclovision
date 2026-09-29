"""
Training script for CNN + ConvLSTM Intensity Model (Model 2)
Models temporal sequences [B, T=6, C=2, H=128, W=128] to predict 7 IMD categories.
Saves checkpoint to models/intensity/intensity_best.pt.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from ml.intensity.intensity_model import CycloneIntensityModel
from scripts.generate_demo_data import generate_cyclonic_cloud_frame

class SyntheticIntensitySequenceDataset(Dataset):
    def __init__(self, size: int = 140, seq_len: int = 6):
        self.samples = []
        # 7 classes: 0: Depression ... 6: Super Cyclonic Storm
        for i in range(size):
            cat = i % 7
            intensity_scale = 0.2 + (cat / 6.0) * 1.0 # 0.2 to 1.2
            has_eye = cat >= 3
            frames = []
            for t in range(seq_len):
                # slight evolution over sequence
                f_scale = intensity_scale + (t * 0.02)
                f = generate_cyclonic_cloud_frame(t, total_t=seq_len, intensity_scale=f_scale, has_eye=has_eye)
                # [H, W, 2] -> [2, H, W]
                frames.append(np.transpose(f, (2, 0, 1)))
            seq_np = np.stack(frames, axis=0) # [T, 2, H, W]
            self.samples.append((torch.from_numpy(seq_np).float(), cat))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def train_intensity():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training Intensity Model (CNN+ConvLSTM) on Device: {device}")
    
    os.makedirs("models/intensity", exist_ok=True)
    dataset = SyntheticIntensitySequenceDataset(size=140, seq_len=6)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
    
    model = CycloneIntensityModel(in_channels=2, num_classes=7, spatial_features=64, convlstm_hidden=64).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
    
    best_acc = 0.0
    for epoch in range(1, 11):
        model.train()
        total_loss, correct = 0.0, 0
        for seq, y in train_loader:
            seq, y = seq.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(seq)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            
        train_acc = correct / train_size
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for seq, y in val_loader:
                seq, y = seq.to(device), y.to(device)
                logits = model(seq)
                val_correct += (logits.argmax(dim=1) == y).sum().item()
        val_acc = val_correct / val_size
        print(f"Intensity Epoch {epoch:02d} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")
        
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "num_classes": 7
            }, "models/intensity/intensity_best.pt")
            
    print(f"[SUCCESS] Best Intensity Model saved with Val Acc: {best_acc:.3f}")

if __name__ == "__main__":
    train_intensity()