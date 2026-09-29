"""
Evaluation and Metrics Engine for CycloVision
Generates Accuracy, Precision, Recall, F1, PR-AUC, and Confusion Matrices
for Cyclone Detection, Intensity Classification, and Rapid Intensification.
Saves JSON summaries and visualization figures to outputs/
"""

import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, average_precision_score
)

from ml.detection.detector import CycloneDetector
from ml.intensity.intensity_model import CycloneIntensityModel
from ml.rapid_intensification.ri_model import RapidIntensificationModel
from ml.training.train_detection import SyntheticSatelliteDataset
from ml.training.train_intensity import SyntheticIntensitySequenceDataset
from ml.training.train_ri import SyntheticRIDataset

def evaluate_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/predictions", exist_ok=True)
    results = {}
    
    # 1. Evaluate Detection
    det_ckpt = "models/detection/detector_best.pt"
    if os.path.exists(det_ckpt):
        model = CycloneDetector(in_channels=2, num_classes=2).to(device)
        ckpt = torch.load(det_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        
        test_ds = SyntheticSatelliteDataset(size=80)
        y_true, y_pred, y_probs = [], [], []
        with torch.no_grad():
            for x, y in test_ds:
                x = x.unsqueeze(0).to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred = np.argmax(probs)
                y_true.append(y)
                y_pred.append(pred)
                y_probs.append(probs[1])
                
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        auc = roc_auc_score(y_true, y_probs)
        cm = confusion_matrix(y_true, y_pred).tolist()
        
        results["detection"] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
            "confusion_matrix": cm
        }
        print("[EVAL] Detection Model:", results["detection"])
        
    # 2. Evaluate Intensity
    int_ckpt = "models/intensity/intensity_best.pt"
    if os.path.exists(int_ckpt):
        model = CycloneIntensityModel(in_channels=2, num_classes=7).to(device)
        ckpt = torch.load(int_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        
        test_ds = SyntheticIntensitySequenceDataset(size=70, seq_len=6)
        y_true, y_pred = [], []
        with torch.no_grad():
            for seq, y in test_ds:
                seq = seq.unsqueeze(0).to(device)
                logits = model(seq)
                pred = torch.argmax(logits, dim=1).item()
                y_true.append(y)
                y_pred.append(pred)
                
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_true, y_pred).tolist()
        results["intensity"] = {
            "accuracy": round(float(acc), 4),
            "macro_precision": round(float(p), 4),
            "macro_recall": round(float(r), 4),
            "macro_f1": round(float(f1), 4),
            "confusion_matrix": cm
        }
        print("[EVAL] Intensity Model:", results["intensity"])

    # 3. Evaluate RI
    ri_ckpt = "models/ri/ri_best.pt"
    if os.path.exists(ri_ckpt):
        model = RapidIntensificationModel(in_channels=2, spatial_features=48, convlstm_hidden=48, tabular_dim=4).to(device)
        ckpt = torch.load(ri_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        
        test_ds = SyntheticRIDataset(size=60, seq_len=6)
        y_true, y_pred, y_scores = [], [], []
        with torch.no_grad():
            for seq, tab, y in test_ds:
                seq, tab = seq.unsqueeze(0).to(device), tab.unsqueeze(0).to(device)
                logit = model(seq, tab)
                prob = torch.sigmoid(logit).item()
                y_true.append(int(y.item()))
                y_scores.append(prob)
                y_pred.append(1 if prob >= 0.5 else 0)
                
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        pr_auc = average_precision_score(y_true, y_scores)
        cm = confusion_matrix(y_true, y_pred).tolist()
        results["rapid_intensification"] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f1), 4),
            "pr_auc": round(float(pr_auc), 4),
            "confusion_matrix": cm
        }
        print("[EVAL] RI Model:", results["rapid_intensification"])
        
    with open("outputs/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[SUCCESS] All evaluation results saved to outputs/evaluation_results.json")

if __name__ == "__main__":
    evaluate_all()