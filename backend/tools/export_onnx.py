"""
Convert the CycloVision PyTorch CNNs to ONNX (CPU) so the cloud runtime can
use onnxruntime and never import torch.

Usage (dev machine, torch installed):
  python backend/tools/export_onnx.py

Writes:
  models/onnx/detector.onnx   input [1,2,128,128]            -> logits [1,2]
  models/onnx/intensity.onnx  input [1,6,2,128,128]          -> logits [1,7]
  models/onnx/ri.onnx         inputs seq [1,6,2,128,128],
                                     tabular [1,4]           -> logit [1,1]
"""

import os

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(ROOT, "models")
ONNX_DIR = os.path.join(MODELS_DIR, "onnx")


def _ckpt(name: str):
    return os.path.join(MODELS_DIR, name)


def _save(module, args, out_path: str, input_names, output_names):
    os.makedirs(ONNX_DIR, exist_ok=True)
    torch.onnx.export(
        module, args, out_path,
        input_names=input_names, output_names=output_names,
        opset_version=17, do_constant_folding=True,
    )
    print(f"[OK] {out_path} ({os.path.getsize(out_path) // 1024} KB)")


def main() -> None:
    from ml.detection.detector import CycloneDetector
    from ml.intensity.intensity_model import CycloneIntensityModel
    from ml.rapid_intensification.ri_model import RapidIntensificationModel

    torch.manual_seed(0)

    # --- detector ---
    det = CycloneDetector(in_channels=2, num_classes=2)
    det.load_state_dict(torch.load(_ckpt("detection/detector_best.pt"),
                                   map_location="cpu")["model_state_dict"])
    det.eval()
    _save(det, torch.randn(1, 2, 128, 128),
          os.path.join(ONNX_DIR, "detector.onnx"),
          ["input"], ["logits"])

    # --- intensity (CNN + ConvLSTM over a 6-frame sequence) ---
    itn = CycloneIntensityModel(in_channels=2, num_classes=7)
    itn.load_state_dict(torch.load(_ckpt("intensity/intensity_best.pt"),
                                   map_location="cpu")["model_state_dict"])
    itn.eval()
    _save(itn, torch.randn(1, 6, 2, 128, 128),
          os.path.join(ONNX_DIR, "intensity.onnx"),
          ["input"], ["logits"])

    # --- rapid intensification (seq + tabular) ---
    ri = RapidIntensificationModel(in_channels=2, spatial_features=48,
                                   convlstm_hidden=48, tabular_dim=4)
    ri.load_state_dict(torch.load(_ckpt("ri/ri_best.pt"),
                                  map_location="cpu")["model_state_dict"])
    ri.eval()
    _save(ri, (torch.randn(1, 6, 2, 128, 128), torch.randn(1, 4)),
          os.path.join(ONNX_DIR, "ri.onnx"),
          ["seq", "tabular"], ["logit"])

    print("All ONNX exports complete.")


if __name__ == "__main__":
    main()