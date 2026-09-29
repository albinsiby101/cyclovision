"""
Grad-CAM Explainability Module for CycloVision
Generates visual attention heatmaps highlighting the exact convective cloud structures
(e.g., eye wall, central dense overcast, spiraling rainbands) that influenced the AI prediction.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Optional, Tuple

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        input_tensor: [1, C, H, W]
        Returns normalized 2D heatmap in range [0.0, 1.0]
        """
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        # Target score
        score = output[0, class_idx]
        score.backward()
        
        # Compute channel-wise mean of gradients (importance weights alpha)
        # self.gradients: [1, Channels, H', W']
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam) # Apply ReLU to keep features that have positive influence
        
        cam = cam.squeeze().cpu().numpy()
        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam = np.zeros_like(cam)
            
        # Resize CAM to input tensor dimensions
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        return cam_resized

    @staticmethod
    def overlay_heatmap(
        background_img: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.55,
        colormap: int = cv2.COLORMAP_JET
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Overlays heatmap on satellite image.
        background_img: [H, W] or [H, W, 3] in [0, 255] or [0, 1]
        heatmap: [H, W] in [0, 1]
        Returns (colored_heatmap, blended_image)
        """
        if background_img.max() <= 1.0:
            bg = (background_img * 255).astype(np.uint8)
        else:
            bg = background_img.astype(np.uint8)
            
        if bg.ndim == 2:
            bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2RGB)
        elif bg.shape[2] == 1:
            bg = cv2.cvtColor(bg[:, :, 0], cv2.COLOR_GRAY2RGB)
        elif bg.shape[2] == 2:
            # Multi-channel IR + WV: use IR channel for luminance base
            bg = cv2.cvtColor(bg[:, :, 0], cv2.COLOR_GRAY2RGB)
            
        h_uint8 = np.uint8(255 * heatmap)
        colored_cam = cv2.applyColorMap(h_uint8, colormap)
        colored_cam = cv2.cvtColor(colored_cam, cv2.COLOR_BGR2RGB)
        
        blended = cv2.addWeighted(bg, 1.0 - alpha, colored_cam, alpha, 0)
        return colored_cam, blended