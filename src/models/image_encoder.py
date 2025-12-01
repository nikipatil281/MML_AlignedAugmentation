# src/models/image_encoder.py

import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256, pretrained: bool = True):
        super().__init__()

        # Load ResNet-50 backbone
        resnet = models.resnet50(pretrained=pretrained)

        # Remove the classifier (fc layer)
        modules = list(resnet.children())[:-1]  # everything up to global avgpool
        self.backbone = nn.Sequential(*modules)  # outputs [B, 2048, 1, 1]

        # Projection to common embedding space
        self.proj = nn.Linear(2048, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 3, H, W]
        Returns:
            img_embeds: [B, embed_dim] (L2-normalized)
        """
        x = self.backbone(images)         # [B, 2048, 1, 1]
        x = x.squeeze(-1).squeeze(-1)     # [B, 2048]
        x = self.proj(x)                  # [B, embed_dim]
        x = nn.functional.normalize(x, p=2, dim=-1)
        return x

