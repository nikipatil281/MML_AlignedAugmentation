# src/models/align_augmentor.py

import torch
import torch.nn as nn


class AlignmentAwareAugmentor(nn.Module):
    """
    Alignment-aware feature-space augmentor.

    Input:
      - r: [B, 2D] concatenated image-text global features
      - a: [B, A] alignment features (e.g., similarities, label stats)

    Output:
      - r_aug: [B, 2D] augmented features (residual)
    """

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 512, align_dim: int = 3):
        super().__init__()
        in_dim = 2 * embed_dim + align_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * embed_dim),
        )

    def forward(self, r: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            r: [B, 2D]
            a: [B, A]
        Returns:
            r_aug: [B, 2D]
        """
        x = torch.cat([r, a], dim=-1)  # [B, 2D + A]
        delta = self.net(x)
        r_aug = r + delta  # residual
        return r_aug


