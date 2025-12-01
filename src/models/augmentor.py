# src/models/augmentor.py

import torch
import torch.nn as nn


class FeatureAugmentor(nn.Module):
    """
    LeMDA-style feature-space augmentor.

    Takes concatenated [img_emb, txt_emb] of dim 2D and outputs
    a residual-perturbed version of same dim.
    """

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        in_dim = 2 * embed_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Args:
            r: [B, 2D] concatenated embeddings
        Returns:
            r_aug: [B, 2D] augmented features
        """
        delta = self.net(r)
        r_aug = r + delta  # residual perturbation
        return r_aug

