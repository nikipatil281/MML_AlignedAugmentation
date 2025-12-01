# src/models/pairing.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairingNetwork(nn.Module):
    """
    Learns a semantic embedding for pairing samples.

    Input: concatenated [img_emb, txt_emb] of dim 2D
    Output: pairing embedding e of dim P (default = D)
    """

    def __init__(self, embed_dim: int = 256, pairing_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        in_dim = 2 * embed_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, pairing_dim),
        )

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_emb: [B, D]
            txt_emb: [B, D]
        Returns:
            e: [B, pairing_dim] (L2-normalized)
        """
        x = torch.cat([img_emb, txt_emb], dim=-1)  # [B, 2D]
        e = self.net(x)                            # [B, pairing_dim]
        e = F.normalize(e, p=2, dim=-1)
        return e

