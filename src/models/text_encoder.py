# src/models/text_encoder.py

import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", embed_dim: int = 256):
        super().__init__()

        self.text_model = AutoModel.from_pretrained(model_name)
        hidden_size = self.text_model.config.hidden_size  # 768 for BERT-base

        self.proj = nn.Linear(hidden_size, embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, L]
            attention_mask: [B, L]
        Returns:
            txt_embeds: [B, embed_dim] (L2-normalized)
        """
        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # CLS token representation
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [B, hidden_size]
        x = self.proj(cls_emb)                        # [B, embed_dim]
        x = nn.functional.normalize(x, p=2, dim=-1)
        return x

