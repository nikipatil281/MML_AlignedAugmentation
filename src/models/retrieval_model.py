# src/models/retrieval_model.py

import torch
import torch.nn as nn

from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder


class ImageTextRetrievalModel(nn.Module):
    def __init__(
        self,
        img_embed_dim: int = 256,
        txt_embed_dim: int = 256,
        txt_model_name: str = "bert-base-uncased",
    ):
        super().__init__()

        assert img_embed_dim == txt_embed_dim, "Embedding dims must match for cosine similarity."

        self.image_encoder = ImageEncoder(embed_dim=img_embed_dim)
        self.text_encoder = TextEncoder(model_name=txt_model_name, embed_dim=txt_embed_dim)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """
        Returns:
            img_embeds: [B, D]
            txt_embeds: [B, D]
        """
        img_embeds = self.image_encoder(images)
        txt_embeds = self.text_encoder(input_ids, attention_mask)
        return img_embeds, txt_embeds

