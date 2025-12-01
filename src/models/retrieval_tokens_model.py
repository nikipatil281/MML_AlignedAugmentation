# src/models/retrieval_tokens_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoModel
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ImageEncoderTokens(nn.Module):
    """
    ResNet-based image encoder that returns patch tokens + global embedding.
    """

    def __init__(self, embed_dim: int = 256, pretrained: bool = True):
        super().__init__()

        resnet = models.resnet50(pretrained=pretrained)
        # Keep everything up to the last conv block (exclude avgpool & fc)
        modules = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)  # [B, 2048, Hf, Wf]

        self.token_proj = nn.Linear(2048, embed_dim)

    def forward(self, images: torch.Tensor):
        """
        Args:
            images: [B, 3, H, W]
        Returns:
            img_tokens: [B, N_patches, D]
            img_global: [B, D]
        """
        x = self.backbone(images)                # [B, 2048, Hf, Wf]
        B, C, Hf, Wf = x.shape
        x = x.view(B, C, Hf * Wf)                # [B, C, N]
        x = x.permute(0, 2, 1)                   # [B, N, C]

        tokens = self.token_proj(x)              # [B, N, D]
        tokens = F.normalize(tokens, p=2, dim=-1)

        global_emb = tokens.mean(dim=1)          # [B, D]
        global_emb = F.normalize(global_emb, p=2, dim=-1)

        return tokens, global_emb


class TextEncoderTokens(nn.Module):
    """
    BERT-based text encoder that returns token embeddings + global embedding.
    """

    def __init__(self, model_name: str = "bert-base-uncased", embed_dim: int = 256):
        super().__init__()

        self.text_model = AutoModel.from_pretrained(model_name)
        hidden_size = self.text_model.config.hidden_size

        self.token_proj = nn.Linear(hidden_size, embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Args:
            input_ids: [B, L]
            attention_mask: [B, L]
        Returns:
            txt_tokens: [B, L, D]
            txt_global: [B, D]
        """
        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        token_states = outputs.last_hidden_state      # [B, L, H]

        tokens = self.token_proj(token_states)        # [B, L, D]
        tokens = F.normalize(tokens, p=2, dim=-1)

        # Use CLS token (pos 0) as global, or mean over non-padded tokens
        cls_emb = tokens[:, 0, :]                    # [B, D]
        global_emb = F.normalize(cls_emb, p=2, dim=-1)

        return tokens, global_emb


class ImageTextRetrievalTokensModel(nn.Module):
    """
    Retrieval model with token/patch-level features + global embeddings.
    """

    def __init__(
        self,
        img_embed_dim: int = 256,
        txt_embed_dim: int = 256,
        txt_model_name: str = "bert-base-uncased",
    ):
        super().__init__()

        assert img_embed_dim == txt_embed_dim
        self.embed_dim = img_embed_dim

        self.image_encoder = ImageEncoderTokens(embed_dim=img_embed_dim)
        self.text_encoder = TextEncoderTokens(model_name=txt_model_name, embed_dim=txt_embed_dim)

    def forward(self, images, input_ids, attention_mask):
        """
        Returns:
            img_tokens: [B, Np, D]
            txt_tokens: [B, L, D]
            img_global: [B, D]
            txt_global: [B, D]
        """
        img_tokens, img_global = self.image_encoder(images)
        txt_tokens, txt_global = self.text_encoder(input_ids, attention_mask)
        return img_tokens, txt_tokens, img_global, txt_global

