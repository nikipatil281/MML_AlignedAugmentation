# scripts/train_retrieval.py

import os
import sys
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Make project root importable
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.datasets.coco_retrieval import build_coco_retrieval_splits
from src.datasets.collate import coco_retrieval_collate_fn
from src.models.retrieval_model import ImageTextRetrievalModel


def contrastive_loss(
    img_embeds: torch.Tensor,
    txt_embeds: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Symmetric InfoNCE loss for image-text retrieval.
    """
    device = img_embeds.device
    batch_size = img_embeds.size(0)

    # [B, B] similarity matrix
    logits = img_embeds @ txt_embeds.t()  # cosine since both are normalized
    logits = logits / temperature

    labels = torch.arange(batch_size, device=device)

    loss_i2t = nn.functional.cross_entropy(logits, labels)
    loss_t2i = nn.functional.cross_entropy(logits.t(), labels)

    loss = (loss_i2t + loss_t2i) / 2.0
    return loss


def train_one_epoch(
    model: ImageTextRetrievalModel,
    tokenizer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch} [train]"):
        images = batch["image"].to(device)               # [B, 3, 224, 224]
        captions = batch["caption"]                      # list[str]

        # Tokenize captions
        enc = tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)          # [B, L]
        attention_mask = enc["attention_mask"].to(device)

        optimizer.zero_grad()

        img_embeds, txt_embeds = model(images, input_ids, attention_mask)
        loss = contrastive_loss(img_embeds, txt_embeds)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    return avg_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --- Paths ---
    images_root = "data/coco/images/val2017"
    captions_json = "data/coco/annotations/captions_val2017.json"
    instances_json = "data/coco/annotations/instances_val2017.json"

    # --- Build datasets & dataloaders ---
    splits = build_coco_retrieval_splits(
        images_root=images_root,
        captions_json_path=captions_json,
        instances_json_path=instances_json,
        num_train=4000,
        num_val=500,
        num_test=500,
        seed=42,
        num_categories=91,
    )

    train_loader = DataLoader(
        splits.train,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=coco_retrieval_collate_fn,
    )

    # --- Model & tokenizer ---
    model = ImageTextRetrievalModel(
        img_embed_dim=256,
        txt_embed_dim=256,
        txt_model_name="bert-base-uncased",
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    num_epochs = 2  # start small just to test everything runs

    for epoch in range(1, num_epochs + 1):
        avg_train_loss = train_one_epoch(
            model=model,
            tokenizer=tokenizer,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )
        print(f"Epoch {epoch}: avg train loss = {avg_train_loss:.4f}")

    # Save a checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/retrieval_baseline.pt")
    print("Saved model to checkpoints/retrieval_baseline.pt")


if __name__ == "__main__":
    main()

