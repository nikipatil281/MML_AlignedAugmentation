# scripts/train_retrieval_full_coco.py

import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import torch.nn as nn

# project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.datasets.coco_retrieval import build_coco_dataset
from src.datasets.collate import coco_retrieval_collate_fn
from src.models.retrieval_model import ImageTextRetrievalModel


def contrastive_loss(
    img_embeds: torch.Tensor,
    txt_embeds: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    device = img_embeds.device
    batch_size = img_embeds.size(0)

    logits = img_embeds @ txt_embeds.t()
    logits = logits / temperature

    labels = torch.arange(batch_size, device=device)

    loss_i2t = nn.functional.cross_entropy(logits, labels)
    loss_t2i = nn.functional.cross_entropy(logits.t(), labels)

    return 0.5 * (loss_i2t + loss_t2i)


def train_one_epoch(
    model,
    tokenizer,
    dataloader,
    optimizer,
    device,
    epoch: int,
):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch} [train-full]"):
        images = batch["image"].to(device)
        captions = batch["caption"]

        enc = tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        optimizer.zero_grad()
        img_emb, txt_emb = model(images, input_ids, attention_mask)
        loss = contrastive_loss(img_emb, txt_emb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_images_root = "data/coco/images/train2017"
    train_captions_json = "data/coco/annotations/captions_train2017.json"
    train_instances_json = "data/coco/annotations/instances_train2017.json"

    train_dataset = build_coco_dataset(
        images_root=train_images_root,
        captions_json_path=train_captions_json,
        instances_json_path=train_instances_json,
        num_categories=91,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,        # you can adjust based on GPU memory
        shuffle=True,
        num_workers=4,
        collate_fn=coco_retrieval_collate_fn,
    )

    embed_dim = 256
    model = ImageTextRetrievalModel(
        img_embed_dim=embed_dim,
        txt_embed_dim=embed_dim,
        txt_model_name="bert-base-uncased",
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # CHANGE EPOCHS HERE FOR FULL TRAINING:
    num_epochs = 10  # e.g., 10–20 for full COCO

    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(model, tokenizer, train_loader, optimizer, device, epoch)
        print(f"Epoch {epoch}: avg train loss (full COCO) = {avg_loss:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/retrieval_baseline_full_coco.pt")
    print("Saved full COCO baseline model.")


if __name__ == "__main__":
    main()
