# scripts/train_retrieval_lemdalike.py

import os
import sys

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
from src.models.augmentor import FeatureAugmentor


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

    logits = img_embeds @ txt_embeds.t()  # cosine since normalized
    logits = logits / temperature

    labels = torch.arange(batch_size, device=device)

    loss_i2t = nn.functional.cross_entropy(logits, labels)
    loss_t2i = nn.functional.cross_entropy(logits.t(), labels)

    return 0.5 * (loss_i2t + loss_t2i)


def train_one_epoch_lemdalike(
    model: ImageTextRetrievalModel,
    augmentor: FeatureAugmentor,
    tokenizer,
    dataloader: DataLoader,
    optim_model: torch.optim.Optimizer,
    optim_aug: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    lambda_aug_for_model: float = 1.0,
    lambda_reg_for_aug: float = 1e-3,
) -> float:
    """
    Train one epoch with LeMDA-style feature augmentation.

    - model (θ) is trained to minimize L_orig + λ * L_aug
    - augmentor (φ) is trained to maximize L_aug while keeping r_aug close to r
    """
    model.train()
    augmentor.train()

    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch} [train LeMDA-Lite]"):
        images = batch["image"].to(device)
        captions = batch["caption"]

        # ---- Tokenize captions ----
        enc = tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # ---- Forward through retrieval model ----
        img_emb, txt_emb = model(images, input_ids, attention_mask)  # [B, D], [B, D]
        B, D = img_emb.size()

        # Concatenated representation r = [img_emb, txt_emb]
        r = torch.cat([img_emb, txt_emb], dim=-1)  # [B, 2D]

        # ============================================================
        # 1) Update augmentor φ: make augmentations "hard" but close
        # ============================================================

        # Detach r so we don't change encoders while updating augmentor
        r_det = r.detach()
        r_aug_for_aug = augmentor(r_det)  # [B, 2D]

        img_emb_aug_for_aug = r_aug_for_aug[:, :D]
        txt_emb_aug_for_aug = r_aug_for_aug[:, D:]

        # Normalize augmented embeddings
        img_emb_aug_for_aug = nn.functional.normalize(img_emb_aug_for_aug, p=2, dim=-1)
        txt_emb_aug_for_aug = nn.functional.normalize(txt_emb_aug_for_aug, p=2, dim=-1)

        L_aug_only = contrastive_loss(img_emb_aug_for_aug, txt_emb_aug_for_aug)

        # Regularization: keep r_aug close to r_det
        reg = torch.mean((r_aug_for_aug - r_det) ** 2)

        # Augmentor wants to MAXIMIZE L_aug (hard examples), MINIMIZE reg
        L_G = -L_aug_only + lambda_reg_for_aug * reg

        optim_aug.zero_grad()
        L_G.backward()
        optim_aug.step()

        # ============================================================
        # 2) Update retrieval model θ: handle original + augmented
        # ============================================================

        # Get a "frozen" augmentation for model update
        with torch.no_grad():
            r_aug_for_model = augmentor(r.detach())  # no grad to augmentor here

        img_emb_aug = r_aug_for_model[:, :D]
        txt_emb_aug = r_aug_for_model[:, D:]
        img_emb_aug = nn.functional.normalize(img_emb_aug, p=2, dim=-1)
        txt_emb_aug = nn.functional.normalize(txt_emb_aug, p=2, dim=-1)

        L_orig = contrastive_loss(img_emb, txt_emb)
        L_aug = contrastive_loss(img_emb_aug, txt_emb_aug)

        L_theta = L_orig + lambda_aug_for_model * L_aug

        optim_model.zero_grad()
        L_theta.backward()
        optim_model.step()

        total_loss += L_theta.item()
        num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    return avg_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- Paths ----
    images_root = "data/coco/images/val2017"
    captions_json = "data/coco/annotations/captions_val2017.json"
    instances_json = "data/coco/annotations/instances_val2017.json"

    # ---- Data ----
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

    # ---- Model & Augmentor ----
    embed_dim = 256
    model = ImageTextRetrievalModel(
        img_embed_dim=embed_dim,
        txt_embed_dim=embed_dim,
        txt_model_name="bert-base-uncased",
    ).to(device)

    augmentor = FeatureAugmentor(embed_dim=embed_dim, hidden_dim=512).to(device)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # ---- Optimizers ----
    optim_model = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )
    optim_aug = torch.optim.AdamW(
        augmentor.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    num_epochs = 2  # start small to test; later bump

    for epoch in range(1, num_epochs + 1):
        avg_train_loss = train_one_epoch_lemdalike(
            model=model,
            augmentor=augmentor,
            tokenizer=tokenizer,
            dataloader=train_loader,
            optim_model=optim_model,
            optim_aug=optim_aug,
            device=device,
            epoch=epoch,
            lambda_aug_for_model=1.0,
            lambda_reg_for_aug=1e-3,
        )
        print(f"Epoch {epoch}: avg train loss (LeMDA-Lite) = {avg_train_loss:.4f}")

    # Save checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/retrieval_lemdalike_model.pt")
    torch.save(augmentor.state_dict(), "checkpoints/retrieval_lemdalike_augmentor.pt")
    print("Saved LeMDA-Lite model and augmentor.")


if __name__ == "__main__":
    main()

