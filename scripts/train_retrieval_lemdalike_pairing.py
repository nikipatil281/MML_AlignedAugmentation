# scripts/train_retrieval_lemdalike_pairing.py

import os
import sys
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import torch.nn.functional as F

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
from src.models.pairing import PairingNetwork


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


def compute_label_overlap_matrix(category_multi_hot: torch.Tensor) -> torch.Tensor:
    """
    Compute binary overlap matrix O[i, j] = 1 if sample i and j share at least one category.
    category_multi_hot: [B, K] 0/1 tensor
    """
    # [B, B]: intersection counts
    inter = category_multi_hot @ category_multi_hot.t()
    overlap = (inter > 0).float()
    return overlap


def pairing_margin_ranking_loss(
    pairing_emb: torch.Tensor,
    category_multi_hot: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """
    Margin ranking loss for pairing.

    pairing_emb: [B, P] normalized embeddings
    category_multi_hot: [B, K] 0/1
    """
    device = pairing_emb.device
    B = pairing_emb.size(0)

    # Cosine similarity matrix [B, B]
    sim_matrix = pairing_emb @ pairing_emb.t()

    # Overlap matrix [B, B]
    overlap = compute_label_overlap_matrix(category_multi_hot)  # 1 if share a label

    losses: List[torch.Tensor] = []

    for i in range(B):
        # Exclude self
        pos_mask = (overlap[i] > 0).clone()
        pos_mask[i] = False

        neg_mask = (overlap[i] == 0).clone()
        neg_mask[i] = False

        pos_indices = pos_mask.nonzero(as_tuple=False).squeeze(-1)
        neg_indices = neg_mask.nonzero(as_tuple=False).squeeze(-1)

        if pos_indices.numel() == 0 or neg_indices.numel() == 0:
            # No valid pos or neg in batch for this anchor
            continue

        # Randomly sample one positive and one negative
        j_pos = pos_indices[torch.randint(len(pos_indices), (1,), device=device)]
        j_neg = neg_indices[torch.randint(len(neg_indices), (1,), device=device)]

        sim_pos = sim_matrix[i, j_pos]
        sim_neg = sim_matrix[i, j_neg]

        # Margin ranking: want sim_pos > sim_neg + margin
        loss_i = F.relu(margin + sim_neg - sim_pos)
        losses.append(loss_i)

    if len(losses) == 0:
        return torch.tensor(0.0, device=device)

    return torch.mean(torch.stack(losses))


def train_one_epoch_lemdalike_pairing(
    model: ImageTextRetrievalModel,
    augmentor: FeatureAugmentor,
    pairing_net: PairingNetwork,
    tokenizer,
    dataloader: DataLoader,
    optim_model: torch.optim.Optimizer,
    optim_aug: torch.optim.Optimizer,
    optim_pairing: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    lambda_aug_for_model: float = 1.0,
    lambda_reg_for_aug: float = 1e-3,
    lambda_pair_for_model: float = 0.1,
) -> float:
    model.train()
    augmentor.train()
    pairing_net.train()

    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch} [train LeMDA+Pairing]"):
        images = batch["image"].to(device)
        captions = batch["caption"]
        category_multi_hot = batch["category_multi_hot"].to(device)  # [B, K]

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
        # 1) Update augmentor φ
        # ============================================================
        r_det = r.detach()
        r_aug_for_aug = augmentor(r_det)

        img_emb_aug_for_aug = r_aug_for_aug[:, :D]
        txt_emb_aug_for_aug = r_aug_for_aug[:, D:]
        img_emb_aug_for_aug = nn.functional.normalize(img_emb_aug_for_aug, p=2, dim=-1)
        txt_emb_aug_for_aug = nn.functional.normalize(txt_emb_aug_for_aug, p=2, dim=-1)

        L_aug_only = contrastive_loss(img_emb_aug_for_aug, txt_emb_aug_for_aug)
        reg = torch.mean((r_aug_for_aug - r_det) ** 2)

        L_G = -L_aug_only + lambda_reg_for_aug * reg

        optim_aug.zero_grad()
        L_G.backward()
        optim_aug.step()

        # ============================================================
        # 2) Pairing network ψ: learn semantic similarity
        # ============================================================
        pairing_emb = pairing_net(img_emb.detach(), txt_emb.detach())  # [B, P]
        L_pair = pairing_margin_ranking_loss(pairing_emb, category_multi_hot, margin=0.2)

        optim_pairing.zero_grad()
        L_pair.backward()
        optim_pairing.step()

        # ============================================================
        # 3) Update retrieval model θ: L_orig + λ_aug * L_aug + λ_pair * L_pair
        # ============================================================
        with torch.no_grad():
            r_aug_for_model = augmentor(r.detach())

        img_emb_aug = r_aug_for_model[:, :D]
        txt_emb_aug = r_aug_for_model[:, D:]
        img_emb_aug = nn.functional.normalize(img_emb_aug, p=2, dim=-1)
        txt_emb_aug = nn.functional.normalize(txt_emb_aug, p=2, dim=-1)

        L_orig = contrastive_loss(img_emb, txt_emb)
        L_aug = contrastive_loss(img_emb_aug, txt_emb_aug)

        # reuse L_pair but detached from pairing_net graph
        L_pair_for_model = L_pair.detach()

        L_theta = L_orig + lambda_aug_for_model * L_aug + lambda_pair_for_model * L_pair_for_model

        optim_model.zero_grad()
        L_theta.backward()
        optim_model.step()

        total_loss += L_theta.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    images_root = "data/coco/images/val2017"
    captions_json = "data/coco/annotations/captions_val2017.json"
    instances_json = "data/coco/annotations/instances_val2017.json"

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

    embed_dim = 256
    model = ImageTextRetrievalModel(
        img_embed_dim=embed_dim,
        txt_embed_dim=embed_dim,
        txt_model_name="bert-base-uncased",
    ).to(device)

    augmentor = FeatureAugmentor(embed_dim=embed_dim, hidden_dim=512).to(device)
    pairing_net = PairingNetwork(embed_dim=embed_dim, pairing_dim=embed_dim, hidden_dim=512).to(device)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    optim_model = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    optim_aug = torch.optim.AdamW(augmentor.parameters(), lr=1e-4, weight_decay=1e-4)
    optim_pairing = torch.optim.AdamW(pairing_net.parameters(), lr=1e-4, weight_decay=1e-4)

    num_epochs = 2

    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch_lemdalike_pairing(
            model=model,
            augmentor=augmentor,
            pairing_net=pairing_net,
            tokenizer=tokenizer,
            dataloader=train_loader,
            optim_model=optim_model,
            optim_aug=optim_aug,
            optim_pairing=optim_pairing,
            device=device,
            epoch=epoch,
            lambda_aug_for_model=1.0,
            lambda_reg_for_aug=1e-3,
            lambda_pair_for_model=0.1,
        )
        print(f"Epoch {epoch}: avg train loss (LeMDA+Pairing) = {avg_loss:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/retrieval_lemdalike_pairing_model.pt")
    torch.save(augmentor.state_dict(), "checkpoints/retrieval_lemdalike_pairing_augmentor.pt")
    torch.save(pairing_net.state_dict(), "checkpoints/retrieval_lemdalike_pairing_net.pt")
    print("Saved LeMDA+Pairing model, augmentor, and pairing network.")


if __name__ == "__main__":
    main()

