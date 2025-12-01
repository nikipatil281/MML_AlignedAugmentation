# scripts/train_retrieval_tokenmix_align.py

import os
import sys
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.datasets.coco_retrieval import build_coco_retrieval_splits
from src.datasets.collate import coco_retrieval_collate_fn
from src.models.retrieval_tokens_model import ImageTextRetrievalTokensModel
from src.models.pairing import PairingNetwork
from src.models.align_augmentor import AlignmentAwareAugmentor


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

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)

    return 0.5 * (loss_i2t + loss_t2i)


def compute_label_overlap_matrix(category_multi_hot: torch.Tensor) -> torch.Tensor:
    inter = category_multi_hot @ category_multi_hot.t()
    return (inter > 0).float()


def pairing_margin_ranking_loss(
    pairing_emb: torch.Tensor,
    category_multi_hot: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    device = pairing_emb.device
    B = pairing_emb.size(0)

    sim_matrix = pairing_emb @ pairing_emb.t()
    overlap = compute_label_overlap_matrix(category_multi_hot)

    losses: List[torch.Tensor] = []
    for i in range(B):
        pos_mask = (overlap[i] > 0).clone()
        pos_mask[i] = False

        neg_mask = (overlap[i] == 0).clone()
        neg_mask[i] = False

        pos_indices = pos_mask.nonzero(as_tuple=False).squeeze(-1)
        neg_indices = neg_mask.nonzero(as_tuple=False).squeeze(-1)

        if pos_indices.numel() == 0 or neg_indices.numel() == 0:
            continue

        j_pos = pos_indices[torch.randint(len(pos_indices), (1,), device=device)]
        j_neg = neg_indices[torch.randint(len(neg_indices), (1,), device=device)]

        sim_pos = sim_matrix[i, j_pos]
        sim_neg = sim_matrix[i, j_neg]

        losses.append(F.relu(margin + sim_neg - sim_pos))

    if len(losses) == 0:
        return torch.tensor(0.0, device=device)

    return torch.mean(torch.stack(losses))


def choose_mixing_partners(pairing_emb: torch.Tensor, top_k: int = 5) -> torch.Tensor:
    device = pairing_emb.device
    B = pairing_emb.size(0)

    sim = pairing_emb @ pairing_emb.t()  # [B,B]
    sim = sim - torch.eye(B, device=device) * 1e9

    k = min(top_k, B - 1)
    topk_indices = sim.topk(k=k, dim=-1).indices

    rand_idx = torch.randint(0, k, (B,), device=device)
    partner_indices = topk_indices[torch.arange(B, device=device), rand_idx]

    return partner_indices


def compute_modality_mix_weights(pairing_emb: torch.Tensor, partner_indices: torch.Tensor):
    device = pairing_emb.device
    B = pairing_emb.size(0)

    partner_emb = pairing_emb[partner_indices]
    sim_vals = torch.sum(pairing_emb * partner_emb, dim=-1)  # in [-1,1]

    lambda_img = 0.5 + 0.25 * sim_vals
    lambda_img = torch.clamp(lambda_img, 0.25, 0.75)
    lambda_txt = 1.0 - lambda_img

    return lambda_img.view(B, 1, 1), lambda_txt.view(B, 1, 1)


def compute_alignment_features(
    img_global: torch.Tensor,
    txt_global: torch.Tensor,
    img_mix_global: torch.Tensor,
    txt_mix_global: torch.Tensor,
    category_multi_hot: torch.Tensor,
) -> torch.Tensor:
    """
    Returns alignment features a: [B, 3]
    """
    # cosines
    s_raw = torch.sum(img_global * txt_global, dim=-1)          # [B]
    s_mix = torch.sum(img_mix_global * txt_mix_global, dim=-1)  # [B]

    # label density
    K = category_multi_hot.size(1)
    label_counts = category_multi_hot.sum(dim=-1) / float(K)    # [B]

    a = torch.stack([s_raw, s_mix, label_counts], dim=-1)       # [B,3]
    return a


def train_one_epoch_tokenmix_align(
    model: ImageTextRetrievalTokensModel,
    augmentor: AlignmentAwareAugmentor,
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

    for batch in tqdm(dataloader, desc=f"Epoch {epoch} [train TokenMix+Align]"):
        images = batch["image"].to(device)
        captions = batch["caption"]
        category_multi_hot = batch["category_multi_hot"].to(device)

        enc = tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # ---- Encoders ----
        img_tokens, txt_tokens, img_global, txt_global = model(images, input_ids, attention_mask)
        B, Np, D = img_tokens.shape
        _, L, _ = txt_tokens.shape

        # ---- Pairing from globals ----
        pairing_emb = pairing_net(img_global.detach(), txt_global.detach())

        # 1) Update pairing_net
        L_pair = pairing_margin_ranking_loss(pairing_emb, category_multi_hot, margin=0.2)
        optim_pairing.zero_grad()
        L_pair.backward(retain_graph=True)
        optim_pairing.step()

        # 2) Semantic partners + λs
        partner_indices = choose_mixing_partners(pairing_emb.detach(), top_k=5)
        lambda_img, lambda_txt = compute_modality_mix_weights(pairing_emb.detach(), partner_indices)

        # 3) Token & patch mixing
        img_tokens_partner = img_tokens[partner_indices]  # [B, Np, D]
        txt_tokens_partner = txt_tokens[partner_indices]  # [B, L, D]

        img_tokens_mix = lambda_img * img_tokens + (1.0 - lambda_img) * img_tokens_partner
        txt_tokens_mix = lambda_txt * txt_tokens + (1.0 - lambda_txt) * txt_tokens_partner

        img_tokens_mix = F.normalize(img_tokens_mix, p=2, dim=-1)
        txt_tokens_mix = F.normalize(txt_tokens_mix, p=2, dim=-1)

        img_mix_global = img_tokens_mix.mean(dim=1)          # [B, D]
        txt_mix_global = txt_tokens_mix[:, 0, :]             # CLS
        img_mix_global = F.normalize(img_mix_global, p=2, dim=-1)
        txt_mix_global = F.normalize(txt_mix_global, p=2, dim=-1)

        # 4) Alignment features
        a = compute_alignment_features(
            img_global=img_global.detach(),
            txt_global=txt_global.detach(),
            img_mix_global=img_mix_global.detach(),
            txt_mix_global=txt_mix_global.detach(),
            category_multi_hot=category_multi_hot.detach(),
        )  # [B,3]

        # 5) Update augmentor on (r_mix, a)
        r_mix = torch.cat([img_mix_global, txt_mix_global], dim=-1)  # [B, 2D]
        r_mix_det = r_mix.detach()
        a_det = a.detach()

        r_aug_for_aug = augmentor(r_mix_det, a_det)

        img_aug_for_aug = r_aug_for_aug[:, :D]
        txt_aug_for_aug = r_aug_for_aug[:, D:]
        img_aug_for_aug = F.normalize(img_aug_for_aug, p=2, dim=-1)
        txt_aug_for_aug = F.normalize(txt_aug_for_aug, p=2, dim=-1)

        L_aug_only = contrastive_loss(img_aug_for_aug, txt_aug_for_aug)
        reg = torch.mean((r_aug_for_aug - r_mix_det) ** 2)

        L_G = -L_aug_only + lambda_reg_for_aug * reg

        optim_aug.zero_grad()
        L_G.backward()
        optim_aug.step()

        # 6) Update retrieval model with alignment-aware aug
        with torch.no_grad():
            r_aug_for_model = augmentor(r_mix.detach(), a.detach())
        img_aug = r_aug_for_model[:, :D]
        txt_aug = r_aug_for_model[:, D:]
        img_aug = F.normalize(img_aug, p=2, dim=-1)
        txt_aug = F.normalize(txt_aug, p=2, dim=-1)

        L_orig = contrastive_loss(img_global, txt_global)
        L_aug = contrastive_loss(img_aug, txt_aug)
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
    model = ImageTextRetrievalTokensModel(
        img_embed_dim=embed_dim,
        txt_embed_dim=embed_dim,
        txt_model_name="bert-base-uncased",
    ).to(device)

    augmentor = AlignmentAwareAugmentor(embed_dim=embed_dim, hidden_dim=512, align_dim=3).to(device)
    pairing_net = PairingNetwork(embed_dim=embed_dim, pairing_dim=embed_dim, hidden_dim=512).to(device)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    optim_model = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    optim_aug = torch.optim.AdamW(augmentor.parameters(), lr=1e-4, weight_decay=1e-4)
    optim_pairing = torch.optim.AdamW(pairing_net.parameters(), lr=1e-4, weight_decay=1e-4)

    num_epochs = 2

    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch_tokenmix_align(
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
        print(f"Epoch {epoch}: avg train loss (TokenMix+Align) = {avg_loss:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/retrieval_tokenmix_align_model.pt")
    torch.save(augmentor.state_dict(), "checkpoints/retrieval_tokenmix_align_augmentor.pt")
    torch.save(pairing_net.state_dict(), "checkpoints/retrieval_tokenmix_align_pairing.pt")
    print("Saved TokenMix+Align model, augmentor, and pairing network.")


if __name__ == "__main__":
    main()

