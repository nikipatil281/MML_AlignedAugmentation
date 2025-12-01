import os
import sys
from typing import List

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ---- project root ----
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.datasets.coco_retrieval import build_coco_dataset, CocoRetrievalDataset
from src.datasets.collate import coco_retrieval_collate_fn
from src.models.retrieval_tokens_model import ImageTextRetrievalTokensModel
from src.models.pairing import PairingNetwork


# ----------------- helpers from training logic ----------------- #

def build_single_caption_subset(dataset: CocoRetrievalDataset) -> CocoRetrievalDataset:
    """
    Restrict to a single caption per image (index 0) so that
    we have 1:1 images <-> captions.
    """
    subset = dataset
    subset._pairs = [(img_id, 0) for img_id in subset.image_ids]
    return subset


def choose_mixing_partners(pairing_emb: torch.Tensor, top_k: int = 5) -> torch.Tensor:
    """
    Same partner selection logic as training:
      - cosine similarity over pairing_emb
      - ignore self
      - pick a random neighbor among top_k most similar
    """
    device = pairing_emb.device
    B = pairing_emb.size(0)

    sim = pairing_emb @ pairing_emb.t()  # [B, B]
    sim = sim - torch.eye(B, device=device) * 1e9  # avoid self

    k = min(top_k, B - 1)
    topk_indices = sim.topk(k=k, dim=-1).indices  # [B, k]

    rand_idx = torch.randint(0, k, (B,), device=device)
    partner_indices = topk_indices[torch.arange(B, device=device), rand_idx]  # [B]

    return partner_indices


def compute_modality_mix_weights(pairing_emb: torch.Tensor, partner_indices: torch.Tensor):
    """
    Same λ_img, λ_txt logic as training:
      sim ∈ [-1,1] -> λ_img ∈ [0.25, 0.75], λ_txt = 1 - λ_img
    """
    device = pairing_emb.device
    B = pairing_emb.size(0)

    partner_emb = pairing_emb[partner_indices]                     # [B, P]
    sim_vals = torch.sum(pairing_emb * partner_emb, dim=-1)        # [B]

    lambda_img = 0.5 + 0.25 * sim_vals
    lambda_img = torch.clamp(lambda_img, 0.25, 0.75)
    lambda_txt = 1.0 - lambda_img

    return lambda_img, lambda_txt


# ----------------- image loading / blending ----------------- #

def load_raw_image(dataset: CocoRetrievalDataset, image_id: int, size=(256, 256)) -> Image.Image:
    rel_path = dataset.image_id_to_filename[image_id]
    img_path = os.path.join(dataset.images_root, rel_path)
    img = Image.open(img_path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    return img


def blend_images(img_a: Image.Image, img_b: Image.Image, alpha: float) -> Image.Image:
    """
    Simple pixel-space blend:
      mixed = alpha * A + (1 - alpha) * B
    """
    arr_a = np.asarray(img_a).astype(np.float32) / 255.0
    arr_b = np.asarray(img_b).astype(np.float32) / 255.0

    mixed = alpha * arr_a + (1.0 - alpha) * arr_b
    mixed = np.clip(mixed, 0.0, 1.0)
    mixed = (mixed * 255).astype(np.uint8)
    return Image.fromarray(mixed)


# ----------------- main visualization script ----------------- #

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --- load val2017 dataset ---
    images_root = "data/coco/images/val2017"
    captions_json = "data/coco/annotations/captions_val2017.json"
    instances_json = "data/coco/annotations/instances_val2017.json"

    dataset = build_coco_dataset(
        images_root=images_root,
        captions_json_path=captions_json,
        instances_json_path=instances_json,
        num_categories=91,
    )
    dataset = build_single_caption_subset(dataset)

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,   # order is consistent with dataset.image_ids
        num_workers=4,
        collate_fn=coco_retrieval_collate_fn,
    )

    # --- load model + pairing network from full TokenMix+Align training ---
    embed_dim = 256
    model = ImageTextRetrievalTokensModel(
        img_embed_dim=embed_dim,
        txt_embed_dim=embed_dim,
        txt_model_name="bert-base-uncased",
    ).to(device)

    pairing_net = PairingNetwork(
        embed_dim=embed_dim, pairing_dim=embed_dim, hidden_dim=512
    ).to(device)

    model_ckpt = "checkpoints/retrieval_tokenmix_align_full_coco_model.pt"
    pairing_ckpt = "checkpoints/retrieval_tokenmix_align_full_coco_pairing.pt"

    print(f"Loading model from {model_ckpt}")
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    print(f"Loading pairing net from {pairing_ckpt}")
    pairing_net.load_state_dict(torch.load(pairing_ckpt, map_location=device))

    model.eval()
    pairing_net.eval()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # --- collect globals & pairing embeddings for whole val set ---
    all_img_globals: List[torch.Tensor] = []
    all_txt_globals: List[torch.Tensor] = []
    all_pair_embs: List[torch.Tensor] = []
    all_captions: List[str] = []
    all_image_ids: List[int] = list(dataset.image_ids)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding for TokenMix viz"):
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

            _, _, img_global, txt_global = model(images, input_ids, attention_mask)
            pair_emb = pairing_net(img_global, txt_global)

            all_img_globals.append(img_global.cpu())
            all_txt_globals.append(txt_global.cpu())
            all_pair_embs.append(pair_emb.cpu())
            all_captions.extend(captions)

    pairing_mat = torch.cat(all_pair_embs, dim=0).to(device)  # [N, D]
    N = pairing_mat.size(0)

    # --- choose partners and mixing weights (exact same logic as training) ---
    partner_indices = choose_mixing_partners(pairing_mat, top_k=5)        # [N]
    lambda_img, lambda_txt = compute_modality_mix_weights(pairing_mat, partner_indices)

    lambda_img = lambda_img.cpu().numpy()   # [N]
    partner_indices = partner_indices.cpu().numpy()

    # --- pick some random examples to visualize ---
    np.random.seed(123)
    num_examples = 6
    chosen_indices = np.random.choice(N, size=num_examples, replace=False)

    os.makedirs("viz", exist_ok=True)

    for idx in chosen_indices:
        j = int(partner_indices[idx])
        alpha = float(lambda_img[idx])     # how much of anchor vs partner

        anchor_id = all_image_ids[idx]
        partner_id = all_image_ids[j]

        anchor_cap = all_captions[idx]
        partner_cap = all_captions[j]

        img_a = load_raw_image(dataset, anchor_id, size=(256, 256))
        img_b = load_raw_image(dataset, partner_id, size=(256, 256))
        img_mix = blend_images(img_a, img_b, alpha=alpha)

        # --- plot: A | MIX | B ---
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(img_a)
        axes[0].axis("off")
        axes[0].set_title(f"Anchor {anchor_id}\nλ_img={alpha:.2f}\n{anchor_cap}", fontsize=8)

        axes[1].imshow(img_mix)
        axes[1].axis("off")
        axes[1].set_title("Pixel-space blend\n(not used in training)", fontsize=9)

        axes[2].imshow(img_b)
        axes[2].axis("off")
        axes[2].set_title(f"Partner {partner_id}\n(1-λ)={1-alpha:.2f}\n{partner_cap}", fontsize=8)

        plt.tight_layout()
        out_path = f"viz/mixed_pair_anchor_{anchor_id}_partner_{partner_id}.png"
        plt.savefig(out_path)
        plt.close(fig)

        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

