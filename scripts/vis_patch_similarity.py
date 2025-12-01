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

# ----------------- project imports ----------------- #

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.datasets.coco_retrieval import build_coco_dataset, CocoRetrievalDataset
from src.datasets.collate import coco_retrieval_collate_fn
from src.models.retrieval_tokens_model import ImageTextRetrievalTokensModel
from src.models.pairing import PairingNetwork


# ----------------- dataset helpers ----------------- #

def build_single_caption_subset(dataset: CocoRetrievalDataset) -> CocoRetrievalDataset:
    """
    Use exactly one caption (index 0) per image.
    This keeps indexing simple: index i <-> image_ids[i]
    """
    subset = dataset
    subset._pairs = [(img_id, 0) for img_id in subset.image_ids]
    return subset


def load_raw_image(dataset: CocoRetrievalDataset, image_id: int, size=(256, 256)) -> Image.Image:
    rel_path = dataset.image_id_to_filename[image_id]
    img_path = os.path.join(dataset.images_root, rel_path)
    img = Image.open(img_path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    return img


# ----------------- main visualization ----------------- #

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- COCO val2017 paths ----
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
        shuffle=False,
        num_workers=4,
        collate_fn=coco_retrieval_collate_fn,
    )

    # ---- load TokenMix model & pairing network ----
    embed_dim = 256
    model = ImageTextRetrievalTokensModel(
        img_embed_dim=embed_dim,
        txt_embed_dim=embed_dim,
        txt_model_name="bert-base-uncased",
    ).to(device)

    pairing_net = PairingNetwork(
        embed_dim=embed_dim,
        pairing_dim=embed_dim,
        hidden_dim=512,
    ).to(device)

    model_ckpt = "checkpoints/retrieval_tokenmix_align_full_coco_model.pt"
    pairing_ckpt = "checkpoints/retrieval_tokenmix_align_full_coco_pairing.pt"

    print(f"Loading model from   {model_ckpt}")
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    print(f"Loading pairing net from {pairing_ckpt}")
    pairing_net.load_state_dict(torch.load(pairing_ckpt, map_location=device))

    model.eval()
    pairing_net.eval()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # ---- collect tokens + pairing embeddings for a subset of val2017 ----
    max_samples = 1024   # limit for memory; change if you want more
    all_img_tokens: List[torch.Tensor] = []
    all_pair_embs: List[torch.Tensor] = []
    all_image_ids: List[int] = list(dataset.image_ids)
    all_captions: List[str] = []

    num_collected = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding val2017 for patch similarity"):
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

            img_tokens, _, img_global, txt_global = model(images, input_ids, attention_mask)
            pair_emb = pairing_net(img_global, txt_global)

            all_img_tokens.append(img_tokens.cpu())   # [B, Np, D]
            all_pair_embs.append(pair_emb.cpu())      # [B, P]
            all_captions.extend(captions)

            num_collected += img_tokens.size(0)
            if num_collected >= max_samples:
                break

    img_tokens_all = torch.cat(all_img_tokens, dim=0)        # [M, Np, D]
    pair_emb_all = torch.cat(all_pair_embs, dim=0)           # [M, P]
    M, Np, D = img_tokens_all.shape
    print(f"Collected {M} samples, {Np} patches per image, dim={D}")

    # Determine grid size (e.g. 7x7 for ResNet-50 224x224)
    grid_side = int(np.sqrt(Np))
    assert grid_side * grid_side == Np, "Patch count is not a perfect square"
    Hf = Wf = grid_side

    # Move pairing embeddings to numpy for similarity computation
    pair_np = pair_emb_all.numpy()                           # [M, P]
    sim = pair_np @ pair_np.T                                # cosine since embeddings are normalized

    # For each anchor, choose most similar partner (excluding self)
    np.fill_diagonal(sim, -1e9)
    partner_indices = np.argmax(sim, axis=1)                 # [M]

    # ---- choose a few anchors to visualize ----
    np.random.seed(42)
    num_examples = 6
    chosen_indices = np.random.choice(M, size=num_examples, replace=False)

    os.makedirs("viz", exist_ok=True)

    for anchor_idx in chosen_indices:
        partner_idx = int(partner_indices[anchor_idx])

        # Get tokens
        t_anchor = img_tokens_all[anchor_idx]   # [Np, D]
        t_partner = img_tokens_all[partner_idx] # [Np, D]

        # Ensure L2-normalized along D (should already be, but just to be safe)
        t_anchor = torch.nn.functional.normalize(t_anchor, p=2, dim=-1)
        t_partner = torch.nn.functional.normalize(t_partner, p=2, dim=-1)

        # Patch-wise cosine similarity: [Np]
        patch_sim = (t_anchor * t_partner).sum(dim=-1).numpy()
        patch_sim_grid = patch_sim.reshape(Hf, Wf)

        # Normalize to [0,1] for visualization
        min_val, max_val = patch_sim_grid.min(), patch_sim_grid.max()
        if max_val > min_val:
            patch_vis = (patch_sim_grid - min_val) / (max_val - min_val)
        else:
            patch_vis = patch_sim_grid

        # Load raw images
        anchor_img_id = all_image_ids[anchor_idx]
        partner_img_id = all_image_ids[partner_idx]
        anchor_cap = all_captions[anchor_idx]
        partner_cap = all_captions[partner_idx]

        img_a = load_raw_image(dataset, anchor_img_id, size=(256, 256))
        img_b = load_raw_image(dataset, partner_img_id, size=(256, 256))

        # ---- plot: anchor | partner | patch similarity heatmap ----
                # ---- upsample patch heatmap to image size for overlay ----
        patch_img = Image.fromarray((patch_vis * 255).astype(np.uint8))
        patch_img = patch_img.resize(img_a.size, resample=Image.BILINEAR)
        patch_overlay = np.array(patch_img).astype(np.float32) / 255.0

        # ---- plot: anchor | anchor+heatmap | partner ----
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # 1) Anchor
        axes[0].imshow(img_a)
        axes[0].axis("off")
        axes[0].set_title(f"Anchor {anchor_img_id}\n{anchor_cap}", fontsize=8)

        # 2) Anchor with similarity overlay
        axes[1].imshow(img_a)
        axes[1].imshow(
            patch_overlay,
            cmap="viridis",
            alpha=0.5,      # transparency of heatmap
            vmin=0.0,
            vmax=1.0,
        )
        axes[1].axis("off")
        axes[1].set_title("Anchor + patch similarity\n(to partner)", fontsize=9)

        # 3) Partner
        axes[2].imshow(img_b)
        axes[2].axis("off")
        axes[2].set_title(f"Partner {partner_img_id}\n{partner_cap}", fontsize=8)

        plt.tight_layout()
        out_path = f"viz/patch_similarity_anchor_{anchor_img_id}_partner_{partner_img_id}.png"
        plt.savefig(out_path)
        plt.close(fig)

        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

