# scripts/eval_retrieval_full_coco.py

import os
import sys
from typing import List

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm

# Make project root importable
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.datasets.coco_retrieval import build_coco_dataset, CocoRetrievalDataset
from src.datasets.collate import coco_retrieval_collate_fn
from src.models.retrieval_model import ImageTextRetrievalModel
from src.utils.metrics import compute_retrieval_metrics


def build_single_caption_subset(dataset: CocoRetrievalDataset) -> CocoRetrievalDataset:
    """
    Reuse the same dataset object but restrict to exactly
    ONE caption per image (caption index 0).

    This keeps the evaluation square: N images vs N captions.
    """
    subset = dataset
    subset._pairs = [(img_id, 0) for img_id in subset.image_ids]
    return subset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ----- Paths for FULL COCO VAL2017 -----
    images_root = "data/coco/images/val2017"
    captions_json = "data/coco/annotations/captions_val2017.json"
    instances_json = "data/coco/annotations/instances_val2017.json"

    # ----- Build full val2017 dataset -----
    val_dataset = build_coco_dataset(
        images_root=images_root,
        captions_json_path=captions_json,
        instances_json_path=instances_json,
        num_categories=91,  # COCO category count (or adjust if you use different)
    )

    # Use only one caption per image for eval (square matrix)
    val_dataset = build_single_caption_subset(val_dataset)

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,            # tune based on GPU/CPU memory
        shuffle=False,
        num_workers=4,
        collate_fn=coco_retrieval_collate_fn,
    )

    # ----- Model -----
    embed_dim = 256
    model = ImageTextRetrievalModel(
        img_embed_dim=embed_dim,
        txt_embed_dim=embed_dim,
        txt_model_name="bert-base-uncased",
    ).to(device)

    ckpt_path = "checkpoints/retrieval_baseline_full_coco.pt"
    print(f"Loading checkpoint from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    all_img_embeds: List[torch.Tensor] = []
    all_txt_embeds: List[torch.Tensor] = []

    # ----- Encode all images + captions -----
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Encoding val2017"):
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

            img_emb, txt_emb = model(images, input_ids, attention_mask)

            all_img_embeds.append(img_emb.cpu())
            all_txt_embeds.append(txt_emb.cpu())

    all_img_embeds = torch.cat(all_img_embeds, dim=0).numpy()  # [N, D]
    all_txt_embeds = torch.cat(all_txt_embeds, dim=0).numpy()  # [N, D]

    # ----- Similarities + retrieval metrics -----
    sim_i2t = all_img_embeds @ all_txt_embeds.T   # [N, N]
    sim_t2i = all_txt_embeds @ all_img_embeds.T   # [N, N]

    metrics_i2t = compute_retrieval_metrics(sim_i2t)
    metrics_t2i = compute_retrieval_metrics(sim_t2i)

    print("\nImage-to-Text retrieval on full COCO val2017:")
    for k, v in metrics_i2t.items():
        print(f"  {k}: {v:.2f}")

    print("\nText-to-Image retrieval on full COCO val2017:")
    for k, v in metrics_t2i.items():
        print(f"  {k}: {v:.2f}")


if __name__ == "__main__":
    main()

