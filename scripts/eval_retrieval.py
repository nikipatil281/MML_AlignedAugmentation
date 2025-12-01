# scripts/eval_retrieval.py

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

from src.datasets.coco_retrieval import build_coco_retrieval_splits, CocoRetrievalDataset
from src.datasets.collate import coco_retrieval_collate_fn
from src.models.retrieval_model import ImageTextRetrievalModel
from src.utils.metrics import compute_retrieval_metrics


def build_single_caption_subset(dataset: CocoRetrievalDataset) -> CocoRetrievalDataset:
    """
    Build a shallow copy of the dataset that uses only ONE caption per image.

    This is just for quick, square similarity evaluation (N images, N captions).
    We simply keep caption index 0 for each image.
    """
    # Hack: we'll reuse the same dataset class but override its internal _pairs list.
    subset = dataset
    subset._pairs = [(img_id, 0) for img_id in subset.image_ids]  # one caption per image_id
    return subset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Paths (same as training)
    images_root = "data/coco/images/val2017"
    captions_json = "data/coco/annotations/captions_val2017.json"
    instances_json = "data/coco/annotations/instances_val2017.json"

    # Build splits (same as before)
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

    # Use test split (you can also evaluate on val)
    test_dataset = splits.test
    test_dataset = build_single_caption_subset(test_dataset)

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=coco_retrieval_collate_fn,
    )

    # Model & tokenizer
    model = ImageTextRetrievalModel(
        img_embed_dim=256,
        txt_embed_dim=256,
        txt_model_name="bert-base-uncased",
    ).to(device)

    ckpt_path = "checkpoints/retrieval_baseline.pt"
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Collect embeddings
    all_img_embeds: List[torch.Tensor] = []
    all_txt_embeds: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Encoding test set"):
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

    # Similarity matrices
    sim_i2t = all_img_embeds @ all_txt_embeds.T   # [N, N]
    sim_t2i = all_txt_embeds @ all_img_embeds.T   # [N, N]

    metrics_i2t = compute_retrieval_metrics(sim_i2t)
    metrics_t2i = compute_retrieval_metrics(sim_t2i)

    print("\nImage-to-Text retrieval:")
    for k, v in metrics_i2t.items():
        print(f"  {k}: {v:.2f}")

    print("\nText-to-Image retrieval:")
    for k, v in metrics_t2i.items():
        print(f"  {k}: {v:.2f}")


if __name__ == "__main__":
    main()
