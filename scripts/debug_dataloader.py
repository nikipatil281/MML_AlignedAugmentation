# scripts/debug_dataloader.py

import os
import sys

# Add project root to sys.path so "src" is importable if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from torch.utils.data import DataLoader

from src.datasets.coco_retrieval import build_coco_retrieval_splits
from src.datasets.collate import coco_retrieval_collate_fn


def main():
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
        num_categories=91,  # adjust later if you want exact number
    )

    train_loader = DataLoader(
        splits.train,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # 0 for easier debugging
        collate_fn=coco_retrieval_collate_fn,
    )

    batch = next(iter(train_loader))
    print("image batch:", batch["image"].shape)
    print("caption batch:", batch["caption"])
    print("image_ids:", batch["image_id"])
    print("category_ids (per sample):", batch["category_ids"])
    if "category_multi_hot" in batch:
        print("category_multi_hot:", batch["category_multi_hot"].shape)


if __name__ == "__main__":
    main()

