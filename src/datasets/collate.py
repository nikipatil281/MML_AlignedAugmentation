# src/datasets/collate.py

from typing import List, Dict, Any
import torch


def coco_retrieval_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for CocoRetrievalDataset.

    - Stacks images into a tensor [B, C, H, W]
    - Keeps captions as a list of strings
    - Keeps image_ids as a list of ints
    - Keeps category_ids as a list of lists (variable length)
    - Stacks category_multi_hot if present
    """
    images = torch.stack([b["image"] for b in batch], dim=0)
    captions = [b["caption"] for b in batch]
    image_ids = [b["image_id"] for b in batch]
    category_ids = [b["category_ids"] for b in batch]

    out: Dict[str, Any] = {
        "image": images,
        "caption": captions,
        "image_id": image_ids,
        "category_ids": category_ids,
    }

    if "category_multi_hot" in batch[0] and batch[0]["category_multi_hot"] is not None:
        category_multi_hot = torch.stack(
            [b["category_multi_hot"] for b in batch], dim=0
        )  # [B, num_categories]
        out["category_multi_hot"] = category_multi_hot

    return out

