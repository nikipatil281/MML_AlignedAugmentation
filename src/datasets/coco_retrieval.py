# src/datasets/coco_retrieval.py

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Set

from PIL import Image
import torch
from torch.utils.data import Dataset


def load_coco_annotations(
    captions_json_path: str,
    instances_json_path: str,
) -> Tuple[
    Dict[int, str],              # image_id -> file_name
    Dict[int, List[str]],        # image_id -> list of captions
    Dict[int, Set[int]],         # image_id -> set of category_ids (multi-label)
    Dict[int, str],              # category_id -> category_name
]:
    """
    Load COCO-style captions and instances JSONs and build:
      - image_id -> file_name
      - image_id -> [captions]
      - image_id -> {category_ids}
      - category_id -> category_name
    """

    # ---- 1. Load captions JSON ----
    with open(captions_json_path, "r") as f:
        caps_data = json.load(f)

    # Standard COCO captions file has "images" and "annotations"
    # images: [{"id": 179765, "file_name": "000000179765.jpg", ...}, ...]
    # annotations: [{"image_id": 179765, "caption": "...", ...}, ...]
    image_id_to_filename: Dict[int, str] = {}
    for img in caps_data.get("images", []):
        img_id = img["id"]
        file_name = img["file_name"]
        image_id_to_filename[img_id] = file_name

    image_id_to_captions: Dict[int, List[str]] = defaultdict(list)
    for ann in caps_data.get("annotations", []):
        img_id = ann["image_id"]
        caption = ann["caption"]
        image_id_to_captions[img_id].append(caption)

    # ---- 2. Load instances JSON ----
    with open(instances_json_path, "r") as f:
        inst_data = json.load(f)

    # categories: [{"id": 1, "name": "person", "supercategory": "person"}, ...]
    category_id_to_name: Dict[int, str] = {}
    for cat in inst_data.get("categories", []):
        category_id_to_name[cat["id"]] = cat["name"]

    # annotations: [{"image_id": ..., "category_id": ..., ...}, ...]
    image_id_to_category_ids: Dict[int, Set[int]] = defaultdict(set)
    for ann in inst_data.get("annotations", []):
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        image_id_to_category_ids[img_id].add(cat_id)

    # NOTE:
    # Some image_ids may appear in captions but not in instances (or vice versa).
    # For retrieval we care that images have captions; for labels we can allow empty sets.
    # We'll keep all image_ids that have at least ONE caption.

    # Filter to only image_ids that have captions
    valid_image_ids = set(image_id_to_captions.keys())
    image_id_to_filename = {
        img_id: fn for img_id, fn in image_id_to_filename.items()
        if img_id in valid_image_ids
    }

    # For any valid image_id that has no instances, ensure we still have an empty set
    for img_id in valid_image_ids:
        if img_id not in image_id_to_category_ids:
            image_id_to_category_ids[img_id] = set()

    return (
        image_id_to_filename,
        image_id_to_captions,
        image_id_to_category_ids,
        category_id_to_name,
    )

import random

def split_image_ids(
    image_ids: List[int],
    num_train: int,
    num_val: int,
    num_test: int,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Deterministically split image_ids into train/val/test given requested sizes.

    Assumes: num_train + num_val + num_test <= len(image_ids).
    """

    assert num_train + num_val + num_test <= len(image_ids), \
        "Requested split sizes exceed number of available images."

    rng = random.Random(seed)
    image_ids_shuffled = image_ids.copy()
    rng.shuffle(image_ids_shuffled)

    train_ids = image_ids_shuffled[:num_train]
    val_ids = image_ids_shuffled[num_train:num_train + num_val]
    test_ids = image_ids_shuffled[num_train + num_val:num_train + num_val + num_test]

    return train_ids, val_ids, test_ids


from typing import Optional, Any, Sequence
import torchvision.transforms as T


class CocoRetrievalDataset(Dataset):
    """
    COCO image-text retrieval dataset.

    Each item is one (image, caption) pair:
      - Image loaded from disk and transformed to tensor.
      - Caption is a raw string.
      - category_ids: list of category ids (multi-label).
    """

    def __init__(
        self,
        images_root: str,
        image_ids: Sequence[int],
        image_id_to_filename: Dict[int, str],
        image_id_to_captions: Dict[int, List[str]],
        image_id_to_category_ids: Dict[int, Set[int]],
        num_categories: Optional[int] = None,
        transform: Optional[Any] = None,
    ):
        """
        Args:
            images_root: path to directory containing images (e.g. .../val2017)
            image_ids: list of image_ids belonging to this split (train/val/test)
            image_id_to_filename: mapping image_id -> file_name
            image_id_to_captions: mapping image_id -> list of captions (strings)
            image_id_to_category_ids: mapping image_id -> set of category_ids
            num_categories: if provided, will build multi-hot vectors of this length
            transform: torchvision transforms to apply to PIL image
        """
        self.images_root = images_root
        self.image_ids = list(image_ids)
        self.image_id_to_filename = image_id_to_filename
        self.image_id_to_captions = image_id_to_captions
        self.image_id_to_category_ids = image_id_to_category_ids
        self.num_categories = num_categories

        # Default transform: resize + center crop + to tensor + normalize
        if transform is None:
            self.transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            self.transform = transform

        # Build a flat list of (image_id, caption_idx) pairs
        self._pairs = []
        for img_id in self.image_ids:
            captions = self.image_id_to_captions.get(img_id, [])
            for cap_idx in range(len(captions)):
                self._pairs.append((img_id, cap_idx))

    def __len__(self) -> int:
        return len(self._pairs)

    def _get_multi_hot(self, category_ids: Set[int]) -> Optional[torch.Tensor]:
        if self.num_categories is None:
            return None
        # COCO category ids are not necessarily contiguous from 0..(num_categories-1)
        # For now we assume they are 1..K and we map them to 0..K-1.
        # If needed, we can build a mapping in a helper later.
        multi_hot = torch.zeros(self.num_categories, dtype=torch.float32)
        for cid in category_ids:
            if 1 <= cid <= self.num_categories:
                multi_hot[cid - 1] = 1.0
        return multi_hot

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_id, cap_idx = self._pairs[idx]

        # ---- Image ----
        file_name = self.image_id_to_filename[img_id]
        img_path = os.path.join(self.images_root, file_name)
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        image_tensor = self.transform(img)

        # ---- Caption ----
        captions = self.image_id_to_captions[img_id]
        caption = captions[cap_idx]

        # ---- Multi-label categories ----
        category_ids_set = self.image_id_to_category_ids.get(img_id, set())
        category_ids = sorted(list(category_ids_set))
        category_multi_hot = self._get_multi_hot(category_ids_set)

        sample = {
            "image": image_tensor,               # Tensor, CxHxW
            "caption": caption,                  # str
            "image_id": img_id,                  # int
            "category_ids": category_ids,        # List[int]
        }

        if category_multi_hot is not None:
            sample["category_multi_hot"] = category_multi_hot  # Tensor [num_categories]

        return sample


from typing import NamedTuple


class CocoRetrievalSplits(NamedTuple):
    train: CocoRetrievalDataset
    val: CocoRetrievalDataset
    test: CocoRetrievalDataset


def build_coco_retrieval_splits(
    images_root: str,
    captions_json_path: str,
    instances_json_path: str,
    num_train: int,
    num_val: int,
    num_test: int,
    seed: int = 42,
    num_categories: Optional[int] = 91,   # COCO has 80 'things' + some 'stuff'; adjust if needed
) -> CocoRetrievalSplits:
    (
        image_id_to_filename,
        image_id_to_captions,
        image_id_to_category_ids,
        category_id_to_name,
    ) = load_coco_annotations(captions_json_path, instances_json_path)

    all_image_ids = sorted(list(image_id_to_captions.keys()))

    train_ids, val_ids, test_ids = split_image_ids(
        all_image_ids, num_train=num_train, num_val=num_val, num_test=num_test, seed=seed
    )

    # You may later want to save category_id_to_name somewhere global
    # for analysis or label visualization.

    train_dataset = CocoRetrievalDataset(
        images_root=images_root,
        image_ids=train_ids,
        image_id_to_filename=image_id_to_filename,
        image_id_to_captions=image_id_to_captions,
        image_id_to_category_ids=image_id_to_category_ids,
        num_categories=num_categories,
    )
    val_dataset = CocoRetrievalDataset(
        images_root=images_root,
        image_ids=val_ids,
        image_id_to_filename=image_id_to_filename,
        image_id_to_captions=image_id_to_captions,
        image_id_to_category_ids=image_id_to_category_ids,
        num_categories=num_categories,
    )
    test_dataset = CocoRetrievalDataset(
        images_root=images_root,
        image_ids=test_ids,
        image_id_to_filename=image_id_to_filename,
        image_id_to_captions=image_id_to_captions,
        image_id_to_category_ids=image_id_to_category_ids,
        num_categories=num_categories,
    )

    return CocoRetrievalSplits(train=train_dataset, val=val_dataset, test=test_dataset)


def build_coco_dataset(
    images_root: str,
    captions_json_path: str,
    instances_json_path: str,
    num_categories: int = 91,
) -> CocoRetrievalDataset:
    (
        image_id_to_filename,
        image_id_to_captions,
        image_id_to_category_ids,
        category_id_to_name,
    ) = load_coco_annotations(captions_json_path, instances_json_path)

    all_image_ids = sorted(list(image_id_to_captions.keys()))

    dataset = CocoRetrievalDataset(
        images_root=images_root,
        image_ids=all_image_ids,
        image_id_to_filename=image_id_to_filename,
        image_id_to_captions=image_id_to_captions,
        image_id_to_category_ids=image_id_to_category_ids,
        num_categories=num_categories,
    )

    return dataset
