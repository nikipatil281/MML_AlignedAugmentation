## 2️⃣ `SCALE_TO_COCO2017.md`

# Scaling the Project to Full COCO 2017
This document explains how to scale the entire pipeline from the small **val2017-only** setup to **full COCO 2017**:
- Use **train2017** for training.
- Use **val2017** for validation / test.
- Increase **epochs** for each method.
- Specify **which scripts to run**, in which order.
---

## 0. Dataset Setup for Full COCO 2017
Download the full COCO 2017 data (if not already done):
- `train2017` images
- `val2017` images
- `captions_train2017.json`
- `captions_val2017.json`
- `instances_train2017.json`
- `instances_val2017.json`

Recommended structure:
data/
└── coco/
    ├── images/
    │   ├── train2017/
    │   └── val2017/
    └── annotations/
        ├── captions_train2017.json
        ├── captions_val2017.json
        ├── instances_train2017.json
        └── instances_val2017.json

Right now, your code uses **only val2017** for all splits. For scaling, we want:
* **Train** on: `train2017` (≈118k images)
* **Val/Test** on: `val2017` (≈5k images)
We’ll do this with **two separate calls** to the dataset-building function—one for train, one for eval.
---
## 1. Updating Data Loading for Full COCO (Train vs Val)
File to modify: `src/datasets/coco_retrieval.py`
Currently, `build_coco_retrieval_splits(...)` uses a **single pair** of JSONs (val2017) and internally splits image_ids into train/val/test.
For full COCO, a clean approach is:
### 1.1 New helper to build a dataset from specific JSON + images root
Add this function at the bottom of `coco_retrieval.py`:
```python
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
```
This will let you build:
* `train_dataset` from train2017
* `val_dataset` / `test_dataset` from val2017
---
## 2. Training on Full Train2017
### 2.1 Baseline training script (full COCO)
Create a new script (or modify the existing one carefully):
**Option A – New script recommended:**
`scripts/train_retrieval_full_coco.py`
```python
# scripts/train_retrieval_full_coco.py

import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import torch.nn as nn

# project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.datasets.coco_retrieval import build_coco_dataset
from src.datasets.collate import coco_retrieval_collate_fn
from src.models.retrieval_model import ImageTextRetrievalModel


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


def train_one_epoch(
    model,
    tokenizer,
    dataloader,
    optimizer,
    device,
    epoch: int,
):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch} [train-full]"):
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

        optimizer.zero_grad()
        img_emb, txt_emb = model(images, input_ids, attention_mask)
        loss = contrastive_loss(img_emb, txt_emb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_images_root = "data/coco/images/train2017"
    train_captions_json = "data/coco/annotations/captions_train2017.json"
    train_instances_json = "data/coco/annotations/instances_train2017.json"

    train_dataset = build_coco_dataset(
        images_root=train_images_root,
        captions_json_path=train_captions_json,
        instances_json_path=train_instances_json,
        num_categories=91,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,        # you can adjust based on GPU memory
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

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # CHANGE EPOCHS HERE FOR FULL TRAINING:
    num_epochs = 10  # e.g., 10–20 for full COCO

    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(model, tokenizer, train_loader, optimizer, device, epoch)
        print(f"Epoch {epoch}: avg train loss (full COCO) = {avg_loss:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/retrieval_baseline_full_coco.pt")
    print("Saved full COCO baseline model.")


if __name__ == "__main__":
    main()
```

### 2.2 Run full baseline training
```bash
python scripts/train_retrieval_full_coco.py
```
---
## 3. Evaluation on Full Val2017
Make a full COCO eval script (or adapt your existing one):
`scripts/eval_retrieval_full_coco.py`
* Use `build_coco_dataset` with:
  * `images_root = data/coco/images/val2017`
  * `captions_val2017.json`, `instances_val2017.json`
* Load `checkpoints/retrieval_baseline_full_coco.pt`
* Use the same evaluation logic as `eval_retrieval.py`.
You can reuse `src/utils/metrics.py` the same way.
---
## 4. Scaling Advanced Methods (LeMDA, Pairing, TokenMix, Align)
Each **training script** has `num_epochs` hardcoded; for full COCO you should:
* Increase `num_epochs` in each file to more realistic values, e.g.:

| Script                                         | Purpose                                     | Suggested Epochs (full COCO)                |
| ---------------------------------------------- | ------------------------------------------- | ------------------------------------------- |
| `scripts/train_retrieval_full_coco.py`         | Baseline                                    | 10–20                                       |
| `scripts/train_retrieval_lemdalike.py`         | LeMDA-Lite (global)                         | 5–10 (start from baseline ckpt if you want) |
| `scripts/train_retrieval_lemdalike_pairing.py` | LeMDA + Pairing                             | 5–10                                        |
| `scripts/train_retrieval_hybrid_premix.py`     | Global semantic premix + LeMDA              | 5–10                                        |
| `scripts/train_retrieval_hybrid_tokenmix.py`   | TokenMix (patch/token mixing)               | 5–10                                        |
| `scripts/train_retrieval_tokenmix_align.py`    | TokenMix + Alignment-aware augmentor (FULL) | 5–10                                        |

> You can start lower (e.g., 5 epochs) and increase once you confirm stability.

In each of these scripts, look for the line:
```python
num_epochs = 2  # <= change this to e.g. 5, 10, etc
```
and modify accordingly.
---

## 5. Recommended Run Order (Full COCO)
1. **Train full baseline**
   `python scripts/train_retrieval_full_coco.py`
   → `checkpoints/retrieval_baseline_full_coco.pt`
2. **Evaluate full baseline**
   `python scripts/eval_retrieval_full_coco.py`
   → Get R@1/5/10 baseline.
3. **Train LeMDA-Lite** (optional: initialise from baseline ckpt):
   * Update `train_retrieval_lemdalike.py` to use:
     * `train2017` JSONs via `build_coco_dataset` or a new full-COCO version of that script.
     * `num_epochs = 5–10`.
   * Run:
     ```bash
     python scripts/train_retrieval_lemdalike.py
     ```
4. **Train LeMDA + Pairing**
   * Adapt `train_retrieval_lemdalike_pairing.py` to train on train2017.
   * Increase `num_epochs`.
   * Run:
     ```bash
     python scripts/train_retrieval_lemdalike_pairing.py
     ```
5. **Train Hybrid Premix (Global mixing)**
   ```bash
   python scripts/train_retrieval_hybrid_premix.py
   ```
   (after adapting it to train2017, similar to Step 3)
6. **Train TokenMix** (patch/token level)
   ```bash
   python scripts/train_retrieval_hybrid_tokenmix.py
   ```
7. **Train TokenMix + Align (Full model)**
   ```bash
   python scripts/train_retrieval_tokenmix_align.py
   ```
8. **Evaluate token-based models** on val2017
   * Use `scripts/eval_retrieval_token_models.py` and switch:
     ```python
     ckpt_path = "checkpoints/retrieval_tokenmix_model.pt"
     # and
     ckpt_path = "checkpoints/retrieval_tokenmix_align_model.pt"
     ```
   * Collect R@1/5/10 for:
     * TokenMix
     * TokenMix+Align
---
## 6. Final Checklist for Scaling
* [ ] COCO 2017 data organised under `data/coco/` (train2017 + val2017).
* [ ] `build_coco_dataset(...)` added to `coco_retrieval.py`.
* [ ] New (or updated) scripts for:
  * `train_retrieval_full_coco.py`
  * `eval_retrieval_full_coco.py`
* [ ] Updated `num_epochs` in all training scripts for full COCO.
* [ ] All advanced methods (LeMDA, Pairing, Premix, TokenMix, Align) adapted to use **train2017** for training and **val2017** for evaluation.
* [ ] R@1/R@5/R@10 recorded for:
  * Baseline (full COCO)
  * * LeMDA
  * * Pairing
  * * Premix
  * * TokenMix
  * * TokenMix+Align (final)

This gives you a **full-scale training & evaluation pipeline** on complete COCO 2017, ready for thesis / paper experiments.


