## 1️⃣ `BASELINE_TRAINING.md`

# Baseline Image–Text Retrieval Training Pipeline
This document explains how to train and evaluate the **baseline retrieval model** on **COCO val2017 (4k train images)** using the codebase.
The goal here is to get a **clean, working baseline** before enabling LeMDA, pairing, TokenMix, etc.
---
## 0. Project Structure & Environment (recap)
Assumed layout:
FINALproject/
├── .venv/
├── data/
│   └── coco/
│       ├── images/
│       │   └── val2017/                 # COCO val2017 images
│       └── annotations/
│           ├── captions_val2017.json
│           └── instances_val2017.json
├── src/
│   ├── datasets/
│   │   ├── coco_retrieval.py
│   │   └── collate.py
│   ├── models/
│   │   ├── image_encoder.py
│   │   ├── text_encoder.py
│   │   └── retrieval_model.py
│   └── utils/
│       └── metrics.py
└── scripts/
    ├── debug_dataloader.py
    ├── train_retrieval.py
    └── eval_retrieval.py

Environment (already done once):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
All commands below assume you are in the **project root**:
```bash
cd ~/Desktop/MML/FINALproject
source .venv/bin/activate
```
---

## 1. Check Data Loading (Debug Dataloader)
Script: `scripts/debug_dataloader.py`
This script:
* Builds COCO splits from `captions_val2017.json` and `instances_val2017.json`
* Creates `CocoRetrievalDataset` with:
  * Images
  * Captions
  * Multi-label COCO categories
* Uses `coco_retrieval_collate_fn` to batch correctly

Run:
```bash
python scripts/debug_dataloader.py
```

Expected output (similar):
```text
image batch: torch.Size([4, 3, 224, 224])
caption batch: ['A man in a black tie...', '...', ...]
image_ids: [262895, 104803, ...]
category_ids (per sample): [[1, 32], [70], ...]
category_multi_hot: torch.Size([4, 91])
```
If this works, your **data + labels pipeline is correct**.

---

## 2. Train Baseline Retrieval Model (No Augmentation)
Script: `scripts/train_retrieval.py`
Model: `src/models/retrieval_model.py`
### 2.1 What this script does
* Uses:
  * `ImageEncoder` (ResNet-50) → `img_emb`  ∈ ℝᴰ
  * `TextEncoder`  (BERT)      → `txt_emb` ∈ ℝᴰ
* Trains with **symmetric contrastive loss** (`image ↔ text`) on:
  * 4,000 train images (internal split of val2017)
* Saves baseline weights to:
```text
checkpoints/retrieval_baseline.pt
```

### 2.2 Where to change number of epochs

Inside `scripts/train_retrieval.py`:

```python
num_epochs = 2  # <--- change this to 10, 20, etc for longer training
```
### 2.3 Run training
```bash
python scripts/train_retrieval.py
```
Example output:

```text
Using device: cuda
Epoch 1 [train]: 100%|██████████| 626/626 [01:05<...>]
Epoch 1: avg train loss = 0.7898
Epoch 2 [train]: 100%|██████████| 626/626 [01:05<...>]
Epoch 2: avg train loss = 0.3536
Saved model to checkpoints/retrieval_baseline.pt
```
---
## 3. Evaluate Baseline Retrieval
Script: `scripts/eval_retrieval.py`
Metrics: **R@1, R@5, R@10** for:
* Image → Text
* Text → Image
### 3.1 What this script does
* Loads:
  * `checkpoints/retrieval_baseline.pt`
* Builds the **test split** (500 images) out of val2017.
* Uses **one caption per image** (caption index 0) for evaluation.
* Encodes all images & captions.
* Computes similarity matrix and R@K metrics using `src/utils/metrics.py`.
### 3.2 Run evaluation
```bash
python scripts/eval_retrieval.py
```
Example output:
```text
Image-to-Text retrieval:
  R@1: 24.00
  R@5: 61.80
  R@10: 76.20
Text-to-Image retrieval:
  R@1: 24.40
  R@5: 59.00
  R@10: 76.40
```
These serve as your **baseline metrics** to compare against all more advanced augmentation methods later.
---

## 4. Summary – Baseline Pipeline Order
1. **(Once)** Create venv + install requirements
2. **(Once / after changes)** Verify dataloader:
   ```bash
   python scripts/debug_dataloader.py
   ```
3. **Train baseline** retrieval model:
   ```bash
   python scripts/train_retrieval.py
   ```
4. **Evaluate baseline** on test split:
   ```bash
   python scripts/eval_retrieval.py
   ```
This completes the **baseline training and evaluation** on COCO val2017 (4k train images).

---
