# Hybrid Multimodal Augmentation for Image–Text Retrieval

This project implements a full multimodal retrieval pipeline on **MS-COCO**, extended with a novel hybrid augmentation framework that combines:

* **Learned semantic pairing**
* **Fine-grained patch/token mixing (TokenMix)**
* **Alignment-aware feature augmentation**

The system progressively evolves from a simple baseline to a sophisticated, alignment-sensitive augmentation engine that creates meaningful synthetic training examples in **feature space**, not pixel space.

This README explains:

* How the project works
* The logical evolution of the method
* What each component does
* How to train, evaluate, and visualize everything
* A clean, thesis-ready overview

---

# 1. Problem Overview

We solve **image–text retrieval** on the MS-COCO dataset:

* Given an image → retrieve its caption
* Given a caption → retrieve its correct image

COCO provides:

* Images
* Five human-written captions per image
* Object category labels (dogs, beds, cars, etc.)

Our goal is to build **a stronger multimodal embedding space** by augmenting training with **synthetic, semantically meaningful mixed examples** — created directly in feature space.

---

# 2. Project Evolution (The Full Story)

Below is the entire journey from **baseline → semantic pairing → structured mixing → alignment-aware augmentation**.

---

## Step 1 — Baseline Image–Text Retrieval

Creates the standard multimodal retrieval model:

* **ResNet50** encodes images into 256-dim embeddings
* **BERT** encodes captions into 256-dim embeddings
* A contrastive loss pulls matched pairs together and pushes mismatched pairs apart

### Why this step

To ensure all infrastructure works: COCO dataloaders, encoders, training loop, and evaluation.

### Files

* `src/models/retrieval_model.py`
* `scripts/train_retrieval.py`
* `scripts/train_retrieval_full_coco.py`
* `scripts/eval_retrieval.py`
* `scripts/eval_retrieval_full_coco.py`

---

## Step 2 — LeMDA-Lite: Learnable Feature-Space Augmentation

Adds a small neural network (the **augmentor**) that:

1. Takes the fused image–text embedding
2. Slightly modifies it
3. Produces harder augmented examples

### Why this step

Baseline sees only real examples; synthetic variants improve robustness.

### Files

* `src/models/augmentor.py`
* `scripts/train_retrieval_lemdalike.py`

---

## Step 3 — Learned Semantic Pairing

A pairing network learns which samples **should be mixed** using COCO object labels.

### Why this step

Random mixing can create meaningless hybrids. We want **semantic mixing**.

### Files

* `src/models/pairing.py`
* `scripts/train_retrieval_lemdalike_pairing.py`

---

## Step 4 — Hybrid Global Mixing (Premix)

Uses the pairing network to:

1. Select semantically similar partners
2. Mix their global features with similarity-dependent strength λ

### Why this step

Introduce **semantically consistent synthetic examples**.

### Files

* `scripts/train_retrieval_hybrid_premix.py`

---

## Step 5 — TokenMix (Patch-level + Token-level Mixing)

Extends mixing to a **fine-grained level**:

* Mixes visual patches from ResNet feature maps (7×7 grid)
* Mixes text tokens from BERT
* Mixing strength guided by the semantic pairing network

### Why this step

Creates richer, structured synthetic examples.

### Files

* `src/models/retrieval_tokens_model.py`
* `scripts/train_retrieval_hybrid_tokenmix.py`

---

## Step 6 — Alignment-Aware Augmentation (Final Method)

Makes the augmentor aware of:

* Original image–text similarity
* Mixed image–text similarity
* Label density

### Why this step

Without alignment awareness, mixing may confuse the model. This stabilizes augmentation.

### Files

* `src/models/align_augmentor.py`
* `scripts/train_tokenmix_align_full_coco.py`

---

# 3. Repository Structure

```
project/
│
├─ src/
│   ├─ models/
│   │   ├─ retrieval_model.py
│   │   ├─ retrieval_tokens_model.py
│   │   ├─ augmentor.py
│   │   ├─ align_augmentor.py
│   │   ├─ pairing.py
│   │   └─ simple_autoencoder.py
│   │
│   ├─ datasets/
│   │   ├─ coco_retrieval.py
│   │   └─ collate.py
│   │
│   └─ utils/
│       ├─ metrics.py
│       └─ transforms.py
│
├─ scripts/
│   ├─ train_retrieval.py
│   ├─ train_retrieval_full_coco.py
│   ├─ train_retrieval_lemdalike.py
│   ├─ train_retrieval_lemdalike_pairing.py
│   ├─ train_retrieval_hybrid_premix.py
│   ├─ train_retrieval_hybrid_tokenmix.py
│   ├─ train_tokenmix_align_full_coco.py
│   ├─ eval_retrieval_full_coco.py
│   ├─ eval_tokenmix_align_full_coco.py
│   ├─ vis_retrieval_examples.py
│   ├─ vis_pairing_neighbors.py
│   ├─ vis_patch_similarity.py
│   ├─ train_autoencoder_small_coco.py
│   └─ vis_autoencoder_mixed_images.py
│
└─ checkpoints/
    └─ *.pt
```

---

# 4. How to Run

### Train Baseline

```
python scripts/train_retrieval.py
```

### Evaluate Baseline

```
python scripts/eval_retrieval.py
```

### Train Final TokenMix+Align Model

```
python scripts/train_tokenmix_align_full_coco.py
```

### Evaluate Final Model

```
python scripts/eval_tokenmix_align_full_coco.py
```

---

# 5. Visualization Tools

### Retrieve examples:

```
python scripts/vis_retrieval_examples.py
```

### Semantic pairing neighbors:

```
python scripts/vis_pairing_neighbors.py
```

### Patch similarity overlays:

```
python scripts/vis_patch_similarity.py
```

### Autoencoder mixed images:

```
python scripts/vis_autoencoder_mixed_images.py
```

---

# 6. Summary of Novel Contributions

### 1. Learned Semantic Pairing

Determines which samples should be mixed based on object labels and learned embeddings.

### 2. TokenMix

Mixes visual patches and text tokens to create fine-grained, semantically coherent synthetic examples.

### 3. Alignment-Aware Augmentation

Augmentor adjusts its behavior based on image–text alignment and object label density.

Together, these components build a hybrid multimodal augmentation framework for robust image–text retrieval.

---

This README summarizes the design, purpose, and workflow of the entire project, from baseline to your final hybrid augmentation method.
