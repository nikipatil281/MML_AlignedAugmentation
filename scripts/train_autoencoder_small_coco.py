import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.datasets.coco_retrieval import build_coco_dataset, CocoRetrievalDataset
from src.datasets.collate import coco_retrieval_collate_fn
from src.models.simple_autoencoder import SimpleConvAutoencoder


# ImageNet normalization (likely what your transforms use)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def unnormalize(img: torch.Tensor) -> torch.Tensor:
    # img: normalized tensor [B,3,H,W] → [0,1]
    return img * IMAGENET_STD.to(img.device) + IMAGENET_MEAN.to(img.device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Use val2017 (5k images) – enough for a toy experiment
    images_root = "data/coco/images/val2017"
    captions_json = "data/coco/annotations/captions_val2017.json"
    instances_json = "data/coco/annotations/instances_val2017.json"

    dataset = build_coco_dataset(
        images_root=images_root,
        captions_json_path=captions_json,
        instances_json_path=instances_json,
        num_categories=91,
    )

    # Optionally restrict to a smaller subset, e.g. 3000 images
    max_images = 3000
    if len(dataset.image_ids) > max_images:
        dataset.image_ids = dataset.image_ids[:max_images]
        dataset._pairs = [(img_id, 0) for img_id in dataset.image_ids]

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        collate_fn=coco_retrieval_collate_fn,
    )

    model = SimpleConvAutoencoder(latent_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    num_epochs = 10

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch} [train AE]"):
            images = batch["image"].to(device)  # normalized [0,1] then normed by ImageNet in your transforms

            # Undo ImageNet normalization so AE learns 0–1 pixels
            imgs_unnorm = torch.clamp(unnormalize(images), 0.0, 1.0)

            optimizer.zero_grad()
            recon = model(imgs_unnorm)
            loss = criterion(recon, imgs_unnorm)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        print(f"Epoch {epoch}: avg recon loss = {avg_loss:.6f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/simple_autoencoder_coco_val2017.pt")
    print("Saved autoencoder checkpoint to checkpoints/simple_autoencoder_coco_val2017.pt")


if __name__ == "__main__":
    main()

