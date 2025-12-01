# scripts/train_vae_coco_small.py

import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from experiments.vae_mixing.vae import ConvVAE, vae_loss
from experiments.vae_mixing.coco_vae_dataset import CocoVAEDataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- paths for train2017 ----
    train_images_root = "data/coco/images/train2017"
    train_captions_json = "data/coco/annotations/captions_train2017.json"
    train_instances_json = "data/coco/annotations/instances_train2017.json"

    # ---- dataset: small subset ----
    dataset = CocoVAEDataset(
        images_root=train_images_root,
        captions_json_path=train_captions_json,
        instances_json_path=train_instances_json,
        subset_size=20000,  # CHANGE HERE if you want smaller/bigger
        seed=42,
        image_size=64,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # ---- VAE ----
    latent_dim = 128
    vae = ConvVAE(latent_dim=latent_dim).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    num_epochs = 15  # CHANGE HERE for more/less training
    beta = 1.0       # KL weight

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        vae.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch} [VAE train]"):
            images = batch["image"].to(device)  # [B,3,64,64]

            optimizer.zero_grad()
            x_recon, mu, logvar = vae(images)
            loss, recon_loss, kl_loss = vae_loss(images, x_recon, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        avg_recon = total_recon / max(1, num_batches)
        avg_kl = total_kl / max(1, num_batches)
        print(
            f"Epoch {epoch}: loss={avg_loss:.3f}, "
            f"recon={avg_recon:.3f}, kl={avg_kl:.3f}"
        )

        # save checkpoint each epoch (or only last)
        torch.save(vae.state_dict(), "checkpoints/coco_vae_small.pt")

    print("Saved VAE checkpoint to checkpoints/coco_vae_small.pt")


if __name__ == "__main__":
    main()

