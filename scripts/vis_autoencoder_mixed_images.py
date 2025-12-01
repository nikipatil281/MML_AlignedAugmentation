import os
import sys
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.datasets.coco_retrieval import build_coco_dataset, CocoRetrievalDataset
from src.datasets.collate import coco_retrieval_collate_fn
from src.models.simple_autoencoder import SimpleConvAutoencoder


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def unnormalize(img: torch.Tensor) -> torch.Tensor:
    return img * IMAGENET_STD.to(img.device) + IMAGENET_MEAN.to(img.device)


def build_single_caption_subset(dataset: CocoRetrievalDataset) -> CocoRetrievalDataset:
    subset = dataset
    subset._pairs = [(img_id, 0) for img_id in subset.image_ids]
    return subset


def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    """
    img: [3,H,W] in [0,1]
    """
    img = torch.clamp(img, 0.0, 1.0)
    np_img = (img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(np_img)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    images_root = "data/coco/images/val2017"
    captions_json = "data/coco/annotations/captions_val2017.json"
    instances_json = "data/coco/annotations/instances_val2017.json"

    dataset = build_coco_dataset(
        images_root=images_root,
        captions_json_path=captions_json,
        instances_json_path=instances_json,
        num_categories=91,
    )
    dataset = build_single_caption_subset(dataset)

    # Use only a subset for speed
    max_images = 1000
    if len(dataset.image_ids) > max_images:
        dataset.image_ids = dataset.image_ids[:max_images]
        dataset._pairs = [(img_id, 0) for img_id in dataset.image_ids]

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        collate_fn=coco_retrieval_collate_fn,
    )

    # Load autoencoder
    model = SimpleConvAutoencoder(latent_dim=64).to(device)
    ckpt_path = "checkpoints/simple_autoencoder_coco_val2017.pt"
    print(f"Loading autoencoder from {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Collect a tensor of images
    all_imgs: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting images"):
            images = batch["image"].to(device)   # normalized
            imgs_unnorm = torch.clamp(unnormalize(images), 0.0, 1.0)
            all_imgs.append(imgs_unnorm.cpu())
            if sum(x.size(0) for x in all_imgs) >= max_images:
                break

    imgs = torch.cat(all_imgs, dim=0)[:max_images]  # [N,3,224,224]
    N = imgs.size(0)
    print(f"Collected {N} images")

    # Encode all to latents
    with torch.no_grad():
        latents = []
        for i in range(0, N, 64):
            batch = imgs[i:i+64].to(device)
            z = model.encode(batch)    # [B, C, 14, 14]
            latents.append(z.cpu())
        latents = torch.cat(latents, dim=0)  # [N, C, 14, 14]

    # Choose random pairs and mix
    np.random.seed(123)
    num_examples = 8
    indices_a = np.random.choice(N, size=num_examples, replace=False)
    indices_b = np.random.choice(N, size=num_examples, replace=False)

    os.makedirs("viz_autoencoder", exist_ok=True)

    with torch.no_grad():
        for idx, (ia, ib) in enumerate(zip(indices_a, indices_b)):
            img_a = imgs[ia].to(device)      # [3,224,224]
            img_b = imgs[ib].to(device)
            z_a = latents[ia].to(device)     # [C,14,14]
            z_b = latents[ib].to(device)

            # Choose a mixing coefficient alpha
            alpha = np.random.uniform(0.3, 0.7)

            z_mix = alpha * z_a + (1.0 - alpha) * z_b

            # Decode
            rec_a = model.decode(z_a.unsqueeze(0))[0]      # [3,H,W]
            rec_b = model.decode(z_b.unsqueeze(0))[0]
            rec_mix = model.decode(z_mix.unsqueeze(0))[0]

            # Convert to PIL
            pil_a = tensor_to_pil(rec_a)
            pil_mix = tensor_to_pil(rec_mix)
            pil_b = tensor_to_pil(rec_b)

            # Make triptych figure
            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            axes[0].imshow(pil_a)
            axes[0].axis("off")
            axes[0].set_title(f"A (idx={ia})", fontsize=8)

            axes[1].imshow(pil_mix)
            axes[1].axis("off")
            axes[1].set_title(f"Mixed (α={alpha:.2f})", fontsize=8)

            axes[2].imshow(pil_b)
            axes[2].axis("off")
            axes[2].set_title(f"B (idx={ib})", fontsize=8)

            plt.tight_layout()
            out_path = f"viz_autoencoder/mixed_triplet_{idx}_ia{ia}_ib{ib}.png"
            plt.savefig(out_path)
            plt.close(fig)

            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

