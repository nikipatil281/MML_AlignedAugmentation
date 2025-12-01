import os
import sys
from typing import List

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.datasets.coco_retrieval import build_coco_dataset, CocoRetrievalDataset
from src.datasets.collate import coco_retrieval_collate_fn
from src.models.retrieval_tokens_model import ImageTextRetrievalTokensModel


def build_single_caption_subset(dataset: CocoRetrievalDataset) -> CocoRetrievalDataset:
    subset = dataset
    subset._pairs = [(img_id, 0) for img_id in subset.image_ids]
    return subset


def load_image(dataset: CocoRetrievalDataset, image_id: int) -> Image.Image:
    img_rel_path = dataset.image_id_to_filename[image_id]
    img_path = os.path.join(dataset.images_root, img_rel_path)
    return Image.open(img_path).convert("RGB")


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

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        collate_fn=coco_retrieval_collate_fn,
    )

    # --- model ---
    embed_dim = 256
    model = ImageTextRetrievalTokensModel(
        img_embed_dim=embed_dim,
        txt_embed_dim=embed_dim,
        txt_model_name="bert-base-uncased",
    ).to(device)

    ckpt_path = "checkpoints/retrieval_tokenmix_align_full_coco_model.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # --- collect embeddings and captions ---
    all_img_embeds: List[torch.Tensor] = []
    all_txt_embeds: List[torch.Tensor] = []
    all_captions: List[str] = []
    all_image_ids: List[int] = list(dataset.image_ids)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding val2017"):
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

            _, _, img_global, txt_global = model(images, input_ids, attention_mask)

            all_img_embeds.append(img_global.cpu())
            all_txt_embeds.append(txt_global.cpu())
            all_captions.extend(captions)

    img_mat = torch.cat(all_img_embeds, dim=0).numpy()  # [N,D]
    txt_mat = torch.cat(all_txt_embeds, dim=0).numpy()  # [N,D]

    sim_i2t = img_mat @ txt_mat.T  # [N,N]

    # --- pick some random indices to visualize ---
    np.random.seed(42)
    N = img_mat.shape[0]
    num_examples = 5
    indices = np.random.choice(N, size=num_examples, replace=False)

    os.makedirs("viz", exist_ok=True)

    for idx in indices:
        image_id = all_image_ids[idx]
        gt_caption = all_captions[idx]

        sims = sim_i2t[idx]
        topk = np.argsort(-sims)[:5]

        topk_caps = [all_captions[j] for j in topk]
        topk_scores = [float(sims[j]) for j in topk]

        img = load_image(dataset, image_id)

        # plot
        fig = plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Image ID: {image_id}")

        plt.subplot(1, 2, 2)
        lines = [f"GT: {gt_caption}\n", "\nTop-5 retrieved:\n"]
        for rank, (cap, score) in enumerate(zip(topk_caps, topk_scores), start=1):
            lines.append(f"{rank}. ({score:.3f}) {cap}\n")
        plt.text(0.01, 0.99, "".join(lines), va="top", wrap=True)
        plt.axis("off")

        out_path = f"viz/retrieval_example_{image_id}.png"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)

        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

