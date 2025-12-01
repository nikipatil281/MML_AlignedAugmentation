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

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.datasets.coco_retrieval import build_coco_dataset, CocoRetrievalDataset
from src.datasets.collate import coco_retrieval_collate_fn
from src.models.retrieval_tokens_model import ImageTextRetrievalTokensModel
from src.models.pairing import PairingNetwork


def build_single_caption_subset(dataset: CocoRetrievalDataset) -> CocoRetrievalDataset:
    subset = dataset
    subset._pairs = [(img_id, 0) for img_id in subset.image_ids]
    return subset


def load_image(dataset: CocoRetrievalDataset, image_id: int):
    img_rel_path = dataset.image_id_to_filename[image_id]
    img_path = os.path.join(dataset.images_root, img_rel_path)
    return Image.open(img_path).convert("RGB")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    embed_dim = 256
    model = ImageTextRetrievalTokensModel(
        img_embed_dim=embed_dim,
        txt_embed_dim=embed_dim,
        txt_model_name="bert-base-uncased",
    ).to(device)

    pairing_net = PairingNetwork(
        embed_dim=embed_dim, pairing_dim=embed_dim, hidden_dim=512
    ).to(device)

    # Load final full-COCO checkpoints (same ones used in training)
    model_ckpt = "checkpoints/retrieval_tokenmix_align_full_coco_model.pt"
    pairing_ckpt = "checkpoints/retrieval_tokenmix_align_full_coco_pairing.pt"

    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    pairing_net.load_state_dict(torch.load(pairing_ckpt, map_location=device))

    model.eval()
    pairing_net.eval()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    all_img_globals: List[torch.Tensor] = []
    all_txt_globals: List[torch.Tensor] = []
    all_pair_embs: List[torch.Tensor] = []
    all_image_ids: List[int] = list(dataset.image_ids)
    all_captions: List[str] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding for pairing"):
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
            pair_emb = pairing_net(img_global, txt_global)

            all_img_globals.append(img_global.cpu())
            all_txt_globals.append(txt_global.cpu())
            all_pair_embs.append(pair_emb.cpu())
            all_captions.extend(captions)

    pair_mat = torch.cat(all_pair_embs, dim=0).numpy()  # [N,D]
    N = pair_mat.shape[0]

    # cosine similarity matrix
    sim = pair_mat @ pair_mat.T

    np.random.seed(0)
    num_examples = 4
    anchor_indices = np.random.choice(N, size=num_examples, replace=False)

    os.makedirs("viz", exist_ok=True)

    for anchor_idx in anchor_indices:
        sims = sim[anchor_idx]
        sims[anchor_idx] = -1e9  # ignore self
        topk = np.argsort(-sims)[:5]

        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        axes = axes.flatten()

        anchor_img_id = all_image_ids[anchor_idx]
        anchor_img = load_image(dataset, anchor_img_id)
        anchor_cap = all_captions[anchor_idx]

        axes[0].imshow(anchor_img)
        axes[0].axis("off")
        axes[0].set_title(f"Anchor {anchor_img_id}\n{anchor_cap}", fontsize=8)

        for ax, idx in zip(axes[1:], topk):
            img_id = all_image_ids[idx]
            img = load_image(dataset, img_id)
            cap = all_captions[idx]
            score = sims[idx]

            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"id {img_id}, sim={score:.3f}\n{cap}", fontsize=8)

        plt.tight_layout()
        out_path = f"viz/pairing_neighbors_anchor_{anchor_img_id}.png"
        plt.savefig(out_path)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

