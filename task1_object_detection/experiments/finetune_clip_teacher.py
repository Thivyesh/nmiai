"""
Fine-tune CLIP on product crops + names for better classification teacher.

Strategy:
- Use openai/clip-vit-base-patch32
- Contrastive learning: match crops to their product names
- Freeze most of vision encoder, fine-tune last layers + projection
- After fine-tuning: embed all product names, use as teacher
  (crop → CLIP embedding → cosine sim to product names → soft labels)

Usage:
    uv run python task1_object_detection/experiments/finetune_clip_teacher.py
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor

TASK_ROOT = Path(__file__).resolve().parent.parent
CLASSIFIER_DIR = TASK_ROOT / "output" / "classifier_dataset"
ANN_FILE = TASK_ROOT / "data" / "coco_dataset" / "train" / "annotations.json"
OUTPUT_DIR = TASK_ROOT / "output" / "clip_teacher"

SUPERS = {
    "knekkebroed": ["knekkebrød", "knekke", "flatbrød", "wasa", "sigdal", "leksands", "ryvita", "korni"],
    "coffee": ["kaffe", "coffee", "espresso", "nescafe", "evergood", "friele", "dolce gusto", "cappuccino", "kapsel", "tassimo", "zoegas"],
    "tea": [" te ", "tea", "twinings", "lipton", "pukka", "urtete"],
    "cereal": ["frokost", "havre", "müsli", "granola", "corn flakes", "cheerios", "cruesli", "puffet", "fras", "weetabix", "fitness"],
    "eggs": ["egg"],
    "spread": ["smør", "bremykt", "brelett", "margarin", "olivero", "meierismør"],
    "cookies": ["kjeks", "cookie", "grissini", "surdeig"],
    "chocolate": ["sjokolade", "nugatti", "regia", "cocoa", "kakao"],
}
GROUP_NAMES = list(SUPERS.keys()) + ["other"]


class ProductCropDataset(Dataset):
    """Pairs of (crop image, product name) for contrastive learning."""

    def __init__(self, split_dir, cat_map, processor, max_per_class=30):
        self.processor = processor
        self.samples = []  # (img_path, product_name, cat_id)

        for cat_dir in sorted(split_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            cat_id = int(cat_dir.name)
            name = cat_map.get(cat_id, "unknown product")
            crops = sorted(cat_dir.glob("*.jpg"))[:max_per_class]
            for crop_path in crops:
                self.samples.append((str(crop_path), name, cat_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, name, cat_id = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        return image, name, cat_id


def collate_fn(batch, processor):
    images, names, cat_ids = zip(*batch)
    inputs = processor(
        text=list(names),
        images=list(images),
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    return inputs, torch.tensor(cat_ids)


def finetune():
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)

    with open(ANN_FILE) as f:
        coco = json.load(f)
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}

    # Freeze most of vision encoder — only fine-tune last 2 layers + projection
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze: visual projection, text projection, last 2 vision layers
    for param in model.visual_projection.parameters():
        param.requires_grad = True
    for param in model.text_projection.parameters():
        param.requires_grad = True
    # Last 2 encoder layers
    for layer in model.vision_model.encoder.layers[-2:]:
        for param in layer.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M params")

    # Datasets
    train_ds = ProductCropDataset(CLASSIFIER_DIR / "train", cat_map, processor, max_per_class=30)
    val_ds = ProductCropDataset(CLASSIFIER_DIR / "val", cat_map, processor, max_per_class=15)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0,
                          collate_fn=lambda b: collate_fn(b, processor))
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0,
                        collate_fn=lambda b: collate_fn(b, processor))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-5, weight_decay=0.01
    )

    # Pre-compute all product name text embeddings for eval
    all_cat_ids = sorted(cat_map.keys())
    all_names = [f"a photo of {cat_map[cid]}" for cid in all_cat_ids]

    best_acc = 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, 11):
        # Train: contrastive loss (CLIP's native loss)
        model.train()
        total_loss = 0
        n_batches = 0

        for inputs, cat_ids in train_dl:
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs, return_loss=True)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        # Eval: for each val crop, find nearest product name
        model.eval()
        correct = 0
        total = 0

        # Encode all product names
        with torch.no_grad():
            text_features_list = []
            for i in range(0, len(all_names), 32):
                batch_names = all_names[i:i + 32]
                text_inputs = processor(text=batch_names, return_tensors="pt",
                                        padding=True, truncation=True).to(device)
                text_out = model.get_text_features(**text_inputs)
                if not isinstance(text_out, torch.Tensor):
                    text_out = text_out.pooler_output
                text_features_list.append(text_out / text_out.norm(dim=-1, keepdim=True))
            text_features = torch.cat(text_features_list, dim=0)  # [356, 512]

        with torch.no_grad():
            for inputs, cat_ids in val_dl:
                inputs = {k: v.to(device) for k, v in inputs.items()}

                img_out = model.get_image_features(pixel_values=inputs["pixel_values"])
                if not isinstance(img_out, torch.Tensor):
                    img_out = img_out.pooler_output
                img_features = img_out / img_out.norm(dim=-1, keepdim=True)

                # Cosine similarity to all product names
                sims = img_features @ text_features.T  # [B, 356]
                pred_indices = sims.argmax(dim=1)

                for j in range(len(cat_ids)):
                    pred_cat = all_cat_ids[pred_indices[j].item()]
                    if pred_cat == cat_ids[j].item():
                        correct += 1
                    total += 1

        acc = correct / total if total else 0
        print(f"Epoch {epoch}: loss={avg_loss:.4f} val_acc={acc:.3f} ({correct}/{total})")

        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(OUTPUT_DIR / "best_model")
            processor.save_pretrained(OUTPUT_DIR / "best_model")

            # Save text embeddings
            np.save(OUTPUT_DIR / "text_embeddings.npy", text_features.cpu().numpy())
            with open(OUTPUT_DIR / "cat_ids.json", "w") as f:
                json.dump(all_cat_ids, f)

    print(f"\nBest val accuracy: {best_acc:.3f}")
    print(f"Saved to {OUTPUT_DIR}")

    # Generate soft labels from fine-tuned CLIP
    print("\nGenerating soft labels from fine-tuned CLIP...")
    model.eval()
    text_features = torch.from_numpy(np.load(OUTPUT_DIR / "text_embeddings.npy")).to(device)

    for split in ["train", "val"]:
        split_dir = CLASSIFIER_DIR / split
        soft_labels = {}

        for cat_dir in sorted(split_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            cat_id = int(cat_dir.name)
            max_crops = 50 if split == "train" else 30

            for crop_path in sorted(cat_dir.glob("*.jpg"))[:max_crops]:
                image = Image.open(crop_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)

                with torch.no_grad():
                    img_out = model.get_image_features(**inputs)
                    if not isinstance(img_out, torch.Tensor):
                        img_out = img_out.pooler_output
                    img_feat = img_out / img_out.norm(dim=-1, keepdim=True)

                sims = (img_feat @ text_features.T).squeeze().cpu().numpy()
                # Convert to probabilities via softmax with temperature
                exp_sims = np.exp((sims - sims.max()) / 0.5)  # temperature=0.5
                probs = exp_sims / exp_sims.sum()

                probs_dict = {int(all_cat_ids[i]): round(float(probs[i]), 4)
                              for i in range(len(all_cat_ids)) if probs[i] > 0.001}

                soft_labels[str(crop_path)] = {
                    "true_label": cat_id,
                    "soft_labels": probs_dict,
                }

        with open(OUTPUT_DIR / f"soft_labels_{split}.json", "w") as f:
            json.dump(soft_labels, f)
        print(f"  {split}: {len(soft_labels)} crops")

    print("Done!")


if __name__ == "__main__":
    finetune()
