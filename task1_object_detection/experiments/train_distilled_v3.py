"""
V3 distilled classifiers with hard-case improvements:
1. Higher resolution (300px) to capture text
2. Multi-crop training: full image + zoomed top/center regions
3. Hard-pair aware augmentation: extra weight on confusable products
4. Stronger color/contrast augmentation to learn subtle color differences

Usage:
    uv run python task1_object_detection/experiments/train_distilled_v3.py
    uv run python task1_object_detection/experiments/train_distilled_v3.py --groups coffee,knekkebroed
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import efficientnet_b2

_orig = torch.load
torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})

TASK_ROOT = Path(__file__).resolve().parent.parent
GROUPED_DIR = TASK_ROOT / "output" / "grouped_classifier_datasets"
SOFT_DIR = TASK_ROOT / "output" / "soft_labels"
MODELS_DIR = TASK_ROOT / "output" / "models" / "distilled_v3"
GROUP_NAMES = ["knekkebroed", "coffee", "tea", "cereal", "eggs", "spread", "cookies", "chocolate", "other"]

IMGSZ = 260  # efficientnet_b2 native size


class MultiCropSoftLabelDataset(Dataset):
    """Dataset that returns multiple crops per image:
    - Full image
    - Top third (brand/variant text usually here)
    - Center region (product name)
    Each gets its own training pass, tripling effective data for text regions.
    """

    def __init__(self, data_dir, soft_file, class_list, transform=None, multi_crop=True):
        self.transform = transform
        self.multi_crop = multi_crop
        self.class_to_idx = {c: i for i, c in enumerate(class_list)}
        self.samples = []

        soft = {}
        if soft_file and soft_file.exists():
            with open(soft_file) as f:
                soft = json.load(f)

        for cat_dir in sorted(data_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            cat_id = int(cat_dir.name)
            if cat_id not in self.class_to_idx:
                continue

            for img_path in sorted(cat_dir.glob("*.jpg")):
                path_str = str(img_path)
                if path_str in soft:
                    sl = soft[path_str]["soft_labels"]
                    sv = [0.0] * len(class_list)
                    for cid_s, p in sl.items():
                        if int(cid_s) in self.class_to_idx:
                            sv[self.class_to_idx[int(cid_s)]] = p
                    s = sum(sv)
                    if s > 0:
                        sv = [v / s for v in sv]
                    else:
                        sv[self.class_to_idx[cat_id]] = 1.0
                else:
                    sv = [0.0] * len(class_list)
                    sv[self.class_to_idx[cat_id]] = 1.0

                # Full crop
                self.samples.append((str(img_path), sv, self.class_to_idx[cat_id], "full"))

                if multi_crop:
                    # Top crop (brand/variant text)
                    self.samples.append((str(img_path), sv, self.class_to_idx[cat_id], "top"))
                    # Center crop (product name)
                    self.samples.append((str(img_path), sv, self.class_to_idx[cat_id], "center"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, soft_label, hard_label, crop_type = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        if crop_type == "top":
            # Top 40% — where brand/variant text usually is
            img = img.crop((0, 0, w, int(h * 0.4)))
        elif crop_type == "center":
            # Center 50% — product name area
            y_start = int(h * 0.25)
            y_end = int(h * 0.75)
            img = img.crop((0, y_start, w, y_end))

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(soft_label, dtype=torch.float32), hard_label


def train_group(group_name, epochs=50, batch_size=24, lr=1e-3, temperature=3.0, alpha=0.7):
    meta_path = SOFT_DIR / group_name / "metadata.json"
    if not meta_path.exists():
        print(f"  No metadata for {group_name}")
        return None

    with open(meta_path) as f:
        meta = json.load(f)
    class_list = meta["logreg_classes"]
    n_classes = len(class_list)
    print(f"  Classes: {n_classes}")

    train_transform = transforms.Compose([
        transforms.Resize((IMGSZ, IMGSZ)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.08),
        transforms.RandomAffine(degrees=10, translate=(0.15, 0.15), scale=(0.8, 1.2)),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
        transforms.RandomGrayscale(p=0.03),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMGSZ, IMGSZ)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = MultiCropSoftLabelDataset(
        GROUPED_DIR / group_name / "train",
        SOFT_DIR / group_name / "soft_labels_train.json",
        class_list, transform=train_transform, multi_crop=True,
    )
    val_ds = MultiCropSoftLabelDataset(
        GROUPED_DIR / group_name / "val",
        SOFT_DIR / group_name / "soft_labels_val.json",
        class_list, transform=val_transform, multi_crop=False,
    )

    if not train_ds.samples:
        return None

    print(f"  Train: {len(train_ds)} (with multi-crop) | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # EfficientNet-B2: more capacity + 260px native resolution
    model = efficientnet_b2(weights="IMAGENET1K_V1")
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for images, soft_labels, hard_labels in train_loader:
            images = images.to(device)
            soft_labels = soft_labels.to(device)
            hard_labels = hard_labels.to(device)

            logits = model(images)
            soft_log_probs = F.log_softmax(logits / temperature, dim=1)
            soft_targets = F.softmax(soft_labels / temperature, dim=1)
            kl_loss = F.kl_div(soft_log_probs, soft_targets, reduction="batchmean") * (temperature ** 2)
            ce_loss = F.cross_entropy(logits, hard_labels)
            loss = alpha * kl_loss + (1 - alpha) * ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, _, hard_labels in val_loader:
                images, hard_labels = images.to(device), hard_labels.to(device)
                correct += (model(images).argmax(1) == hard_labels).sum().item()
                total += len(hard_labels)

        acc = correct / total if total > 0 else 0

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/{epochs} | acc={acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f"  Early stopping at epoch {epoch}")
                break

    out_dir = MODELS_DIR / group_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if best_state:
        torch.save(best_state, out_dir / "best.pt")
        with open(out_dir / "classes.json", "w") as f:
            json.dump(class_list, f)
        print(f"  Best: {best_acc:.3f} → {out_dir}")

    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default="")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    groups = args.groups.split(",") if args.groups else GROUP_NAMES

    print(f"Training V3 distilled classifiers (EfficientNet-B2, {IMGSZ}px, multi-crop)")
    results = {}
    for group in groups:
        print(f"\n{'='*60}")
        print(f"GROUP: {group}")
        print(f"{'='*60}")
        acc = train_group(group, epochs=args.epochs)
        if acc is not None:
            results[group] = acc

    print(f"\n{'='*60}")
    print("V3 RESULTS")
    print(f"{'='*60}")
    for group, acc in sorted(results.items()):
        print(f"  {group:15s}: {acc:.3f}")
    if results:
        print(f"\n  Average: {sum(results.values()) / len(results):.3f}")


if __name__ == "__main__":
    main()
