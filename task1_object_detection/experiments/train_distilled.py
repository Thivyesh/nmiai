"""
Step 2: Train YOLO-cls per group with knowledge distillation from LogReg+OCR soft labels.

Instead of hard one-hot labels, uses soft probability distributions from
the backbone+OCR LogReg classifier. This prevents overfitting and teaches
the YOLO-cls what the LogReg+OCR "knows" about similar products.

Usage:
    uv run python task1_object_detection/experiments/train_distilled.py
    uv run python task1_object_detection/experiments/train_distilled.py --groups chocolate,tea
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, "weights_only": False})

TASK_ROOT = Path(__file__).resolve().parent.parent
SOFT_LABELS_DIR = TASK_ROOT / "output" / "soft_labels"
GROUPED_DIR = TASK_ROOT / "output" / "grouped_classifier_datasets"
MODELS_DIR = TASK_ROOT / "output" / "models" / "distilled"


class SoftLabelDataset(Dataset):
    """Dataset that returns images with soft probability labels."""

    def __init__(self, data_dir, soft_labels_file, class_list, transform=None):
        self.transform = transform
        self.class_list = class_list  # ordered list of cat_ids
        self.class_to_idx = {c: i for i, c in enumerate(class_list)}
        self.samples = []

        # Load soft labels
        soft_labels = {}
        if soft_labels_file and soft_labels_file.exists():
            with open(soft_labels_file) as f:
                soft_labels = json.load(f)

        # Collect all images
        for cat_dir in sorted(data_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            cat_id = int(cat_dir.name)
            if cat_id not in self.class_to_idx:
                continue

            for img_path in sorted(cat_dir.glob("*.jpg")):
                path_str = str(img_path)
                if path_str in soft_labels:
                    # Use soft label
                    sl = soft_labels[path_str]["soft_labels"]
                    soft_vec = np.zeros(len(class_list), dtype=np.float32)
                    for cid_str, prob in sl.items():
                        cid = int(cid_str)
                        if cid in self.class_to_idx:
                            soft_vec[self.class_to_idx[cid]] = prob
                    # Normalize
                    if soft_vec.sum() > 0:
                        soft_vec /= soft_vec.sum()
                    else:
                        soft_vec[self.class_to_idx[cat_id]] = 1.0
                else:
                    # Hard label fallback
                    soft_vec = np.zeros(len(class_list), dtype=np.float32)
                    soft_vec[self.class_to_idx[cat_id]] = 1.0

                self.samples.append((str(img_path), soft_vec, self.class_to_idx[cat_id]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, soft_label, hard_label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.from_numpy(soft_label), hard_label


def train_group(group_name, epochs=30, imgsz=224, batch_size=32, lr=1e-3, temperature=3.0, alpha=0.7):
    """Train a small classifier for one group using distillation."""

    group_data_dir = GROUPED_DIR / group_name
    soft_dir = SOFT_LABELS_DIR / group_name

    if not group_data_dir.exists():
        print(f"  No dataset for {group_name}")
        return None

    # Load metadata
    meta_path = soft_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        class_list = meta["logreg_classes"]
    else:
        # Fallback: use directory names
        class_list = sorted(int(d.name) for d in (group_data_dir / "train").iterdir() if d.is_dir())

    n_classes = len(class_list)
    print(f"  Classes: {n_classes}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
        transforms.RandomAffine(degrees=10, translate=(0.15, 0.15), scale=(0.8, 1.2)),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Datasets
    train_ds = SoftLabelDataset(
        group_data_dir / "train",
        soft_dir / "soft_labels_train.json",
        class_list,
        transform=train_transform,
    )
    val_ds = SoftLabelDataset(
        group_data_dir / "val",
        soft_dir / "soft_labels_val.json",
        class_list,
        transform=val_transform,
    )

    if len(train_ds) == 0:
        print(f"  No training samples")
        return None

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Simple CNN classifier (small, won't overfit as much)
    # Using a mobilenet-style model via torchvision
    from torchvision.models import mobilenet_v3_small
    model = mobilenet_v3_small(weights="IMAGENET1K_V1")
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0
    best_state = None
    patience_counter = 0
    patience = 10

    for epoch in range(1, epochs + 1):
        # Train with distillation loss
        model.train()
        total_loss = 0
        n_batches = 0

        for images, soft_labels, hard_labels in train_loader:
            images = images.to(device)
            soft_labels = soft_labels.to(device)
            hard_labels = hard_labels.to(device)

            logits = model(images)

            # Distillation loss: KL divergence between soft targets and predictions
            soft_log_probs = F.log_softmax(logits / temperature, dim=1)
            soft_targets = F.softmax(soft_labels / temperature, dim=1) if soft_labels.sum() > 0 else soft_labels
            kl_loss = F.kl_div(soft_log_probs, soft_targets, reduction="batchmean") * (temperature ** 2)

            # Hard label loss
            ce_loss = F.cross_entropy(logits, hard_labels)

            # Combined: alpha * soft + (1-alpha) * hard
            loss = alpha * kl_loss + (1 - alpha) * ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate
        model.eval()
        correct = 0
        top5_correct = 0
        total = 0

        with torch.no_grad():
            for images, _, hard_labels in val_loader:
                images = images.to(device)
                hard_labels = hard_labels.to(device)
                logits = model(images)
                preds = logits.argmax(dim=1)
                correct += (preds == hard_labels).sum().item()
                total += len(hard_labels)

                _, top5_preds = logits.topk(min(5, n_classes), dim=1)
                for i, label in enumerate(hard_labels):
                    if label in top5_preds[i]:
                        top5_correct += 1

        acc = correct / total if total > 0 else 0
        top5_acc = top5_correct / total if total > 0 else 0
        avg_loss = total_loss / n_batches if n_batches > 0 else 0

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/{epochs} | loss={avg_loss:.3f} | top-1={acc:.3f} | top-5={top5_acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Save best model
    out_dir = MODELS_DIR / group_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if best_state:
        torch.save(best_state, out_dir / "best.pt")
        # Save class mapping
        with open(out_dir / "classes.json", "w") as f:
            json.dump(class_list, f)
        print(f"  Best top-1: {best_acc:.3f} | Saved to {out_dir}")

    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default="", help="Comma-separated groups to train (default: all)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    args = parser.parse_args()

    if args.groups:
        group_list = args.groups.split(",")
    else:
        group_list = sorted(d.name for d in GROUPED_DIR.iterdir() if d.is_dir())

    print(f"Training distilled classifiers for: {group_list}")
    print(f"Temperature={args.temperature}, Alpha={args.alpha}")

    results = {}
    for group in group_list:
        print(f"\n{'='*60}")
        print(f"GROUP: {group}")
        print(f"{'='*60}")
        acc = train_group(group, epochs=args.epochs, temperature=args.temperature, alpha=args.alpha)
        if acc is not None:
            results[group] = acc

    print(f"\n{'='*60}")
    print("DISTILLED CLASSIFIER RESULTS")
    print(f"{'='*60}")
    for group, acc in sorted(results.items()):
        print(f"  {group:15s}: {acc:.3f}")

    if results:
        avg = sum(results.values()) / len(results)
        print(f"\n  Average: {avg:.3f}")


if __name__ == "__main__":
    main()
