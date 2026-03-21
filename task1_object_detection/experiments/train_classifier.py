"""
Train an image classifier on cropped products to test classification feasibility.
Uses timm (available in sandbox) with a pretrained backbone.

Usage:
    uv run python task1_object_detection/experiments/train_classifier.py
    uv run python task1_object_detection/experiments/train_classifier.py --model efficientnet_b0 --epochs 20
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

TASK_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = TASK_ROOT / "output" / "classifier_dataset"


class ProductCropDataset(Dataset):
    """Load cropped product images from directory structure: split/category_id/*.jpg"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        # Collect all category directories
        cat_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()],
                          key=lambda d: int(d.name))

        for idx, cat_dir in enumerate(cat_dirs):
            cat_id = int(cat_dir.name)
            self.class_to_idx[cat_id] = idx
            self.idx_to_class[idx] = cat_id

            for img_path in cat_dir.glob("*.jpg"):
                self.samples.append((str(img_path), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def train(
    model_name: str = "efficientnet_b0",
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    img_size: int = 224,
    experiment_name: str = "product-classifier",
):
    import timm

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Datasets
    train_dataset = ProductCropDataset(DATASET_DIR / "train", transform=train_transform)
    val_dataset = ProductCropDataset(DATASET_DIR / "val", transform=val_transform)

    num_classes = len(train_dataset.class_to_idx)
    print(f"Train: {len(train_dataset)} samples, {num_classes} classes")
    print(f"Val:   {len(val_dataset)} samples, {len(val_dataset.class_to_idx)} classes")

    # Weighted sampler for class imbalance
    labels = [s[1] for s in train_dataset.samples]
    class_counts = Counter(labels)
    weights = [1.0 / class_counts[l] for l in labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

    # Model
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    mlflow.set_experiment(experiment_name)

    best_val_acc = 0
    best_top5_acc = 0

    with mlflow.start_run(run_name=f"{model_name}_e{epochs}"):
        mlflow.log_params({
            "model": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "img_size": img_size,
            "num_classes": num_classes,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
        })

        for epoch in range(1, epochs + 1):
            # Train
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                train_correct += (outputs.argmax(1) == labels).sum().item()
                train_total += images.size(0)

            scheduler.step()
            train_acc = train_correct / train_total

            # Validate
            model.eval()
            val_loss = 0
            val_correct = 0
            val_top5_correct = 0
            val_total = 0
            per_class_correct = Counter()
            per_class_total = Counter()

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    val_correct += (outputs.argmax(1) == labels).sum().item()
                    val_total += images.size(0)

                    # Top-5 accuracy
                    _, top5_preds = outputs.topk(5, dim=1)
                    for i, label in enumerate(labels):
                        if label in top5_preds[i]:
                            val_top5_correct += 1
                        # Per-class stats
                        per_class_total[label.item()] += 1
                        if outputs.argmax(1)[i] == label:
                            per_class_correct[label.item()] += 1

            val_acc = val_correct / val_total
            val_top5_acc = val_top5_correct / val_total

            mlflow.log_metrics({
                "train_loss": train_loss / train_total,
                "train_acc": train_acc,
                "val_loss": val_loss / val_total,
                "val_acc": val_acc,
                "val_top5_acc": val_top5_acc,
            }, step=epoch)

            print(f"Epoch {epoch}/{epochs} | "
                  f"Train acc: {train_acc:.3f} | "
                  f"Val acc: {val_acc:.3f} | "
                  f"Val top-5: {val_top5_acc:.3f} | "
                  f"Val loss: {val_loss/val_total:.3f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_top5_acc = val_top5_acc

                # Save best model
                model_dir = TASK_ROOT / "output" / "models" / "classifier"
                model_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), model_dir / "best_classifier.pt")

        # Final per-class analysis
        per_class_acc = {}
        for cls_idx in per_class_total:
            cat_id = val_dataset.idx_to_class.get(cls_idx, cls_idx)
            acc = per_class_correct[cls_idx] / per_class_total[cls_idx]
            per_class_acc[cat_id] = {
                "accuracy": round(acc, 4),
                "correct": per_class_correct[cls_idx],
                "total": per_class_total[cls_idx],
            }

        # Sort by accuracy
        sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1]["accuracy"])
        zero_acc = [c for c in sorted_classes if c[1]["accuracy"] == 0]
        perfect_acc = [c for c in sorted_classes if c[1]["accuracy"] == 1.0]

        mlflow.log_metrics({
            "best_val_acc": best_val_acc,
            "best_val_top5_acc": best_top5_acc,
            "zero_acc_classes": len(zero_acc),
            "perfect_acc_classes": len(perfect_acc),
        })

        # Save per-class report
        report_path = TASK_ROOT / "output" / "analysis" / "classifier_per_class.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(per_class_acc, f, indent=2)
        mlflow.log_artifact(str(report_path))

        print(f"\n{'='*60}")
        print("CLASSIFIER RESULTS")
        print(f"{'='*60}")
        print(f"  Best val accuracy:  {best_val_acc:.3f}")
        print(f"  Best val top-5 acc: {best_top5_acc:.3f}")
        print(f"  Zero-acc classes:   {len(zero_acc)} / {len(per_class_acc)}")
        print(f"  Perfect classes:    {len(perfect_acc)} / {len(per_class_acc)}")
        print(f"\n  Worst 10 classes (with >3 samples):")
        for cat_id, info in sorted_classes:
            if info["total"] >= 3:
                print(f"    cat_id={cat_id}: {info['correct']}/{info['total']} = {info['accuracy']:.3f}")
                if sum(1 for c, i in sorted_classes if i["total"] >= 3 and i["accuracy"] <= info["accuracy"]) > 10:
                    break
        print(f"\n  Best 10 classes:")
        for cat_id, info in sorted_classes[-10:][::-1]:
            print(f"    cat_id={cat_id}: {info['correct']}/{info['total']} = {info['accuracy']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()

    train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size,
    )
