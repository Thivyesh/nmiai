"""
NM i AI - NorgesGruppen Object Detection
Submission entry point: python run.py --input /data/images --output /output/predictions.json

Pipeline:
1. YOLOv8m 9-class detector → finds products + super-category (mAP@0.5 ~0.90)
2. Routes each detection to its group's MobileNetV3 classifier (avg 94.7% top-1)
   - Trained with knowledge distillation from YOLO backbone + OCR soft labels
3. Outputs category_id from the group classifier

Sandbox: Python 3.11, NVIDIA L4 (24GB), 8GB RAM, 300s timeout, no network.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
# ultralytics 8.1.0 .pt files need weights_only=False with torch >=2.6
_original_torch_load = torch.load
torch.load = lambda *a, **kw: _original_torch_load(*a, **{**kw, "weights_only": False})
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from PIL import Image
from ultralytics import YOLO

# 9 super-category names matching the detector's class order
GROUP_NAMES = [
    "knekkebroed", "coffee", "tea", "cereal",
    "eggs", "spread", "cookies", "chocolate", "other",
]


def load_detector(device):
    """Load the 9-class superclass detector."""
    model_path = Path(__file__).parent / "detector.pt"
    model = YOLO(str(model_path))
    return model


def load_classifiers(device):
    """Load per-group distilled MobileNetV3 classifiers."""
    classifiers = {}
    classifier_dir = Path(__file__).parent / "classifiers"

    for group_name in GROUP_NAMES:
        model_path = classifier_dir / group_name / "best.pt"
        classes_path = classifier_dir / group_name / "classes.json"

        if not model_path.exists() or not classes_path.exists():
            continue

        with open(classes_path) as f:
            classes = json.load(f)

        n_classes = len(classes)
        model = mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        model = model.to(device)
        model.eval()

        classifiers[group_name] = {
            "model": model,
            "classes": classes,
        }

    return classifiers


def classify_crops_batched(crops, group_indices, classifiers, transform, device, batch_size=64):
    """Classify all crops in batches, grouped by super-category.

    Returns list of (category_id, confidence) for each crop.
    """
    results = [None] * len(crops)

    # Group crops by their super-category
    group_to_indices = {}
    for i, group_name in enumerate(group_indices):
        group_to_indices.setdefault(group_name, []).append(i)

    for group_name, indices in group_to_indices.items():
        if group_name not in classifiers:
            for i in indices:
                results[i] = (0, 0.5)
            continue

        clf = classifiers[group_name]
        model = clf["model"]
        classes = clf["classes"]

        # Process in batches
        for batch_start in range(0, len(indices), batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            batch_tensors = torch.stack([
                transform(crops[i]) for i in batch_indices
            ]).to(device)

            with torch.no_grad():
                logits = model(batch_tensors)
                probs = torch.softmax(logits, dim=1)
                pred_indices = probs.argmax(dim=1)
                confidences = probs.gather(1, pred_indices.unsqueeze(1)).squeeze(1)

            for j, idx in enumerate(batch_indices):
                results[idx] = (classes[pred_indices[j].item()], confidences[j].item())

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load models
    print("Loading 9-class detector...")
    detector = load_detector(device)

    print("Loading group classifiers...")
    classifiers = load_classifiers(device)
    print(f"  Loaded: {list(classifiers.keys())}")

    cls_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Process images
    predictions = []
    input_dir = Path(args.input)
    image_files = sorted(
        f for f in input_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    print(f"Processing {len(image_files)} images...")

    for img_idx, img_path in enumerate(image_files):
        image_id = int(img_path.stem.split("_")[-1])

        # Step 1: Detect products + super-category
        with torch.no_grad():
            results = detector(str(img_path), device=device, verbose=False, conf=0.15)

        if not results or results[0].boxes is None:
            continue

        img = Image.open(img_path).convert("RGB")
        boxes = results[0].boxes

        # Collect all crops for batched classification
        crops = []
        group_names = []
        det_data = []

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            det_conf = float(boxes.conf[i].item())
            det_cls = int(boxes.cls[i].item())
            w, h = x2 - x1, y2 - y1
            if w < 10 or h < 10:
                continue

            group_name = GROUP_NAMES[det_cls] if det_cls < len(GROUP_NAMES) else "other"
            pad_x, pad_y = w * 0.05, h * 0.05
            crop = img.crop((
                max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)),
                min(img.width, int(x2 + pad_x)), min(img.height, int(y2 + pad_y)),
            ))
            crops.append(crop)
            group_names.append(group_name)
            det_data.append((x1, y1, w, h, det_conf))

        # Step 2: Batched classification by group
        if crops:
            cls_results = classify_crops_batched(
                crops, group_names, classifiers, cls_transform, device
            )
            for j, (cat_id, cls_conf) in enumerate(cls_results):
                x1, y1, w, h, det_conf = det_data[j]
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(cat_id),
                    "bbox": [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)],
                    "score": round(det_conf * cls_conf, 3),
                })

        if (img_idx + 1) % 50 == 0:
            print(f"  {img_idx + 1}/{len(image_files)} images, {len(predictions)} predictions")

    # Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
