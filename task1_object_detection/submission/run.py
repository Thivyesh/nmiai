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


def classify_crop(crop, group_name, classifiers, transform, device):
    """Classify a crop using its group's distilled classifier."""
    if group_name not in classifiers:
        return 0, 0.5

    clf = classifiers[group_name]
    tensor = transform(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = clf["model"](tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

    return clf["classes"][pred_idx], confidence


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

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            det_conf = float(boxes.conf[i].item())
            det_cls = int(boxes.cls[i].item())

            w = x2 - x1
            h = y2 - y1
            if w < 10 or h < 10:
                continue

            # Get group name from detector class
            group_name = GROUP_NAMES[det_cls] if det_cls < len(GROUP_NAMES) else "other"

            # Step 2: Crop with padding and classify within group
            pad_x = w * 0.05
            pad_y = h * 0.05
            crop = img.crop((
                max(0, int(x1 - pad_x)),
                max(0, int(y1 - pad_y)),
                min(img.width, int(x2 + pad_x)),
                min(img.height, int(y2 + pad_y)),
            ))

            cat_id, cls_conf = classify_crop(
                crop, group_name, classifiers, cls_transform, device
            )

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
