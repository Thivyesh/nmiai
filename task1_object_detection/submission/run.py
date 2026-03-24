"""
NM i AI - NorgesGruppen Object Detection
python run.py --input /data/images --output /output/predictions.json

WBF ensemble of YOLOv8m (9-class) + YOLOv8x (9-class) detectors.
EfficientNet-b1 classifiers with score blending.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
_torch_load = torch.load
torch.load = lambda f, *a, **kw: _torch_load(f, *a, **{**kw, "weights_only": False})
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b1
from PIL import Image
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

GROUP_NAMES = [
    "knekkebroed", "coffee", "tea", "cereal",
    "eggs", "spread", "cookies", "chocolate", "other",
]

# --- Configuration ---
DET_CONF = 0.001         # Ultra-low: maximize recall, let WBF filter
DET_IOU = 0.6            # NMS IoU threshold
DET_IMGSZ = 1280         # High resolution
WBF_IOU_THR = 0.55       # WBF merge threshold
WBF_SKIP_THR = 0.001     # Don't skip any boxes
CROP_PADDING = 0.15      # 15% padding around crops (winner used this)
SCORE_BLEND = 0.5        # score = det * (blend + (1-blend) * cls)


def load_detectors(device):
    """Load both detector models."""
    base = Path(__file__).parent
    detectors = []

    det_m = base / "detector_m.pt"
    if det_m.exists():
        detectors.append({"model": YOLO(str(det_m)), "weight": 1, "name": "yolov8m"})

    det_x = base / "detector_x.pt"
    if det_x.exists():
        detectors.append({"model": YOLO(str(det_x)), "weight": 2, "name": "yolov8x"})

    # Fallback: single detector
    if not detectors:
        det = base / "detector.pt"
        if det.exists():
            detectors.append({"model": YOLO(str(det)), "weight": 1, "name": "detector"})

    return detectors


def load_classifiers(device):
    base = Path(__file__).parent
    weights = np.load(str(base / "classifiers.npy"))
    with open(base / "classifier_meta.json") as f:
        meta = json.load(f)

    classifiers = {}
    param_data = {}
    for i in range(len(meta["groups"])):
        gn = meta["groups"][i]
        pname = meta["param_names"][i]
        shape = meta["shapes"][i]
        offset = meta["offsets"][i]
        size = 1
        for s in shape:
            size *= s
        arr = weights[offset:offset + size].astype(np.float32).reshape(shape)
        param_data.setdefault(gn, {})[pname] = torch.from_numpy(arr)

    for gn in set(meta["groups"]):
        classes = meta["classes"][gn]
        n_classes = len(classes)
        model = efficientnet_b1(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
        model.load_state_dict(param_data[gn])
        model = model.to(device)
        model.eval()
        classifiers[gn] = {"model": model, "classes": classes}

    return classifiers


def run_detector(det, img_path, device):
    """Run a single detector, return normalized boxes + scores + labels."""
    with torch.no_grad():
        results = det["model"](str(img_path), device=device, verbose=False,
                                conf=DET_CONF, iou=DET_IOU, imgsz=DET_IMGSZ,
                                max_det=500)
    if not results or results[0].boxes is None:
        return [], [], []

    boxes = results[0].boxes
    orig_shape = results[0].orig_shape  # (H, W)
    oh, ow = orig_shape

    boxes_norm = []
    scores = []
    labels = []

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        conf = float(boxes.conf[i].item())
        cls_id = int(boxes.cls[i].item())

        # Normalize to [0, 1]
        boxes_norm.append([x1 / ow, y1 / oh, x2 / ow, y2 / oh])
        scores.append(conf)
        labels.append(cls_id)

    return boxes_norm, scores, labels


def ensemble_detect(detectors, img_path, device):
    """Run all detectors and merge with WBF."""
    all_boxes = []
    all_scores = []
    all_labels = []
    weights = []

    for det in detectors:
        boxes, scores, labels = run_detector(det, img_path, device)
        if boxes:
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            weights.append(det["weight"])

    if not all_boxes:
        return [], [], []

    # WBF merge
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels,
        weights=weights,
        iou_thr=WBF_IOU_THR,
        skip_box_thr=WBF_SKIP_THR,
    )

    return fused_boxes, fused_scores, fused_labels


def classify_crops_batched(crops, group_indices, classifiers, transform, device, batch_size=64):
    results = [None] * len(crops)
    group_to_indices = {}
    for i, gn in enumerate(group_indices):
        group_to_indices.setdefault(gn, []).append(i)

    for gn, indices in group_to_indices.items():
        if gn not in classifiers:
            for i in indices:
                results[i] = (0, 0.5)
            continue

        clf = classifiers[gn]
        model = clf["model"]
        classes = clf["classes"]

        for bs in range(0, len(indices), batch_size):
            bi = indices[bs:bs + batch_size]
            batch = torch.stack([transform(crops[i]) for i in bi]).to(device)
            with torch.no_grad():
                logits = model(batch)
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                confs = probs.gather(1, preds.unsqueeze(1)).squeeze(1)
            for j, idx in enumerate(bi):
                results[idx] = (classes[preds[j].item()], confs[j].item())

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading detectors...")
    detectors = load_detectors(device)
    print(f"  Loaded: {[d['name'] for d in detectors]}")

    print("Loading classifiers...")
    classifiers = load_classifiers(device)
    print(f"  Loaded: {list(classifiers.keys())}")

    cls_transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    predictions = []
    input_dir = Path(args.input)
    image_files = sorted(
        f for f in input_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    print(f"Processing {len(image_files)} images...")

    for img_idx, img_path in enumerate(image_files):
        image_id = int(img_path.stem.split("_")[-1])

        # Step 1: Ensemble detection
        fused_boxes, fused_scores, fused_labels = ensemble_detect(
            detectors, img_path, device
        )

        if len(fused_boxes) == 0:
            continue

        img = Image.open(img_path).convert("RGB")
        ow, oh = img.size

        # Collect crops
        crops = []
        group_names = []
        det_data = []

        for i in range(len(fused_boxes)):
            # Denormalize
            x1 = fused_boxes[i][0] * ow
            y1 = fused_boxes[i][1] * oh
            x2 = fused_boxes[i][2] * ow
            y2 = fused_boxes[i][3] * oh
            det_conf = float(fused_scores[i])
            det_cls = int(fused_labels[i])

            w, h = x2 - x1, y2 - y1
            if w < 5 or h < 5:
                continue

            gn = GROUP_NAMES[det_cls] if det_cls < len(GROUP_NAMES) else "other"

            # Crop with padding
            pad_x = w * CROP_PADDING
            pad_y = h * CROP_PADDING
            crop = img.crop((
                max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)),
                min(ow, int(x2 + pad_x)), min(oh, int(y2 + pad_y)),
            ))
            crops.append(crop)
            group_names.append(gn)
            det_data.append((x1, y1, w, h, det_conf))

        # Step 2: Classify
        if crops:
            cls_results = classify_crops_batched(
                crops, group_names, classifiers, cls_transform, device
            )
            for j, (cat_id, cls_conf) in enumerate(cls_results):
                x1, y1, w, h, det_conf = det_data[j]

                # Score blending: det * (0.5 + 0.5 * cls)
                # Preserves good detections even when classifier is uncertain
                score = det_conf * (SCORE_BLEND + (1 - SCORE_BLEND) * cls_conf)

                predictions.append({
                    "image_id": image_id,
                    "category_id": int(cat_id),
                    "bbox": [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)],
                    "score": round(score, 3),
                })

        if (img_idx + 1) % 50 == 0:
            print(f"  {img_idx + 1}/{len(image_files)} images, {len(predictions)} predictions")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
