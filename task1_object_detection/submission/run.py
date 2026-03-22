"""
NM i AI - NorgesGruppen Object Detection
python run.py --input /data/images --output /output/predictions.json

All inference via ONNX (no pickle, no torch.load).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b1

GROUP_NAMES = [
    "knekkebroed", "coffee", "tea", "cereal",
    "eggs", "spread", "cookies", "chocolate", "other",
]


def load_detector():
    """Load 9-class detector via ONNX runtime."""
    model_path = Path(__file__).parent / "detector.onnx"
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), providers=providers)
    return session


def preprocess_detector(img_path, imgsz=640):
    """Preprocess image for YOLO ONNX detector."""
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size

    # Resize with letterboxing
    scale = min(imgsz / orig_w, imgsz / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    # Pad to imgsz x imgsz
    padded = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
    pad_x = (imgsz - new_w) // 2
    pad_y = (imgsz - new_h) // 2
    padded.paste(img_resized, (pad_x, pad_y))

    arr = np.array(padded).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis]  # [1, 3, 640, 640]
    return arr, img, scale, pad_x, pad_y


def postprocess_detector(output, scale, pad_x, pad_y, conf_threshold=0.15):
    """Parse YOLO ONNX output to boxes."""
    # Output shape: [1, 13, 8400] for 9-class detector
    # 13 = 4 (box) + 9 (class scores)
    pred = output[0][0]  # [13, 8400]

    boxes = pred[:4].T  # [8400, 4] - cx, cy, w, h
    scores = pred[4:].T  # [8400, 9]

    detections = []
    for i in range(len(boxes)):
        class_scores = scores[i]
        max_score = class_scores.max()
        if max_score < conf_threshold:
            continue

        cls_id = int(class_scores.argmax())
        cx, cy, w, h = boxes[i]

        # Remove padding and rescale to original image
        x1 = (cx - w / 2 - pad_x) / scale
        y1 = (cy - h / 2 - pad_y) / scale
        bw = w / scale
        bh = h / scale

        detections.append({
            "x1": float(x1), "y1": float(y1),
            "w": float(bw), "h": float(bh),
            "conf": float(max_score),
            "cls": cls_id,
        })

    return detections


def nms(detections, iou_threshold=0.5):
    """Simple NMS."""
    if not detections:
        return []

    dets = sorted(detections, key=lambda d: d["conf"], reverse=True)
    keep = []

    while dets:
        best = dets.pop(0)
        keep.append(best)
        remaining = []
        for d in dets:
            iou = compute_iou(best, d)
            if iou < iou_threshold:
                remaining.append(d)
        dets = remaining

    return keep


def compute_iou(a, b):
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x1"] + a["w"], b["x1"] + b["w"])
    y2 = min(a["y1"] + a["h"], b["y1"] + b["h"])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = a["w"] * a["h"]
    area_b = b["w"] * b["h"]
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def load_classifiers(device):
    """Load classifiers from .npy weights + reconstruct models."""
    base = Path(__file__).parent

    weights = np.load(str(base / "classifiers.npy"))
    with open(base / "classifier_meta.json") as f:
        meta = json.load(f)

    classifiers = {}
    # Reconstruct each group's model
    groups_seen = set()
    param_data = {}

    for i in range(len(meta["groups"])):
        gn = meta["groups"][i]
        pname = meta["param_names"][i]
        shape = meta["shapes"][i]
        offset = meta["offsets"][i]
        size = 1
        for s in shape:
            size *= s
        arr = weights[offset:offset + size].reshape(shape)

        param_data.setdefault(gn, {})[pname] = torch.from_numpy(arr)
        groups_seen.add(gn)

    for gn in groups_seen:
        classes = meta["classes"][gn]
        n_classes = len(classes)

        model = efficientnet_b1(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
        model.load_state_dict(param_data[gn])
        model = model.to(device)
        model.eval()

        classifiers[gn] = {"model": model, "classes": classes}

    return classifiers


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

    print("Loading detector (ONNX)...")
    detector = load_detector()

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

        # Detect
        inp, img, scale, px, py = preprocess_detector(img_path)
        output = detector.run(None, {detector.get_inputs()[0].name: inp})
        detections = postprocess_detector(output, scale, px, py, conf_threshold=0.15)
        detections = nms(detections, iou_threshold=0.5)

        if not detections:
            continue

        # Crop and classify
        crops = []
        group_names = []
        det_data = []

        for d in detections:
            x1, y1, w, h = d["x1"], d["y1"], d["w"], d["h"]
            if w < 10 or h < 10:
                continue
            gn = GROUP_NAMES[d["cls"]] if d["cls"] < len(GROUP_NAMES) else "other"
            pad_x, pad_y = w * 0.05, h * 0.05
            crop = img.crop((
                max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)),
                min(img.width, int(x1 + w + pad_x)), min(img.height, int(y1 + h + pad_y)),
            ))
            crops.append(crop)
            group_names.append(gn)
            det_data.append((x1, y1, w, h, d["conf"]))

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

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
