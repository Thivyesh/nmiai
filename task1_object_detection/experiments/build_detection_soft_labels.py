"""
Generate soft detection labels using ensemble of detectors + classifier verification.

For each training image:
1. Run 1-class detector → high-recall product boxes
2. Run 9-class detector → category-aware boxes
3. Merge with WBF → averaged box locations
4. Run classifier on each merged box → confidence score
5. Combine into soft detection labels:
   - GT boxes get weight 1.0
   - Ensemble-only boxes (not in GT) with high classifier conf → pseudo-labels
   - Box coordinates are averaged between models for smoother targets

Usage:
    uv run python task1_object_detection/experiments/build_detection_soft_labels.py
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b1
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

_orig = torch.load
torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})

TASK_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = TASK_ROOT / "data" / "coco_dataset" / "train" / "images"
ANN_FILE = TASK_ROOT / "data" / "coco_dataset" / "train" / "annotations.json"
OUTPUT_DIR = TASK_ROOT / "output" / "detection_soft_labels"

GROUP_NAMES = ["knekkebroed", "coffee", "tea", "cereal", "eggs", "spread", "cookies", "chocolate", "other"]

SUPER_CATEGORIES = {
    "knekkebroed": ["knekkebrød", "knekke", "flatbrød", "wasa", "sigdal", "leksands", "ryvita", "korni"],
    "coffee": ["kaffe", "coffee", "espresso", "nescafe", "evergood", "friele", "ali ", "dolce gusto", "cappuccino", "kapsel"],
    "tea": [" te ", "tea", "twinings", "lipton", "pukka", "urtete"],
    "cereal": ["frokost", "havre", "müsli", "granola", "corn flakes", "cheerios", "cruesli", "puffet", "fras"],
    "eggs": ["egg"],
    "spread": ["smør", "bremykt", "brelett", "ost ", "cream cheese"],
    "cookies": ["kjeks", "cookie", "grissini"],
    "chocolate": ["sjokolade", "nugatti", "regia", "cocoa"],
}


def get_group(cat_id, cat_map):
    name = cat_map.get(cat_id, "").lower()
    for g, kws in SUPER_CATEGORIES.items():
        if any(kw in name for kw in kws):
            return g
    return "other"


def load_classifiers(device):
    """Load EfficientNet-b1 classifiers for verification."""
    clfs = {}
    model_dir = TASK_ROOT / "output" / "models" / "distilled_v2_effb1"

    for gn in GROUP_NAMES:
        mp = model_dir / gn / "best.pt"
        cp = model_dir / gn / "classes.json"
        if not mp.exists():
            continue
        with open(cp) as f:
            classes = json.load(f)
        m = efficientnet_b1(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, len(classes))
        m.load_state_dict(torch.load(str(mp), map_location=device))
        m = m.to(device).eval()
        clfs[gn] = {"model": m, "classes": classes}

    return clfs


def iou(a, b):
    """IoU between two [x,y,w,h] boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2])
    y2 = min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0


def build():
    print("Loading annotations...")
    with open(ANN_FILE) as f:
        coco = json.load(f)
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    img_lookup = {img["id"]: img for img in coco["images"]}
    img_anns = {}
    for ann in coco["annotations"]:
        img_anns.setdefault(ann["image_id"], []).append(ann)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Load both detectors
    print("Loading 1-class detector...")
    det_1cls = YOLO(str(TASK_ROOT / "output" / "models" / "detector_yolov8m_e50_img640" / "weights" / "best.pt"))

    print("Loading 9-class detector...")
    det_9cls = YOLO(str(TASK_ROOT / "output" / "models" / "superclass_yolov8m_e50" / "weights" / "best.pt"))

    # Load YOLOv8x if available
    det_x_path = TASK_ROOT / "output" / "models" / "yolov8x_gpu" / "best.pt"
    if det_x_path.exists():
        print("Loading YOLOv8x detector...")
        det_x = YOLO(str(det_x_path))
        detectors = [det_1cls, det_9cls, det_x]
        det_weights = [1, 1, 2]  # weight x higher
    else:
        detectors = [det_1cls, det_9cls]
        det_weights = [1, 1]

    print("Loading classifiers...")
    clfs = load_classifiers(device)
    cls_transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build category remapping for YOLO format
    all_cat_ids = sorted(cat_map.keys())
    cat_remap = {old_id: new_id for new_id, old_id in enumerate(all_cat_ids)}
    cat_to_group = {cid: get_group(cid, cat_map) for cid in cat_map}

    # Process each training image
    soft_labels = {}
    total_pseudo = 0
    total_refined = 0

    for img_idx, (img_id, img_info) in enumerate(img_lookup.items()):
        img_path = IMAGES_DIR / img_info["file_name"]
        if not img_path.exists():
            continue

        ow, oh = img_info["width"], img_info["height"]
        gt_anns = img_anns.get(img_id, [])
        img = Image.open(img_path).convert("RGB")

        # Run all detectors
        all_boxes = []
        all_scores = []
        all_labels = []

        for det in detectors:
            with torch.no_grad():
                results = det(str(img_path), device=device, verbose=False, conf=0.1)
            if not results or results[0].boxes is None:
                all_boxes.append([])
                all_scores.append([])
                all_labels.append([])
                continue

            boxes_norm = []
            scores = []
            labels = []
            for i in range(len(results[0].boxes)):
                x1, y1, x2, y2 = results[0].boxes.xyxy[i].tolist()
                conf = float(results[0].boxes.conf[i].item())
                boxes_norm.append([x1 / ow, y1 / oh, x2 / ow, y2 / oh])
                scores.append(conf)
                labels.append(0)  # all same label for WBF

            all_boxes.append(boxes_norm)
            all_scores.append(scores)
            all_labels.append(labels)

        # WBF merge
        if any(len(b) > 0 for b in all_boxes):
            valid = [(b, s, l) for b, s, l in zip(all_boxes, all_scores, all_labels) if len(b) > 0]
            if valid:
                bl = [v[0] for v in valid]
                sl = [v[1] for v in valid]
                ll = [v[2] for v in valid]
                w = det_weights[:len(valid)]

                fused_boxes, fused_scores, _ = weighted_boxes_fusion(
                    bl, sl, ll, weights=w, iou_thr=0.5, skip_box_thr=0.1
                )
            else:
                fused_boxes, fused_scores = np.array([]), np.array([])
        else:
            fused_boxes, fused_scores = np.array([]), np.array([])

        # Match fused boxes to GT
        refined_anns = []
        used_gt = set()

        for fi in range(len(fused_boxes)):
            fb = fused_boxes[fi]
            fs = fused_scores[fi]
            # Convert normalized xyxy to xywh pixel coords
            fx1 = fb[0] * ow
            fy1 = fb[1] * oh
            fw = (fb[2] - fb[0]) * ow
            fh = (fb[3] - fb[1]) * oh
            fused_xywh = [fx1, fy1, fw, fh]

            # Find best matching GT
            best_iou = 0
            best_gt_idx = None
            for gi, gt in enumerate(gt_anns):
                if gi in used_gt:
                    continue
                io = iou(fused_xywh, gt["bbox"])
                if io > best_iou:
                    best_iou = io
                    best_gt_idx = gi

            if best_iou >= 0.5 and best_gt_idx is not None:
                # Matched GT — refine box coordinates (average GT + detector)
                gt = gt_anns[best_gt_idx]
                used_gt.add(best_gt_idx)

                # Weighted average: 70% GT + 30% detector (GT is more reliable)
                ref_bbox = [
                    0.7 * gt["bbox"][0] + 0.3 * fused_xywh[0],
                    0.7 * gt["bbox"][1] + 0.3 * fused_xywh[1],
                    0.7 * gt["bbox"][2] + 0.3 * fused_xywh[2],
                    0.7 * gt["bbox"][3] + 0.3 * fused_xywh[3],
                ]

                refined_anns.append({
                    "bbox": ref_bbox,
                    "category_id": gt["category_id"],
                    "confidence": 1.0,
                    "source": "gt_refined",
                })
                total_refined += 1
            else:
                # No GT match — potential pseudo-label
                # Verify with classifier
                if fw < 10 or fh < 10:
                    continue

                pad_x, pad_y = fw * 0.05, fh * 0.05
                crop = img.crop((
                    max(0, int(fx1 - pad_x)), max(0, int(fy1 - pad_y)),
                    min(ow, int(fx1 + fw + pad_x)), min(oh, int(fy1 + fh + pad_y)),
                ))

                # Run all classifiers, pick best
                best_cls_conf = 0
                best_cat_id = 0
                best_group = "other"

                inp_t = cls_transform(crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    for gn, clf in clfs.items():
                        probs = torch.softmax(clf["model"](inp_t), 1)
                        mc = probs.max().item()
                        if mc > best_cls_conf:
                            best_cls_conf = mc
                            best_cat_id = clf["classes"][probs.argmax(1).item()]
                            best_group = gn

                # Only add as pseudo-label if both detector and classifier are confident
                if fs >= 0.3 and best_cls_conf >= 0.5:
                    refined_anns.append({
                        "bbox": [fx1, fy1, fw, fh],
                        "category_id": best_cat_id,
                        "confidence": float(fs * best_cls_conf),
                        "source": "pseudo",
                    })
                    total_pseudo += 1

        # Also add any GT boxes that weren't matched (detector missed them)
        for gi, gt in enumerate(gt_anns):
            if gi not in used_gt:
                refined_anns.append({
                    "bbox": gt["bbox"],
                    "category_id": gt["category_id"],
                    "confidence": 1.0,
                    "source": "gt_only",
                })

        soft_labels[img_id] = refined_anns

        if (img_idx + 1) % 50 == 0:
            print(f"  {img_idx + 1}/{len(img_lookup)} images, "
                  f"refined={total_refined}, pseudo={total_pseudo}")

    # Save
    # Convert to YOLO format with soft labels
    yolo_dir = OUTPUT_DIR / "yolo_soft"
    for split_name in ["train"]:
        (yolo_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    for img_id, anns in soft_labels.items():
        img_info = img_lookup[img_id]
        iw, ih = img_info["width"], img_info["height"]

        # Symlink image
        src = IMAGES_DIR / img_info["file_name"]
        dst = yolo_dir / "images" / "train" / img_info["file_name"]
        if not dst.exists():
            dst.symlink_to(src)

        # Write YOLO label with confidence weights
        label_name = Path(img_info["file_name"]).stem + ".txt"
        lines = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            cx = (x + w / 2) / iw
            cy = (y + h / 2) / ih
            nw = w / iw
            nh = h / ih
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            nw = max(0, min(1, nw))
            nh = max(0, min(1, nh))

            cat_id = cat_remap.get(ann["category_id"], 0)
            group = cat_to_group.get(ann["category_id"], "other")
            group_id = GROUP_NAMES.index(group) if group in GROUP_NAMES else 8

            lines.append(f"{group_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        with open(yolo_dir / "labels" / "train" / label_name, "w") as f:
            f.write("\n".join(lines))

    # Write data.yaml
    with open(yolo_dir / "data.yaml", "w") as f:
        f.write(f"path: {yolo_dir.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/train\n")  # use train as val for now
        f.write(f"nc: {len(GROUP_NAMES)}\n")
        f.write("names:\n")
        for i, name in enumerate(GROUP_NAMES):
            f.write(f'  {i}: "{name}"\n')

    # Save metadata
    meta = {
        "total_images": len(soft_labels),
        "total_refined": total_refined,
        "total_pseudo": total_pseudo,
        "total_gt_only": sum(
            sum(1 for a in anns if a["source"] == "gt_only")
            for anns in soft_labels.values()
        ),
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DETECTION SOFT LABELS BUILT")
    print(f"{'='*60}")
    print(f"  Images: {len(soft_labels)}")
    print(f"  GT refined (box smoothing): {total_refined}")
    print(f"  Pseudo-labels added: {total_pseudo}")
    print(f"  GT-only (detector missed): {meta['total_gt_only']}")
    print(f"  Total annotations: {sum(len(a) for a in soft_labels.values())}")
    print(f"  Saved to: {yolo_dir}")


if __name__ == "__main__":
    build()
