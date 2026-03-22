"""
Step 1: Generate soft labels for each crop using YOLO backbone + OCR + LogReg.

For each training crop, produces a probability distribution over classes
within its super-category group. These soft labels are used to train
YOLO-cls with knowledge distillation.

Usage:
    uv run python task1_object_detection/experiments/build_soft_labels.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression

_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, "weights_only": False})

TASK_ROOT = Path(__file__).resolve().parent.parent
CLASSIFIER_DIR = TASK_ROOT / "output" / "classifier_dataset"
ANNOTATIONS_FILE = TASK_ROOT / "data" / "coco_dataset" / "train" / "annotations.json"
OUTPUT_DIR = TASK_ROOT / "output" / "soft_labels"
MODELS_DIR = TASK_ROOT / "output" / "models"

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
    name_lower = cat_map.get(cat_id, "").lower()
    for g, kws in SUPER_CATEGORIES.items():
        if any(kw in name_lower for kw in kws):
            return g
    return "other"


def get_yolo_feature_extractor(model_path):
    """Load YOLO and hook into backbone SPPF layer."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    pytorch_model = model.model
    features = {}

    def hook_fn(name):
        def hook(module, input, output):
            features[name] = output
        return hook

    pytorch_model.model[9].register_forward_hook(hook_fn("sppf"))
    return pytorch_model, features


def extract_backbone_features(pytorch_model, features_dict, img_path, device, imgsz=640):
    """Extract pooled backbone features from a crop."""
    img = Image.open(img_path).convert("RGB").resize((imgsz, imgsz))
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        _ = pytorch_model(tensor)

    feat = features_dict["sppf"]
    pooled = feat.mean(dim=[2, 3]).cpu().numpy().squeeze()
    return pooled


def compute_ocr_features(ocr_reader, img_path, group_cat_ids, cat_map):
    """OCR word-match scores for candidates in a group."""
    try:
        results = ocr_reader.readtext(str(img_path))
        ocr_text = " ".join(r[1] for r in results).upper()
    except Exception:
        ocr_text = ""

    scores = []
    for cid in group_cat_ids:
        name_words = cat_map[cid].upper().split()
        if name_words and ocr_text:
            matches = sum(1 for w in name_words if w in ocr_text)
            scores.append(matches / len(name_words))
        else:
            scores.append(0.0)

    extra = [
        len(ocr_text) / 100.0,
        1.0 if re.search(r"\d+G", ocr_text) else 0.0,
        1.0 if re.search(r"\d+STK", ocr_text) else 0.0,
        1.0 if re.search(r"\d+POS", ocr_text) else 0.0,
    ]
    return np.array(scores + extra, dtype=np.float32)


def build():
    print("Loading annotations...")
    with open(ANNOTATIONS_FILE) as f:
        coco = json.load(f)
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}

    # Group categories
    groups = defaultdict(list)
    for cid, name in cat_map.items():
        groups[get_group(cid, cat_map)].append(cid)
    for g in groups:
        groups[g] = sorted(groups[g])

    # Load YOLO detector for feature extraction
    detector_candidates = list(MODELS_DIR.rglob("detector_*/weights/best.pt"))
    if not detector_candidates:
        detector_candidates = list(MODELS_DIR.rglob("*/weights/best.pt"))
    detector_path = str(max(detector_candidates, key=lambda p: p.stat().st_mtime))
    print(f"Using detector: {detector_path}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pytorch_model, features_dict = get_yolo_feature_extractor(detector_path)
    pytorch_model = pytorch_model.to(device).eval()

    # Load OCR
    try:
        import easyocr
        ocr_reader = easyocr.Reader(["en", "no"], gpu=False, verbose=False)
        use_ocr = True
        print("EasyOCR loaded")
    except ImportError:
        ocr_reader = None
        use_ocr = False
        print("No OCR — using backbone features only")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # For each group, train LogReg and generate soft labels
    for group_name, group_cat_ids in sorted(groups.items()):
        print(f"\n{'='*60}")
        print(f"GROUP: {group_name} ({len(group_cat_ids)} classes)")
        print(f"{'='*60}")

        # Collect features for train and val
        for split in ["train", "val"]:
            split_dir = CLASSIFIER_DIR / split
            X_bb = []
            X_ocr = []
            y_labels = []
            crop_paths = []

            for cid in group_cat_ids:
                cat_dir = split_dir / str(cid)
                if not cat_dir.exists():
                    continue
                crops = sorted(cat_dir.glob("*.jpg"))
                for crop_path in crops:
                    try:
                        bb_feat = extract_backbone_features(
                            pytorch_model, features_dict, crop_path, device
                        )
                        X_bb.append(bb_feat)

                        if use_ocr:
                            ocr_feat = compute_ocr_features(
                                ocr_reader, crop_path, group_cat_ids, cat_map
                            )
                            X_ocr.append(ocr_feat)

                        y_labels.append(cid)
                        crop_paths.append(str(crop_path))
                    except Exception as e:
                        continue

                if len(X_bb) % 100 == 0 and len(X_bb) > 0:
                    print(f"  {split}: {len(X_bb)} crops processed")

            if not X_bb:
                continue

            X_bb = np.array(X_bb, dtype=np.float32)
            y = np.array(y_labels)

            if use_ocr and X_ocr:
                X_ocr = np.array(X_ocr, dtype=np.float32)
                X = np.concatenate([X_bb, X_ocr], axis=1)
            else:
                X = X_bb

            print(f"  {split}: {X.shape[0]} samples, {X.shape[1]} features")

            if split == "train":
                X_train, y_train = X, y
                train_paths = crop_paths
            else:
                X_val, y_val = X, y
                val_paths = crop_paths

        # Train LogReg on this group
        if len(set(y_train)) < 2:
            print(f"  Skipping — only 1 class in train")
            continue

        print(f"  Training LogReg...")
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_train, y_train)

        # Save the classifier
        group_dir = OUTPUT_DIR / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, group_dir / "logreg.joblib")

        # Generate soft labels for train set
        train_probs = clf.predict_proba(X_train)
        train_classes = clf.classes_

        # Build soft label mapping: crop_path -> {cat_id: probability}
        soft_labels = {}
        for i, path in enumerate(train_paths):
            probs_dict = {}
            for j, cid in enumerate(train_classes):
                probs_dict[int(cid)] = round(float(train_probs[i, j]), 4)
            soft_labels[path] = {
                "true_label": int(y_train[i]),
                "soft_labels": probs_dict,
            }

        with open(group_dir / "soft_labels_train.json", "w") as f:
            json.dump(soft_labels, f, indent=2)

        # Evaluate on val
        if 'X_val' in dir() and len(X_val) > 0:
            val_preds = clf.predict(X_val)
            acc = sum(val_preds == y_val) / len(y_val)
            probs_val = clf.predict_proba(X_val)
            top5 = sum(
                1 for i in range(len(y_val))
                if y_val[i] in train_classes[np.argsort(probs_val[i])[-5:]]
            ) / len(y_val)
            print(f"  Val accuracy: top-1={acc:.3f}, top-5={top5:.3f}")

            # Save val soft labels too
            val_soft = {}
            for i, path in enumerate(val_paths):
                probs_dict = {}
                for j, cid in enumerate(train_classes):
                    probs_dict[int(cid)] = round(float(probs_val[i, j]), 4)
                val_soft[path] = {
                    "true_label": int(y_val[i]),
                    "soft_labels": probs_dict,
                }
            with open(group_dir / "soft_labels_val.json", "w") as f:
                json.dump(val_soft, f, indent=2)

        # Save group metadata
        meta = {
            "group": group_name,
            "cat_ids": [int(c) for c in group_cat_ids],
            "cat_names": {int(c): cat_map[c] for c in group_cat_ids},
            "n_train": len(X_train),
            "n_features": int(X.shape[1]),
            "logreg_classes": [int(c) for c in train_classes],
        }
        with open(group_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"  Saved to {group_dir}")

    print(f"\nAll soft labels saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    build()
