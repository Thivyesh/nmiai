"""
Train a lightweight classifier on CLIP embeddings + OCR features.

Pipeline:
1. Compute CLIP image embeddings for all training crops
2. Compute cosine similarity to all 356 product name text embeddings
3. Extract OCR word-match features
4. Train sklearn classifier on [CLIP similarities + OCR features] → category_id

Usage:
    uv run python task1_object_detection/experiments/train_clip_classifier.py
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import joblib

TASK_ROOT = Path(__file__).resolve().parent.parent
CLASSIFIER_DIR = TASK_ROOT / "output" / "classifier_dataset"
ANNOTATIONS_FILE = TASK_ROOT / "data" / "coco_dataset" / "train" / "annotations.json"
OUTPUT_DIR = TASK_ROOT / "output" / "clip_classifier"


def build_text_embeddings(model, processor, cat_map, device):
    """Pre-compute CLIP text embeddings for all product names."""
    print("Computing text embeddings for all products...")
    cat_ids = sorted(cat_map.keys())
    descriptions = [f"a photo of {cat_map[cid]}" for cid in cat_ids]

    all_text_features = []
    batch_size = 32
    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i:i + batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_out = model.get_text_features(**inputs)
            text_features = text_out if isinstance(text_out, torch.Tensor) else text_out.pooler_output
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            all_text_features.append(text_features.cpu())

    text_embeddings = torch.cat(all_text_features, dim=0).numpy()  # [num_cats, 512]
    print(f"  Text embeddings shape: {text_embeddings.shape}")
    return cat_ids, text_embeddings


def compute_clip_features(model, processor, img, text_embeddings, device):
    """Compute CLIP similarity features for a crop."""
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        img_out = model.get_image_features(**inputs)
        img_features = img_out if isinstance(img_out, torch.Tensor) else img_out.pooler_output
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)

    # Cosine similarity to all product text embeddings
    similarities = (img_features.cpu().numpy() @ text_embeddings.T).squeeze()  # [num_cats]
    return similarities


def compute_ocr_features(ocr_reader, img_path, cat_map, cat_ids):
    """Compute OCR-based word match features."""
    try:
        ocr_results = ocr_reader.readtext(str(img_path))
        ocr_text = " ".join(r[1] for r in ocr_results).upper()
    except Exception:
        ocr_text = ""

    # For each product, compute word match ratio
    ocr_features = np.zeros(len(cat_ids), dtype=np.float32)
    for i, cid in enumerate(cat_ids):
        name_words = cat_map[cid].upper().split()
        if name_words and ocr_text:
            matches = sum(1 for w in name_words if w in ocr_text)
            ocr_features[i] = matches / len(name_words)

    # Also extract some generic OCR features
    # - number of text detections
    # - total text length
    # - presence of common size patterns (G, ML, STK, POS)
    extra = np.zeros(5, dtype=np.float32)
    extra[0] = len(ocr_text) / 100.0  # normalized text length
    extra[1] = 1.0 if re.search(r'\d+G', ocr_text) else 0.0
    extra[2] = 1.0 if re.search(r'\d+ML', ocr_text) else 0.0
    extra[3] = 1.0 if re.search(r'\d+STK', ocr_text) else 0.0
    extra[4] = 1.0 if re.search(r'\d+POS', ocr_text) else 0.0

    return ocr_features, extra, ocr_text


def build_features(split, model, processor, ocr_reader, text_embeddings, cat_ids, cat_map, device, max_per_class=50):
    """Build feature matrix for a split."""
    split_dir = CLASSIFIER_DIR / split
    if not split_dir.exists():
        print(f"  {split} dir not found!")
        return None, None

    X_clips = []
    X_ocrs = []
    X_extras = []
    y = []

    cat_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()], key=lambda d: int(d.name))
    total = sum(min(len(list(d.glob("*.jpg"))), max_per_class) for d in cat_dirs)
    processed = 0

    for cat_dir in cat_dirs:
        cat_id = int(cat_dir.name)
        if cat_id not in cat_map:
            continue

        crops = sorted(cat_dir.glob("*.jpg"))[:max_per_class]

        for crop_path in crops:
            try:
                img = Image.open(crop_path).convert("RGB")
            except Exception:
                continue

            # CLIP features
            clip_sims = compute_clip_features(model, processor, img, text_embeddings, device)
            X_clips.append(clip_sims)

            # OCR features
            if ocr_reader:
                ocr_feats, extra_feats, _ = compute_ocr_features(ocr_reader, crop_path, cat_map, cat_ids)
                X_ocrs.append(ocr_feats)
                X_extras.append(extra_feats)

            y.append(cat_id)
            processed += 1

            if processed % 100 == 0:
                print(f"  {split}: {processed}/{total} crops processed")

    X_clip = np.array(X_clips, dtype=np.float32)

    if X_ocrs:
        X_ocr = np.array(X_ocrs, dtype=np.float32)
        X_extra = np.array(X_extras, dtype=np.float32)
        X = np.concatenate([X_clip, X_ocr, X_extra], axis=1)
    else:
        X = X_clip

    y = np.array(y)
    print(f"  {split}: {X.shape[0]} samples, {X.shape[1]} features, {len(set(y))} classes")
    return X, y


def train():
    from transformers import CLIPModel, CLIPProcessor

    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)

    try:
        import easyocr
        ocr_reader = easyocr.Reader(["en", "no"], gpu=False, verbose=False)
        print("EasyOCR loaded")
    except ImportError:
        ocr_reader = None
        print("EasyOCR not available — using CLIP features only")

    # Load annotations
    with open(ANNOTATIONS_FILE) as f:
        coco = json.load(f)
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}

    # Build text embeddings
    cat_ids, text_embeddings = build_text_embeddings(model, processor, cat_map, device)

    # Save text embeddings for inference
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "text_embeddings.npy", text_embeddings)
    with open(OUTPUT_DIR / "cat_ids.json", "w") as f:
        json.dump(cat_ids, f)
    print(f"Saved text embeddings to {OUTPUT_DIR}")

    # Build features
    print("\nBuilding training features...")
    X_train, y_train = build_features(
        "train", model, processor, ocr_reader, text_embeddings, cat_ids, cat_map, device,
        max_per_class=30  # limit to keep it fast
    )

    print("\nBuilding validation features...")
    X_val, y_val = build_features(
        "val", model, processor, ocr_reader, text_embeddings, cat_ids, cat_map, device,
        max_per_class=20
    )

    if X_train is None or X_val is None:
        print("ERROR: Could not build features")
        return

    # Train multiple classifiers and compare
    print(f"\n{'='*60}")
    print("TRAINING CLASSIFIERS")
    print(f"{'='*60}")

    classifiers = {
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=500, num_leaves=63, learning_rate=0.05,
            max_depth=-1, min_child_samples=3, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
            n_jobs=-1, verbose=-1,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, C=1.0, multi_class="multinomial", n_jobs=-1
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(512, 256), max_iter=500, early_stopping=True,
            validation_fraction=0.1, learning_rate="adaptive"
        ),
    }

    best_acc = 0
    best_name = ""

    for name, clf in classifiers.items():
        print(f"\n  Training {name}...")
        clf.fit(X_train, y_train)

        # Evaluate
        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)

        # Top-5 accuracy
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_val)
            # Need label encoder for top_k
            classes = clf.classes_
            top5_acc = 0
            for i in range(len(y_val)):
                top5_preds = classes[np.argsort(probs[i])[-5:]]
                if y_val[i] in top5_preds:
                    top5_acc += 1
            top5_acc /= len(y_val)
        else:
            top5_acc = 0

        print(f"    Top-1 accuracy: {acc:.3f}")
        print(f"    Top-5 accuracy: {top5_acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            best_name = name

        # Save model
        model_path = OUTPUT_DIR / f"{name}.joblib"
        joblib.dump(clf, model_path)

    print(f"\n{'='*60}")
    print(f"BEST: {best_name} with {best_acc:.3f} top-1 accuracy")
    print(f"{'='*60}")

    # Compare with zero-shot CLIP
    print(f"\nBaseline comparison (zero-shot CLIP on val):")
    clip_correct = 0
    clip_top5 = 0
    # For zero-shot, just use the CLIP similarity columns
    clip_features = X_val[:, :len(cat_ids)]  # first N columns are CLIP similarities
    for i in range(len(y_val)):
        pred_idx = np.argmax(clip_features[i])
        pred_cat = cat_ids[pred_idx]
        if pred_cat == y_val[i]:
            clip_correct += 1
        top5_cats = [cat_ids[j] for j in np.argsort(clip_features[i])[-5:]]
        if y_val[i] in top5_cats:
            clip_top5 += 1
    print(f"  Zero-shot CLIP top-1: {clip_correct/len(y_val):.3f}")
    print(f"  Zero-shot CLIP top-5: {clip_top5/len(y_val):.3f}")


if __name__ == "__main__":
    train()
