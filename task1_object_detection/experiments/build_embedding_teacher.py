"""
Build a better teacher using embeddinggemma to encode OCR text.

Pipeline:
1. Pre-compute product name embeddings via embeddinggemma
2. For each training crop: OCR → embed OCR text → cosine sim to all product names
3. Combine [YOLO backbone (576d) + embedding similarities (356d)] → train MLP
4. Generate soft labels from this MLP teacher

Usage:
    uv run python task1_object_detection/experiments/build_embedding_teacher.py
"""

import json
import re
import numpy as np
import requests
from pathlib import Path
from collections import defaultdict
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

TASK_ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_FILE = TASK_ROOT / "data" / "coco_dataset" / "train" / "annotations.json"
CLASSIFIER_DIR = TASK_ROOT / "output" / "classifier_dataset"
OUTPUT_DIR = TASK_ROOT / "output" / "embedding_teacher"
EMBED_URL = "http://localhost:8080/embed"

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
GROUP_NAMES = ["knekkebroed", "coffee", "tea", "cereal", "eggs", "spread", "cookies", "chocolate", "other"]


def get_group(cat_id, cat_map):
    name = cat_map.get(cat_id, "").lower()
    for g, kws in SUPER_CATEGORIES.items():
        if any(kw in name for kw in kws):
            return g
    return "other"


def embed_texts(texts, batch_size=8, dim=768):
    """Embed texts via local embeddinggemma API. One at a time for reliability."""
    all_embeddings = []
    for text in texts:
        clean = text.strip() if text else "unknown product"
        if not clean:
            clean = "unknown product"
        try:
            resp = requests.post(EMBED_URL, json={"inputs": [clean]}, timeout=15)
            data = resp.json()
            if isinstance(data, list) and len(data) == 1 and isinstance(data[0], list) and len(data[0]) == dim:
                all_embeddings.append(data[0])
            else:
                all_embeddings.append([0.0] * dim)
        except Exception:
            all_embeddings.append([0.0] * dim)
    return np.array(all_embeddings, dtype=np.float32)


def build():
    print("Loading annotations...")
    with open(ANNOTATIONS_FILE) as f:
        coco = json.load(f)
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    all_cat_ids = sorted(cat_map.keys())

    # Step 1: Embed all product names
    print("Embedding product names...")
    product_names = [cat_map[cid] for cid in all_cat_ids]
    name_embeddings = embed_texts(product_names)
    print(f"  Product embeddings: {name_embeddings.shape}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "product_name_embeddings.npy", name_embeddings)
    with open(OUTPUT_DIR / "cat_ids.json", "w") as f:
        json.dump(all_cat_ids, f)

    # Step 2: Load YOLO backbone features (already computed)
    bb_train = np.load(TASK_ROOT / "output" / "yolo_classifier" / "train_features.npz")
    bb_val = np.load(TASK_ROOT / "output" / "yolo_classifier" / "val_features.npz")
    X_bb_train, y_train = bb_train["X"], bb_train["y"]
    X_bb_val, y_val = bb_val["X"], bb_val["y"]
    print(f"  Backbone features: train={X_bb_train.shape}, val={X_bb_val.shape}")

    # Step 3: OCR each crop → embed → cosine sim to product names
    print("Computing OCR embeddings for training crops...")
    import easyocr
    reader = easyocr.Reader(["en", "no"], gpu=False, verbose=False)

    def ocr_and_embed(crop_path):
        """OCR a crop, embed the text, return cosine similarities to all products."""
        try:
            results = reader.readtext(str(crop_path))
            ocr_text = " ".join(r[1] for r in results)
        except Exception:
            ocr_text = ""

        if not ocr_text.strip():
            return np.zeros(len(all_cat_ids), dtype=np.float32)

        # Embed OCR text
        try:
            ocr_emb = embed_texts([ocr_text])[0]  # [768]
        except Exception:
            return np.zeros(len(all_cat_ids), dtype=np.float32)

        # Cosine similarity to all product name embeddings
        n = len(all_cat_ids)
        norm_ocr = np.linalg.norm(ocr_emb)
        if norm_ocr == 0:
            return np.zeros(n, dtype=np.float32)

        sims = np.zeros(n, dtype=np.float32)
        for k in range(n):
            norm_p = np.linalg.norm(name_embeddings[k])
            if norm_p > 0:
                sims[k] = np.dot(name_embeddings[k], ocr_emb) / (norm_p * norm_ocr)
        return sims

    # Process crops to get OCR embedding features
    # Match crop order with backbone features (same iteration order)
    for split, X_bb, y in [("train", X_bb_train, y_train), ("val", X_bb_val, y_val)]:
        split_dir = CLASSIFIER_DIR / split
        ocr_features = []
        crop_idx = 0

        cat_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()], key=lambda d: int(d.name))
        max_per_class = 50 if split == "train" else 30

        for cat_dir in cat_dirs:
            cat_id = int(cat_dir.name)
            crops = sorted(cat_dir.glob("*.jpg"))[:max_per_class]

            for crop_path in crops:
                if crop_idx >= len(y):
                    break
                sims = ocr_and_embed(crop_path)
                ocr_features.append(sims)
                crop_idx += 1

                if crop_idx % 200 == 0:
                    print(f"    {split}: {crop_idx}/{len(y)} crops")

        # Ensure all features have same length
        n_cats = len(all_cat_ids)
        fixed_features = []
        for feat in ocr_features[:len(y)]:
            if isinstance(feat, np.ndarray) and feat.shape == (n_cats,):
                fixed_features.append(feat)
            else:
                fixed_features.append(np.zeros(n_cats, dtype=np.float32))
        X_ocr = np.array(fixed_features, dtype=np.float32)
        print(f"  {split} OCR features: {X_ocr.shape}")

        # Combine: backbone (576d) + OCR embedding sims (356d)
        X_combined = np.concatenate([X_bb[:len(X_ocr)], X_ocr], axis=1)
        y_trimmed = y[:len(X_ocr)]

        np.savez(OUTPUT_DIR / f"{split}_features.npz", X=X_combined, y=y_trimmed)
        print(f"  {split} combined: {X_combined.shape}")

    # Step 4: Train MLP teacher per group
    print("\nTraining MLP teachers per group...")
    X_train = np.load(OUTPUT_DIR / "train_features.npz")
    X_val = np.load(OUTPUT_DIR / "val_features.npz")
    X_tr, y_tr = X_train["X"], X_train["y"]
    X_va, y_va = X_val["X"], X_val["y"]

    for group in GROUP_NAMES:
        tr_mask = np.array([get_group(y, cat_map) == group for y in y_tr])
        va_mask = np.array([get_group(y, cat_map) == group for y in y_va])

        Xtr, ytr = X_tr[tr_mask], y_tr[tr_mask]
        Xva, yva = X_va[va_mask], y_va[va_mask]

        if len(set(ytr)) < 2 or len(Xva) == 0:
            continue

        # Compare LogReg vs MLP
        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(Xtr, ytr)
        lr_acc = accuracy_score(yva, lr.predict(Xva))

        mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500,
                            early_stopping=True, validation_fraction=0.1)
        mlp.fit(Xtr, ytr)
        mlp_acc = accuracy_score(yva, mlp.predict(Xva))

        best = mlp if mlp_acc > lr_acc else lr
        best_acc = max(mlp_acc, lr_acc)
        best_name = "MLP" if mlp_acc > lr_acc else "LogReg"

        # Save best teacher
        group_dir = OUTPUT_DIR / group
        group_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(best, group_dir / "teacher.joblib")

        # Generate soft labels
        if hasattr(best, "predict_proba"):
            train_probs = best.predict_proba(Xtr)
            val_probs = best.predict_proba(Xva)
            classes = best.classes_

            # Save soft labels
            soft = {}
            # We need crop paths to match - simplified: save as arrays
            np.savez(group_dir / "soft_labels.npz",
                     train_probs=train_probs, train_y=ytr,
                     val_probs=val_probs, val_y=yva,
                     classes=classes)

        print(f"  {group:15s}: {best_name} acc={best_acc:.3f} "
              f"(LR={lr_acc:.3f} MLP={mlp_acc:.3f}) [{len(set(ytr))} cls]")

    # Compare with old LogReg teacher (backbone only)
    print(f"\nComparison with old backbone-only LogReg:")
    old_bb_tr = np.load(TASK_ROOT / "output" / "yolo_classifier" / "train_features.npz")
    old_bb_va = np.load(TASK_ROOT / "output" / "yolo_classifier" / "val_features.npz")

    for group in GROUP_NAMES:
        tr_mask = np.array([get_group(y, cat_map) == group for y in old_bb_tr["y"]])
        va_mask = np.array([get_group(y, cat_map) == group for y in old_bb_va["y"]])

        Xtr_old = old_bb_tr["X"][tr_mask]
        ytr_old = old_bb_tr["y"][tr_mask]
        Xva_old = old_bb_va["X"][va_mask]
        yva_old = old_bb_va["y"][va_mask]

        if len(set(ytr_old)) < 2 or len(Xva_old) == 0:
            continue

        lr_old = LogisticRegression(max_iter=1000, C=1.0)
        lr_old.fit(Xtr_old, ytr_old)
        old_acc = accuracy_score(yva_old, lr_old.predict(Xva_old))

        # New teacher accuracy
        new_teacher = joblib.load(OUTPUT_DIR / group / "teacher.joblib")
        va_mask_new = np.array([get_group(y, cat_map) == group for y in X_val["y"]])
        new_acc = accuracy_score(X_val["y"][va_mask_new],
                                  new_teacher.predict(X_val["X"][va_mask_new]))

        diff = new_acc - old_acc
        print(f"  {group:15s}: old={old_acc:.3f} new={new_acc:.3f} diff={diff:+.3f}")


if __name__ == "__main__":
    build()
