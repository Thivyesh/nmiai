"""
Use YOLO backbone as feature extractor for classification.
No CLIP, no ONNX — everything runs in ultralytics (sandbox-compatible).

1. Load trained YOLO detector
2. Extract backbone features from product crops
3. Train LightGBM classifier on features
4. Compare with CLIP approach

Usage:
    uv run python task1_object_detection/experiments/test_yolo_features.py
"""

import json
from pathlib import Path

import numpy as np
import torch
import lightgbm as lgb
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib

_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, "weights_only": False})

TASK_ROOT = Path(__file__).resolve().parent.parent
CLASSIFIER_DIR = TASK_ROOT / "output" / "classifier_dataset"
ANNOTATIONS_FILE = TASK_ROOT / "data" / "coco_dataset" / "train" / "annotations.json"
MODELS_DIR = TASK_ROOT / "output" / "models"
OUTPUT_DIR = TASK_ROOT / "output" / "yolo_classifier"


def get_yolo_feature_extractor(model_path):
    """Load YOLO model and create a feature extractor from its backbone."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    pytorch_model = model.model

    # Hook to capture backbone features
    features = {}

    def hook_fn(name):
        def hook(module, input, output):
            features[name] = output
        return hook

    # Register hooks on the backbone's last layers
    # YOLOv8 architecture: model.model = [backbone layers (0-9), head layers (10+)]
    # Layer 9 is SPPF (Spatial Pyramid Pooling Fast) — good feature summary
    backbone = pytorch_model.model
    backbone[9].register_forward_hook(hook_fn("sppf"))

    return model, pytorch_model, features


def extract_features(pytorch_model, features_dict, img_tensor):
    """Run forward pass and extract backbone features."""
    with torch.no_grad():
        _ = pytorch_model(img_tensor)

    # Get SPPF features and global average pool
    feat = features_dict["sppf"]
    # Global average pooling: [B, C, H, W] -> [B, C]
    pooled = feat.mean(dim=[2, 3])
    return pooled.cpu().numpy()


def preprocess_crop(img_path, imgsz=640):
    """Preprocess a crop for YOLO feature extraction."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((imgsz, imgsz))
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return tensor


def build_features(split, pytorch_model, features_dict, max_per_class=50, imgsz=640):
    """Build feature matrix from crops."""
    split_dir = CLASSIFIER_DIR / split
    if not split_dir.exists():
        return None, None

    X_list = []
    y_list = []

    device = next(pytorch_model.parameters()).device

    cat_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()], key=lambda d: int(d.name))
    total_crops = sum(min(len(list(d.glob("*.jpg"))), max_per_class) for d in cat_dirs)
    processed = 0

    for cat_dir in cat_dirs:
        cat_id = int(cat_dir.name)
        crops = sorted(cat_dir.glob("*.jpg"))[:max_per_class]

        for crop_path in crops:
            try:
                tensor = preprocess_crop(crop_path, imgsz).to(device)
                feats = extract_features(pytorch_model, features_dict, tensor)
                X_list.append(feats.squeeze())
                y_list.append(cat_id)
                processed += 1

                if processed % 200 == 0:
                    print(f"    {split}: {processed}/{total_crops}")
            except Exception as e:
                continue

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    print(f"  {split}: {X.shape[0]} samples, {X.shape[1]} features, {len(set(y))} classes")
    return X, y


def test():
    # Find best detector model
    candidates = list(MODELS_DIR.rglob("best.pt"))
    if not candidates:
        print("ERROR: No trained model found")
        return

    # Prefer 1-class detector
    detector_path = None
    for c in candidates:
        if "detector" in str(c) or "1class" in str(c):
            detector_path = str(c)
            break
    if not detector_path:
        detector_path = str(max(candidates, key=lambda p: p.stat().st_mtime))

    print(f"Using model: {detector_path}")

    # Load model and set up feature extraction
    model, pytorch_model, features_dict = get_yolo_feature_extractor(detector_path)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pytorch_model = pytorch_model.to(device)
    pytorch_model.eval()

    # Extract features
    print("\nExtracting training features...")
    X_train, y_train = build_features("train", pytorch_model, features_dict, max_per_class=50)

    print("\nExtracting validation features...")
    X_val, y_val = build_features("val", pytorch_model, features_dict, max_per_class=30)

    if X_train is None or X_val is None:
        return

    # Save features for reuse
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT_DIR / "train_features.npz", X=X_train, y=y_train)
    np.savez(OUTPUT_DIR / "val_features.npz", X=X_val, y=y_val)

    # Train classifiers
    print(f"\n{'='*60}")
    print("TRAINING CLASSIFIERS ON YOLO BACKBONE FEATURES")
    print(f"{'='*60}")

    classifiers = {
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=500, num_leaves=63, learning_rate=0.05,
            max_depth=-1, min_child_samples=3, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
            n_jobs=-1, verbose=-1,
        ),
        "LogReg": LogisticRegression(
            max_iter=1000, C=1.0, multi_class="multinomial", n_jobs=-1
        ),
    }

    best_acc = 0
    best_name = ""

    for name, clf in classifiers.items():
        print(f"\n  Training {name}...")
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)

        # Top-5
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_val)
            classes = clf.classes_
            top5_acc = 0
            top3_acc = 0
            for i in range(len(y_val)):
                top5 = classes[np.argsort(probs[i])[-5:]]
                top3 = classes[np.argsort(probs[i])[-3:]]
                if y_val[i] in top5:
                    top5_acc += 1
                if y_val[i] in top3:
                    top3_acc += 1
            top5_acc /= len(y_val)
            top3_acc /= len(y_val)
        else:
            top5_acc = top3_acc = 0

        print(f"    Top-1: {acc:.3f}")
        print(f"    Top-3: {top3_acc:.3f}")
        print(f"    Top-5: {top5_acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            best_name = name

        joblib.dump(clf, OUTPUT_DIR / f"{name}.joblib")

    print(f"\n{'='*60}")
    print(f"BEST: {best_name} with {best_acc:.3f} top-1")
    print(f"{'='*60}")
    print(f"\nFor comparison:")
    print(f"  CLIP zero-shot: ~69.5% top-1 (on confusable subset)")
    print(f"  CLIP+OCR:       ~75.2% top-1 (on confusable subset)")
    print(f"  YOLO cls train: <1% (overfit)")
    print(f"\n  YOLO backbone + {best_name}: {best_acc:.1%} top-1 (on full val set)")
    print(f"\n  Note: CLIP was tested on 21 confusable products only.")
    print(f"  This test covers all {len(set(y_val))} val classes.")


if __name__ == "__main__":
    test()
