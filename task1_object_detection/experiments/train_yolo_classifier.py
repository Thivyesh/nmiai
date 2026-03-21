"""
Train a YOLOv8 classifier on cropped product images.
Uses the same ultralytics framework as the detector — guaranteed sandbox compat.

The classifier dataset must be organized as:
  classifier_dataset/train/{category_id}/*.jpg
  classifier_dataset/val/{category_id}/*.jpg

Usage:
    uv run python task1_object_detection/experiments/train_yolo_classifier.py
    uv run python task1_object_detection/experiments/train_yolo_classifier.py --model yolov8s-cls.pt --epochs 30
"""

import argparse
import json
import os
from pathlib import Path

import torch
_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, "weights_only": False})

import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"

TASK_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = TASK_ROOT / "output" / "classifier_dataset"
MODELS_DIR = TASK_ROOT / "output" / "models"


def train(
    model_size: str = "yolov8s-cls.pt",
    epochs: int = 30,
    imgsz: int = 224,
    batch_size: int = 64,
    patience: int = 10,
    run_name: str = "",
):
    from ultralytics import YOLO, settings

    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"

    if not train_dir.exists():
        print(f"ERROR: {train_dir} not found. Run build_classifier_dataset.py first.")
        return

    num_classes = len([d for d in train_dir.iterdir() if d.is_dir()])
    print(f"Classes: {num_classes}")
    print(f"Train dir: {train_dir}")
    print(f"Val dir: {val_dir}")

    settings.update({"mlflow": True})
    mlflow.set_experiment("product-classifier")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    run_name = run_name or f"cls_{Path(model_size).stem}_e{epochs}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model": model_size,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch_size": batch_size,
            "num_classes": num_classes,
            "task": "classification",
        })

        model = YOLO(model_size)
        results = model.train(
            data=str(DATASET_DIR),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            patience=patience,
            project=str(MODELS_DIR),
            name=run_name,
            exist_ok=True,
            verbose=True,
        )

        # Evaluate
        best_model = MODELS_DIR / run_name / "weights" / "best.pt"
        if best_model.exists():
            val_model = YOLO(str(best_model))
            val_results = val_model.val(data=str(DATASET_DIR), imgsz=imgsz)

            top1 = float(val_results.top1) if hasattr(val_results, "top1") else 0
            top5 = float(val_results.top5) if hasattr(val_results, "top5") else 0

            mlflow.log_metrics({
                "val/top1_acc": top1,
                "val/top5_acc": top5,
            })
            mlflow.log_artifact(str(best_model), "model")

            print(f"\n{'='*60}")
            print("YOLO CLASSIFIER RESULTS")
            print(f"{'='*60}")
            print(f"  Top-1 accuracy: {top1:.4f}")
            print(f"  Top-5 accuracy: {top5:.4f}")
            print(f"  Model: {best_model}")
        else:
            print("No best.pt found — training may have failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8s-cls.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()

    train(
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        patience=args.patience,
        run_name=args.run_name,
    )
