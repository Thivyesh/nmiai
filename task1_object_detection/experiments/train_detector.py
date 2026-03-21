"""
Train a 1-class product detector (detection only, no classification).
This targets the 70% detection component of the competition score.

Usage:
    uv run python task1_object_detection/experiments/train_detector.py
    uv run python task1_object_detection/experiments/train_detector.py --model yolov8m.pt --epochs 50 --imgsz 1280
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
DATASET_DIR = TASK_ROOT / "output" / "yolo_dataset_1class"
MODELS_DIR = TASK_ROOT / "output" / "models"


def train(
    model_size: str = "yolov8m.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 16,
    patience: int = 15,
    experiment_name: str = "product-detector-1class",
    run_name: str = "",
):
    from ultralytics import YOLO, settings

    data_yaml = DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        print(f"ERROR: {data_yaml} not found. Run prepare_single_class.py first.")
        return

    settings.update({"mlflow": True})
    mlflow.set_experiment(experiment_name)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    run_name = run_name or f"detector_{Path(model_size).stem}_e{epochs}_img{imgsz}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model_size": model_size,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch_size": batch_size,
            "patience": patience,
            "num_classes": 1,
            "task": "detection_only",
        })

        model = YOLO(model_size)
        model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            patience=patience,
            project=str(MODELS_DIR),
            name=run_name,
            exist_ok=True,
            verbose=True,
            # Augmentation
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
        )

        # Evaluate
        best_model = MODELS_DIR / run_name / "weights" / "best.pt"
        if best_model.exists():
            val_model = YOLO(str(best_model))
            results = val_model.val(data=str(data_yaml), imgsz=imgsz)

            metrics = {
                "val/mAP50": float(results.box.map50),
                "val/mAP50-95": float(results.box.map),
                "val/precision": float(results.box.mp),
                "val/recall": float(results.box.mr),
                "detection_score_ceiling": round(0.7 * float(results.box.map50), 4),
            }
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(best_model), "model")

            print(f"\n{'='*60}")
            print("DETECTOR RESULTS (1 class)")
            print(f"{'='*60}")
            print(f"  mAP@0.5:     {metrics['val/mAP50']:.4f}")
            print(f"  Precision:   {metrics['val/precision']:.4f}")
            print(f"  Recall:      {metrics['val/recall']:.4f}")
            print(f"  Detection score ceiling (70%): {metrics['detection_score_ceiling']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8m.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15)
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
