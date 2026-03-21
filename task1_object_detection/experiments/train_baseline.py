"""
Step 2: Train a YOLOv8 baseline model with MLflow + TensorBoard tracking.

Usage:
    uv run python task1_object_detection/experiments/train_baseline.py
    uv run python task1_object_detection/experiments/train_baseline.py --model yolov8s.pt --epochs 100 --imgsz 1280

View results:
    mlflow ui                          # http://localhost:5000
    tensorboard --logdir runs/detect   # http://localhost:6006
"""

import argparse
import json
from pathlib import Path

import torch
# ultralytics 8.1.0 uses pickle-based model saves; torch 2.6+ defaults to weights_only=True
torch.serialization.add_safe_globals([])  # noqa
_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, "weights_only": False})

import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

TASK_ROOT = Path(__file__).resolve().parent.parent
YOLO_DATASET_DIR = TASK_ROOT / "output" / "yolo_dataset"
MODELS_DIR = TASK_ROOT / "output" / "models"


def train(
    model_size: str = "yolov8m.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 16,
    patience: int = 10,
    experiment_name: str = "object-detection-baseline",
    run_name: str = "",
):
    from ultralytics import YOLO, settings

    data_yaml = YOLO_DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        print(f"ERROR: {data_yaml} not found. Run prepare_dataset.py first.")
        return

    # Enable MLflow in ultralytics (auto-logs all training metrics)
    import os
    os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"
    settings.update({"mlflow": True})

    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    run_name = run_name or f"{Path(model_size).stem}_e{epochs}_img{imgsz}"

    with mlflow.start_run(run_name=run_name):
        # Log our custom params
        mlflow.log_params({
            "model_size": model_size,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch_size": batch_size,
            "patience": patience,
            "dataset": str(data_yaml),
        })

        # Log dataset stats
        remap_file = YOLO_DATASET_DIR / "category_remap.json"
        if remap_file.exists():
            with open(remap_file) as f:
                remap = json.load(f)
            mlflow.log_param("num_classes", len(remap))

        # Train
        model = YOLO(model_size)
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            patience=patience,
            project=str(MODELS_DIR),
            name=run_name,
            exist_ok=True,
            verbose=True,
            # Augmentation (no vertical flip for shelves)
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

        # Log final metrics
        best_model = MODELS_DIR / run_name / "weights" / "best.pt"
        if best_model.exists():
            mlflow.log_artifact(str(best_model), "model")

        # Validate and log per-class metrics
        print("\n=== Running validation ===")
        val_model = YOLO(str(best_model)) if best_model.exists() else model
        val_results = val_model.val(data=str(data_yaml), imgsz=imgsz)

        metrics = {
            "val/mAP50": float(val_results.box.map50),
            "val/mAP50-95": float(val_results.box.map),
            "val/precision": float(val_results.box.mp),
            "val/recall": float(val_results.box.mr),
            "estimated_competition_score": round(
                0.7 * float(val_results.box.map50) + 0.3 * float(val_results.box.map50), 4
            ),
        }
        mlflow.log_metrics(metrics)

        # Per-class AP
        if hasattr(val_results.box, "ap50") and val_results.box.ap50 is not None:
            per_class = []
            ap50 = val_results.box.ap50
            names = val_results.names if hasattr(val_results, "names") else {}
            for i, ap in enumerate(ap50):
                per_class.append({
                    "class_id": i,
                    "name": names.get(i, str(i)),
                    "ap50": round(float(ap), 4),
                })
            per_class.sort(key=lambda x: x["ap50"])

            # Save per-class metrics as artifact
            per_class_path = MODELS_DIR / run_name / "per_class_metrics.json"
            with open(per_class_path, "w") as f:
                json.dump(per_class, f, indent=2)
            mlflow.log_artifact(str(per_class_path), "metrics")

            # Log zero-AP classes count
            zero_ap = [c for c in per_class if c["ap50"] == 0]
            mlflow.log_metric("zero_ap_classes", len(zero_ap))

            print(f"\n=== Results ===")
            print(f"mAP@0.5: {metrics['val/mAP50']:.4f}")
            print(f"Precision: {metrics['val/precision']:.4f}")
            print(f"Recall: {metrics['val/recall']:.4f}")
            print(f"Est. competition score: {metrics['estimated_competition_score']:.4f}")
            print(f"Zero-AP classes: {len(zero_ap)} / {len(per_class)}")
            print(f"\nWeakest 10 classes:")
            for c in per_class[:10]:
                print(f"  {c['class_id']:3d} {c['name'][:50]:50s} AP={c['ap50']:.4f}")

    print(f"\nResults logged to MLflow experiment '{experiment_name}'")
    print(f"View with: mlflow ui")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8m.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--experiment", default="object-detection-baseline")
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()

    train(
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        patience=args.patience,
        experiment_name=args.experiment,
        run_name=args.run_name,
    )
