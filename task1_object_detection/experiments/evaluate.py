"""
Step 3: Evaluate a trained model and log detailed per-class metrics to MLflow.
Use this to compare iterations and identify what improved.

Usage:
    uv run python task1_object_detection/experiments/evaluate.py
    uv run python task1_object_detection/experiments/evaluate.py --model output/models/yolov8m_e50_img640/weights/best.pt
    uv run python task1_object_detection/experiments/evaluate.py --model best.pt --tag "after-oversampling"
"""

import argparse
import json
from pathlib import Path

import torch
_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, "weights_only": False})

import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

TASK_ROOT = Path(__file__).resolve().parent.parent
YOLO_DATASET_DIR = TASK_ROOT / "output" / "yolo_dataset"
MODELS_DIR = TASK_ROOT / "output" / "models"


def evaluate(
    model_path: str = "",
    data_yaml: str = "",
    imgsz: int = 640,
    tag: str = "",
    experiment_name: str = "object-detection-evaluation",
):
    from ultralytics import YOLO

    # Find model
    if not model_path:
        # Look for the most recent best.pt
        candidates = list(MODELS_DIR.rglob("best.pt"))
        if not candidates:
            print("ERROR: No trained model found. Run train_baseline.py first.")
            return
        model_path = str(max(candidates, key=lambda p: p.stat().st_mtime))
        print(f"Using most recent model: {model_path}")

    data_yaml = data_yaml or str(YOLO_DATASET_DIR / "data.yaml")

    model = YOLO(model_path)
    results = model.val(data=data_yaml, imgsz=imgsz)

    # Compute metrics
    map50 = float(results.box.map50)
    map50_95 = float(results.box.map)
    precision = float(results.box.mp)
    recall = float(results.box.mr)
    est_score = round(0.7 * map50 + 0.3 * map50, 4)

    # Per-class breakdown
    per_class = []
    if hasattr(results.box, "ap50") and results.box.ap50 is not None:
        ap50 = results.box.ap50
        names = results.names if hasattr(results, "names") else {}
        for i, ap in enumerate(ap50):
            per_class.append({
                "class_id": i,
                "name": names.get(i, str(i)),
                "ap50": round(float(ap), 4),
            })
        per_class.sort(key=lambda x: x["ap50"])

    zero_ap = [c for c in per_class if c["ap50"] == 0]
    low_ap = [c for c in per_class if 0 < c["ap50"] < 0.3]

    # Log to MLflow
    mlflow.set_experiment(experiment_name)
    run_name = tag or Path(model_path).parent.parent.name

    with mlflow.start_run(run_name=run_name):
        if tag:
            mlflow.set_tag("eval_tag", tag)
        mlflow.set_tag("model_path", model_path)

        mlflow.log_metrics({
            "mAP50": map50,
            "mAP50-95": map50_95,
            "precision": precision,
            "recall": recall,
            "estimated_competition_score": est_score,
            "zero_ap_classes": len(zero_ap),
            "low_ap_classes": len(low_ap),
            "total_classes_evaluated": len(per_class),
        })

        # Save per-class as artifact
        if per_class:
            out_path = TASK_ROOT / "output" / "analysis" / f"eval_{run_name}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(per_class, f, indent=2)
            mlflow.log_artifact(str(out_path), "per_class_metrics")

    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION: {run_name}")
    print(f"{'='*60}")
    print(f"  mAP@0.5:       {map50:.4f}")
    print(f"  mAP@0.5:0.95:  {map50_95:.4f}")
    print(f"  Precision:      {precision:.4f}")
    print(f"  Recall:         {recall:.4f}")
    print(f"  Est. score:     {est_score:.4f}")
    print(f"  Zero-AP:        {len(zero_ap)} / {len(per_class)} classes")
    print(f"  Low-AP (<0.3):  {len(low_ap)} / {len(per_class)} classes")

    if per_class:
        print(f"\n  Weakest 10:")
        for c in per_class[:10]:
            print(f"    {c['class_id']:3d} {c['name'][:45]:45s} AP={c['ap50']:.4f}")
        print(f"\n  Strongest 10:")
        for c in per_class[-10:][::-1]:
            print(f"    {c['class_id']:3d} {c['name'][:45]:45s} AP={c['ap50']:.4f}")

    print(f"\nLogged to MLflow experiment '{experiment_name}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="")
    parser.add_argument("--data", default="")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--tag", default="")
    parser.add_argument("--experiment", default="object-detection-evaluation")
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        data_yaml=args.data,
        imgsz=args.imgsz,
        tag=args.tag,
        experiment_name=args.experiment,
    )
