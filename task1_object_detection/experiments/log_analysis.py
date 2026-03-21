"""
Log dataset analysis results to MLflow for tracking across iterations.
Run this after analyze_dataset.py or whenever you modify the dataset.

Usage:
    uv run python task1_object_detection/experiments/log_analysis.py
    uv run python task1_object_detection/experiments/log_analysis.py --tag "after-oversampling"
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
import numpy as np

TASK_ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_FILE = TASK_ROOT / "data" / "coco_dataset" / "train" / "annotations.json"
ANALYSIS_PLOTS_DIR = TASK_ROOT / "analysis_plots"


def log_analysis(tag: str = "initial", annotations_path: str = ""):
    ann_path = Path(annotations_path) if annotations_path else ANNOTATIONS_FILE

    print(f"Loading annotations from {ann_path}")
    with open(ann_path) as f:
        coco = json.load(f)

    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    cat_counts = Counter(ann["category_id"] for ann in coco["annotations"])

    # Include zero-count categories
    for cat in coco["categories"]:
        if cat["id"] not in cat_counts:
            cat_counts[cat["id"]] = 0

    counts = list(cat_counts.values())

    # Bbox stats
    areas = []
    for ann in coco["annotations"]:
        w, h = ann["bbox"][2], ann["bbox"][3]
        if w > 0 and h > 0:
            areas.append(w * h)

    mlflow.set_experiment("dataset-analysis")

    with mlflow.start_run(run_name=f"analysis-{tag}"):
        mlflow.set_tag("analysis_tag", tag)
        mlflow.set_tag("annotations_file", str(ann_path))

        # Dataset overview
        mlflow.log_metrics({
            "total_images": len(coco["images"]),
            "total_annotations": len(coco["annotations"]),
            "total_categories": len(coco["categories"]),
        })

        # Class distribution
        mlflow.log_metrics({
            "class_count_mean": float(np.mean(counts)),
            "class_count_median": float(np.median(counts)),
            "class_count_min": int(np.min(counts)),
            "class_count_max": int(np.max(counts)),
            "class_count_std": float(np.std(counts)),
            "classes_with_0_annotations": sum(1 for c in counts if c == 0),
            "classes_with_lt_3_annotations": sum(1 for c in counts if c < 3),
            "classes_with_lt_5_annotations": sum(1 for c in counts if c < 5),
            "classes_with_lt_10_annotations": sum(1 for c in counts if c < 10),
            "classes_with_lt_20_annotations": sum(1 for c in counts if c < 20),
            "class_imbalance_ratio": int(np.max(counts)) / max(int(np.min(counts)), 1),
        })

        # Bbox stats
        if areas:
            mlflow.log_metrics({
                "bbox_area_mean": float(np.mean(areas)),
                "bbox_area_median": float(np.median(areas)),
                "bbox_area_std": float(np.std(areas)),
            })

        # Save full distribution as artifact
        distribution = []
        for cat_id, count in sorted(cat_counts.items(), key=lambda x: x[1]):
            distribution.append({
                "category_id": cat_id,
                "name": cat_map.get(cat_id, f"unknown_{cat_id}"),
                "count": count,
            })

        dist_path = TASK_ROOT / "output" / "analysis" / f"distribution_{tag}.json"
        dist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dist_path, "w") as f:
            json.dump(distribution, f, indent=2)
        mlflow.log_artifact(str(dist_path), "distributions")

        # Log analysis plots if they exist
        if ANALYSIS_PLOTS_DIR.exists():
            for plot in ANALYSIS_PLOTS_DIR.glob("*.png"):
                mlflow.log_artifact(str(plot), "plots")

        print(f"\nLogged analysis '{tag}' to MLflow")
        print(f"  Images: {len(coco['images'])}")
        print(f"  Annotations: {len(coco['annotations'])}")
        print(f"  Categories: {len(coco['categories'])}")
        print(f"  Rare (<10 ann): {sum(1 for c in counts if c < 10)}")
        print(f"  Very rare (<3): {sum(1 for c in counts if c < 3)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="initial", help="Tag for this analysis run")
    parser.add_argument("--annotations", default="", help="Path to annotations.json")
    args = parser.parse_args()
    log_analysis(args.tag, args.annotations)
