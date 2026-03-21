"""LangChain tools for object detection dataset analysis, augmentation, and training."""

import json
import logging
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from langchain_core.tools import tool

from task1_object_detection.agent.config import (
    ANALYSIS_DIR,
    ANNOTATIONS_FILE,
    AUGMENTED_DIR,
    COCO_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_IMGSZ,
    DEFAULT_MODEL_SIZE,
    DEFAULT_PATIENCE,
    IMAGES_DIR,
    INFERENCE_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    RARE_CLASS_THRESHOLD,
    VERY_RARE_THRESHOLD,
    YOLO_DATASET_DIR,
)

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _load_coco(annotations_path: str = "") -> dict:
    """Load COCO annotations JSON."""
    path = Path(annotations_path) if annotations_path else ANNOTATIONS_FILE
    if not path.exists():
        raise FileNotFoundError(f"Annotations not found at {path}")
    with open(path) as f:
        return json.load(f)


def _category_map(coco: dict) -> dict[int, str]:
    """Build category_id → name mapping."""
    return {c["id"]: c["name"] for c in coco.get("categories", [])}


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


@tool
def analyze_class_distribution(annotations_path: str = "") -> str:
    """Load COCO annotations and return category distribution statistics.

    Identifies rare and common classes, computes mean/median/min/max annotations
    per category. Essential first step for understanding class imbalance.

    Args:
        annotations_path: Optional path to annotations.json. Uses default if empty.

    Returns:
        JSON string with distribution stats, sorted by count ascending.
    """
    coco = _load_coco(annotations_path)
    cat_map = _category_map(coco)
    cat_counts = Counter(ann["category_id"] for ann in coco["annotations"])

    # Include categories with 0 annotations
    for cat in coco.get("categories", []):
        if cat["id"] not in cat_counts:
            cat_counts[cat["id"]] = 0

    counts = sorted(cat_counts.values())
    total_anns = sum(counts)
    num_cats = len(counts)

    distribution = []
    for cat_id, count in sorted(cat_counts.items(), key=lambda x: x[1]):
        distribution.append({
            "category_id": cat_id,
            "name": cat_map.get(cat_id, f"unknown_{cat_id}"),
            "count": count,
            "percentage": round(count / total_anns * 100, 3) if total_anns > 0 else 0,
        })

    rare_classes = [d for d in distribution if d["count"] < RARE_CLASS_THRESHOLD]
    very_rare = [d for d in distribution if d["count"] < VERY_RARE_THRESHOLD]

    result = {
        "total_annotations": total_anns,
        "total_categories": num_cats,
        "total_images": len(coco.get("images", [])),
        "stats": {
            "mean": round(np.mean(counts), 2),
            "median": float(np.median(counts)),
            "min": int(min(counts)),
            "max": int(max(counts)),
            "std": round(float(np.std(counts)), 2),
        },
        "rare_classes_count": len(rare_classes),
        "very_rare_count": len(very_rare),
        "top_10_most_common": distribution[-10:][::-1],
        "top_10_rarest": distribution[:10],
        "zero_annotation_categories": [d for d in distribution if d["count"] == 0],
        "full_distribution_saved_to": str(ANALYSIS_DIR / "class_distribution.json"),
    }

    # Save full distribution
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ANALYSIS_DIR / "class_distribution.json", "w") as f:
        json.dump(distribution, f, indent=2)

    return json.dumps(result, indent=2)


@tool
def analyze_bbox_distribution(annotations_path: str = "") -> str:
    """Analyze bounding box sizes, aspect ratios, and positions across the dataset.

    Helps identify if certain bbox sizes are under-represented and guides
    augmentation strategies (e.g., scale augmentation range).

    Args:
        annotations_path: Optional path to annotations.json.

    Returns:
        JSON string with bbox size stats, aspect ratio distribution, position heatmap summary.
    """
    coco = _load_coco(annotations_path)

    # Build image dimension lookup
    img_dims = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}

    widths, heights, areas, aspect_ratios = [], [], [], []
    rel_widths, rel_heights = [], []
    cx_positions, cy_positions = [], []

    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue

        widths.append(w)
        heights.append(h)
        areas.append(w * h)
        aspect_ratios.append(w / h)

        img_id = ann["image_id"]
        if img_id in img_dims:
            iw, ih = img_dims[img_id]
            rel_widths.append(w / iw)
            rel_heights.append(h / ih)
            cx_positions.append((x + w / 2) / iw)
            cy_positions.append((y + h / 2) / ih)

    def stats(arr):
        a = np.array(arr)
        return {
            "mean": round(float(np.mean(a)), 2),
            "median": round(float(np.median(a)), 2),
            "min": round(float(np.min(a)), 2),
            "max": round(float(np.max(a)), 2),
            "std": round(float(np.std(a)), 2),
            "p5": round(float(np.percentile(a, 5)), 2),
            "p95": round(float(np.percentile(a, 95)), 2),
        }

    # Size buckets (relative to image)
    rel_areas = [rw * rh for rw, rh in zip(rel_widths, rel_heights)]
    small = sum(1 for a in rel_areas if a < 0.01)
    medium = sum(1 for a in rel_areas if 0.01 <= a < 0.1)
    large = sum(1 for a in rel_areas if a >= 0.1)

    result = {
        "total_bboxes": len(widths),
        "bbox_width_px": stats(widths),
        "bbox_height_px": stats(heights),
        "bbox_area_px": stats(areas),
        "aspect_ratio": stats(aspect_ratios),
        "relative_width": stats(rel_widths) if rel_widths else {},
        "relative_height": stats(rel_heights) if rel_heights else {},
        "size_distribution": {
            "small_lt_1pct": small,
            "medium_1_10pct": medium,
            "large_gt_10pct": large,
        },
        "center_position": {
            "cx": stats(cx_positions) if cx_positions else {},
            "cy": stats(cy_positions) if cy_positions else {},
        },
    }

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ANALYSIS_DIR / "bbox_distribution.json", "w") as f:
        json.dump(result, f, indent=2)

    return json.dumps(result, indent=2)


@tool
def analyze_image_stats(annotations_path: str = "") -> str:
    """Analyze per-image annotation counts and image resolution statistics.

    Identifies images with very few or very many annotations, and resolution
    variation across the dataset.

    Args:
        annotations_path: Optional path to annotations.json.

    Returns:
        JSON string with per-image stats and resolution info.
    """
    coco = _load_coco(annotations_path)

    img_ann_counts = Counter(ann["image_id"] for ann in coco["annotations"])
    resolutions = [(img["width"], img["height"]) for img in coco["images"]]
    widths_img = [r[0] for r in resolutions]
    heights_img = [r[1] for r in resolutions]

    # Images with no annotations
    all_img_ids = {img["id"] for img in coco["images"]}
    annotated_ids = set(img_ann_counts.keys())
    unannotated = all_img_ids - annotated_ids

    counts = list(img_ann_counts.values())
    sorted_counts = sorted(img_ann_counts.items(), key=lambda x: x[1])

    result = {
        "total_images": len(coco["images"]),
        "annotated_images": len(annotated_ids),
        "unannotated_images": len(unannotated),
        "annotations_per_image": {
            "mean": round(np.mean(counts), 2) if counts else 0,
            "median": float(np.median(counts)) if counts else 0,
            "min": min(counts) if counts else 0,
            "max": max(counts) if counts else 0,
            "std": round(float(np.std(counts)), 2) if counts else 0,
        },
        "images_with_fewest_annotations": [
            {"image_id": img_id, "count": c} for img_id, c in sorted_counts[:5]
        ],
        "images_with_most_annotations": [
            {"image_id": img_id, "count": c} for img_id, c in sorted_counts[-5:][::-1]
        ],
        "resolution": {
            "widths": {"min": min(widths_img), "max": max(widths_img), "unique": len(set(widths_img))},
            "heights": {"min": min(heights_img), "max": max(heights_img), "unique": len(set(heights_img))},
            "unique_resolutions": len(set(resolutions)),
        },
    }

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ANALYSIS_DIR / "image_stats.json", "w") as f:
        json.dump(result, f, indent=2)

    return json.dumps(result, indent=2)


@tool
def identify_weak_categories(min_annotations: int = 10, annotations_path: str = "") -> str:
    """Find categories with fewer than N annotations and suggest improvement strategies.

    Args:
        min_annotations: Threshold below which a category is considered weak. Default 10.
        annotations_path: Optional path to annotations.json.

    Returns:
        JSON with weak categories and suggested strategies for each.
    """
    coco = _load_coco(annotations_path)
    cat_map = _category_map(coco)
    cat_counts = Counter(ann["category_id"] for ann in coco["annotations"])

    # Include zero-count categories
    for cat in coco.get("categories", []):
        if cat["id"] not in cat_counts:
            cat_counts[cat["id"]] = 0

    weak = []
    for cat_id, count in sorted(cat_counts.items(), key=lambda x: x[1]):
        if count >= min_annotations:
            continue

        strategies = []
        if count == 0:
            strategies.append("CRITICAL: No training data. Consider removing or finding external data.")
        elif count < VERY_RARE_THRESHOLD:
            strategies.append("Very rare: Heavy oversampling (5-10x), copy-paste augmentation.")
            strategies.append("Consider merging with visually similar categories.")
        else:
            strategies.append(f"Rare ({count} samples): Oversample 3-5x, strong augmentation.")

        strategies.append("Use mosaic augmentation to combine with other images.")

        weak.append({
            "category_id": cat_id,
            "name": cat_map.get(cat_id, f"unknown_{cat_id}"),
            "count": count,
            "strategies": strategies,
        })

    result = {
        "threshold": min_annotations,
        "total_weak_categories": len(weak),
        "total_categories": len(cat_counts),
        "weak_percentage": round(len(weak) / len(cat_counts) * 100, 1),
        "weak_categories": weak,
    }

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ANALYSIS_DIR / "weak_categories.json", "w") as f:
        json.dump(result, f, indent=2)

    return json.dumps(result, indent=2)


@tool
def visualize_annotations(num_samples: int = 5, annotations_path: str = "") -> str:
    """Visualize annotations on sample images using the supervision library.

    Saves annotated images to the analysis output directory.

    Args:
        num_samples: Number of sample images to visualize. Default 5.
        annotations_path: Optional path to annotations.json.

    Returns:
        JSON with paths to saved visualization images.
    """
    try:
        import supervision as sv
        from PIL import Image
    except ImportError:
        return json.dumps({"error": "supervision or PIL not installed. Install with: pip install supervision Pillow"})

    coco = _load_coco(annotations_path)
    cat_map = _category_map(coco)

    # Group annotations by image
    img_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    img_lookup = {img["id"]: img for img in coco["images"]}

    # Sample images (prefer those with many annotations)
    sorted_imgs = sorted(img_anns.keys(), key=lambda x: len(img_anns[x]), reverse=True)
    sample_ids = sorted_imgs[:num_samples]

    vis_dir = ANALYSIS_DIR / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for img_id in sample_ids:
        img_info = img_lookup.get(img_id)
        if not img_info:
            continue

        img_path = IMAGES_DIR / img_info["file_name"]
        if not img_path.exists():
            continue

        image = np.array(Image.open(img_path).convert("RGB"))
        anns = img_anns[img_id]

        xyxy = []
        class_ids = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            xyxy.append([x, y, x + w, y + h])
            class_ids.append(ann["category_id"])
            name = cat_map.get(ann["category_id"], str(ann["category_id"]))
            labels.append(f"{name}")

        detections = sv.Detections(
            xyxy=np.array(xyxy, dtype=np.float32),
            class_id=np.array(class_ids, dtype=int),
        )

        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_scale=0.3, text_thickness=1)

        annotated = box_annotator.annotate(scene=image.copy(), detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        out_path = vis_dir / f"sample_{img_id}.jpg"
        Image.fromarray(annotated).save(out_path, quality=90)
        saved_paths.append(str(out_path))

    return json.dumps({
        "num_visualized": len(saved_paths),
        "saved_to": saved_paths,
        "directory": str(vis_dir),
    }, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET BOOST TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


@tool
def create_yolo_dataset(
    val_split: float = 0.15,
    annotations_path: str = "",
    seed: int = 42,
) -> str:
    """Convert COCO annotations to YOLO format and create train/val split.

    Creates the directory structure expected by ultralytics:
      yolo_dataset/
        images/train/
        images/val/
        labels/train/
        labels/val/
        data.yaml

    Args:
        val_split: Fraction of images for validation. Default 0.15.
        annotations_path: Optional path to annotations.json.
        seed: Random seed for reproducible splits.

    Returns:
        JSON with dataset stats and path to data.yaml.
    """
    coco = _load_coco(annotations_path)
    cat_map = _category_map(coco)
    img_lookup = {img["id"]: img for img in coco["images"]}

    # Group annotations by image
    img_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    # Create directories
    for split in ["train", "val"]:
        (YOLO_DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Split images
    image_ids = list(img_lookup.keys())
    random.seed(seed)
    random.shuffle(image_ids)
    val_count = max(1, int(len(image_ids) * val_split))
    val_ids = set(image_ids[:val_count])
    train_ids = set(image_ids[val_count:])

    # Build category ID remapping (YOLO needs 0-indexed contiguous IDs)
    all_cat_ids = sorted(cat_map.keys())
    cat_remap = {old_id: new_id for new_id, old_id in enumerate(all_cat_ids)}

    stats = {"train_images": 0, "val_images": 0, "train_annotations": 0, "val_annotations": 0}

    for img_id in image_ids:
        img_info = img_lookup[img_id]
        split = "val" if img_id in val_ids else "train"

        # Copy image
        src_path = IMAGES_DIR / img_info["file_name"]
        if not src_path.exists():
            continue
        dst_img = YOLO_DATASET_DIR / "images" / split / img_info["file_name"]
        shutil.copy2(src_path, dst_img)

        # Write YOLO label file
        label_name = Path(img_info["file_name"]).stem + ".txt"
        label_path = YOLO_DATASET_DIR / "labels" / split / label_name

        iw, ih = img_info["width"], img_info["height"]
        lines = []
        for ann in img_anns.get(img_id, []):
            x, y, w, h = ann["bbox"]
            # Convert COCO (x,y,w,h) to YOLO (cx,cy,w,h) normalized
            cx = (x + w / 2) / iw
            cy = (y + h / 2) / ih
            nw = w / iw
            nh = h / ih
            # Clamp to [0, 1]
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            nw = max(0, min(1, nw))
            nh = max(0, min(1, nh))

            class_id = cat_remap.get(ann["category_id"], ann["category_id"])
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

        stats[f"{split}_images"] += 1
        stats[f"{split}_annotations"] += len(lines)

    # Write data.yaml
    names = {cat_remap[old_id]: name for old_id, name in cat_map.items() if old_id in cat_remap}
    data_yaml = {
        "path": str(YOLO_DATASET_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": names,
    }

    yaml_path = YOLO_DATASET_DIR / "data.yaml"
    # Write YAML manually to avoid pyyaml dependency issues
    with open(yaml_path, "w") as f:
        f.write(f"path: {data_yaml['path']}\n")
        f.write(f"train: {data_yaml['train']}\n")
        f.write(f"val: {data_yaml['val']}\n")
        f.write(f"nc: {data_yaml['nc']}\n")
        f.write("names:\n")
        for idx in sorted(names.keys()):
            f.write(f"  {idx}: {names[idx]}\n")

    # Save category remapping for reference
    with open(YOLO_DATASET_DIR / "category_remap.json", "w") as f:
        json.dump({str(k): v for k, v in cat_remap.items()}, f, indent=2)

    result = {
        **stats,
        "num_classes": len(names),
        "data_yaml": str(yaml_path),
        "dataset_dir": str(YOLO_DATASET_DIR),
        "category_remap_saved": str(YOLO_DATASET_DIR / "category_remap.json"),
    }

    return json.dumps(result, indent=2)


@tool
def apply_oversampling(
    oversample_factor: int = 3,
    min_annotations_threshold: int = 10,
    annotations_path: str = "",
) -> str:
    """Duplicate images containing rare classes to balance the dataset.

    Works on an already-created YOLO dataset. Copies images and labels
    with a suffix to increase representation of rare classes.

    Args:
        oversample_factor: How many times to duplicate rare-class images. Default 3.
        min_annotations_threshold: Categories below this count get oversampled. Default 10.
        annotations_path: Optional path to annotations.json.

    Returns:
        JSON with oversampling stats.
    """
    coco = _load_coco(annotations_path)
    cat_counts = Counter(ann["category_id"] for ann in coco["annotations"])

    # Find rare categories
    rare_cats = {cat_id for cat_id, count in cat_counts.items() if count < min_annotations_threshold}

    if not rare_cats:
        return json.dumps({"message": "No rare categories found below threshold.", "threshold": min_annotations_threshold})

    # Find images containing rare categories
    img_anns = defaultdict(set)
    for ann in coco["annotations"]:
        img_anns[ann["image_id"]].add(ann["category_id"])

    rare_images = set()
    for img_id, cats in img_anns.items():
        if cats & rare_cats:
            rare_images.add(img_id)

    img_lookup = {img["id"]: img for img in coco["images"]}

    # Check if YOLO dataset exists
    train_img_dir = YOLO_DATASET_DIR / "images" / "train"
    train_lbl_dir = YOLO_DATASET_DIR / "labels" / "train"
    if not train_img_dir.exists():
        return json.dumps({"error": "YOLO dataset not found. Run create_yolo_dataset first."})

    duplicated = 0
    for img_id in rare_images:
        img_info = img_lookup.get(img_id)
        if not img_info:
            continue

        stem = Path(img_info["file_name"]).stem
        suffix = Path(img_info["file_name"]).suffix

        src_img = train_img_dir / img_info["file_name"]
        src_lbl = train_lbl_dir / f"{stem}.txt"

        if not src_img.exists():
            continue

        for i in range(oversample_factor):
            dst_img = train_img_dir / f"{stem}_dup{i}{suffix}"
            dst_lbl = train_lbl_dir / f"{stem}_dup{i}.txt"

            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)
            if src_lbl.exists() and not dst_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
            duplicated += 1

    result = {
        "rare_categories_count": len(rare_cats),
        "images_containing_rare_classes": len(rare_images),
        "oversample_factor": oversample_factor,
        "total_duplicates_created": duplicated,
        "threshold": min_annotations_threshold,
    }

    return json.dumps(result, indent=2)


@tool
def generate_augmentation_config(
    mosaic: float = 1.0,
    mixup: float = 0.15,
    copy_paste: float = 0.1,
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
    degrees: float = 10.0,
    translate: float = 0.1,
    scale: float = 0.5,
    fliplr: float = 0.5,
    flipud: float = 0.0,
    erasing: float = 0.1,
) -> str:
    """Generate an optimal ultralytics augmentation config based on analysis.

    Creates a YAML config file for ultralytics training augmentation parameters.

    Args:
        mosaic: Mosaic augmentation probability. Default 1.0.
        mixup: MixUp augmentation probability. Default 0.15.
        copy_paste: Copy-paste augmentation probability. Default 0.1.
        hsv_h: HSV-Hue augmentation range. Default 0.015.
        hsv_s: HSV-Saturation augmentation range. Default 0.7.
        hsv_v: HSV-Value augmentation range. Default 0.4.
        degrees: Rotation augmentation degrees. Default 10.0.
        translate: Translation augmentation fraction. Default 0.1.
        scale: Scale augmentation range. Default 0.5.
        fliplr: Horizontal flip probability. Default 0.5.
        flipud: Vertical flip probability. Default 0.0.
        erasing: Random erasing probability. Default 0.1.

    Returns:
        JSON with path to saved augmentation config.
    """
    config = {
        "mosaic": mosaic,
        "mixup": mixup,
        "copy_paste": copy_paste,
        "hsv_h": hsv_h,
        "hsv_s": hsv_s,
        "hsv_v": hsv_v,
        "degrees": degrees,
        "translate": translate,
        "scale": scale,
        "fliplr": fliplr,
        "flipud": flipud,
        "erasing": erasing,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config_path = OUTPUT_DIR / "augmentation_config.yaml"

    with open(config_path, "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    return json.dumps({
        "config": config,
        "saved_to": str(config_path),
        "usage": f"Pass these as keyword arguments to train_yolo_model, or use --cfg {config_path}",
    }, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


@tool
def train_yolo_model(
    model_size: str = "",
    epochs: int = 0,
    imgsz: int = 0,
    batch_size: int = 0,
    data_yaml: str = "",
    patience: int = 0,
    extra_args: str = "{}",
) -> str:
    """Train a YOLOv8 model using ultralytics.

    Args:
        model_size: Model checkpoint, e.g. "yolov8n.pt", "yolov8s.pt", "yolov8m.pt". Default "yolov8m.pt".
        epochs: Number of training epochs. Default 50.
        imgsz: Image size for training. Default 640.
        batch_size: Batch size. Default 16.
        data_yaml: Path to data.yaml. Uses default YOLO dataset if empty.
        patience: Early stopping patience. Default 10.
        extra_args: JSON string of additional ultralytics training arguments.

    Returns:
        JSON with training results, metrics, and model path.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return json.dumps({"error": "ultralytics not installed. Install with: pip install ultralytics==8.1.0"})

    model_size = model_size or DEFAULT_MODEL_SIZE
    epochs = epochs or DEFAULT_EPOCHS
    imgsz = imgsz or DEFAULT_IMGSZ
    batch_size = batch_size or DEFAULT_BATCH_SIZE
    patience = patience or DEFAULT_PATIENCE
    data_yaml = data_yaml or str(YOLO_DATASET_DIR / "data.yaml")

    if not Path(data_yaml).exists():
        return json.dumps({"error": f"data.yaml not found at {data_yaml}. Run create_yolo_dataset first."})

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    extra = json.loads(extra_args) if extra_args and extra_args != "{}" else {}

    model = YOLO(model_size)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        patience=patience,
        project=str(MODELS_DIR),
        name="train",
        exist_ok=True,
        verbose=True,
        **extra,
    )

    # Gather result info
    best_model = MODELS_DIR / "train" / "weights" / "best.pt"
    last_model = MODELS_DIR / "train" / "weights" / "last.pt"

    result = {
        "status": "completed",
        "model_size": model_size,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch_size": batch_size,
        "best_model": str(best_model) if best_model.exists() else "not found",
        "last_model": str(last_model) if last_model.exists() else "not found",
        "results_dir": str(MODELS_DIR / "train"),
    }

    # Try to read metrics
    metrics_file = MODELS_DIR / "train" / "results.csv"
    if metrics_file.exists():
        import csv
        with open(metrics_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                last_row = rows[-1]
                result["final_metrics"] = {k.strip(): v.strip() for k, v in last_row.items()}

    return json.dumps(result, indent=2)


@tool
def evaluate_model(model_path: str = "", data_yaml: str = "", imgsz: int = 0) -> str:
    """Run validation on a trained model and return per-class mAP.

    Args:
        model_path: Path to trained model weights (.pt). Uses best.pt if empty.
        data_yaml: Path to data.yaml. Uses default if empty.
        imgsz: Image size for validation. Default 640.

    Returns:
        JSON with overall and per-class mAP metrics.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return json.dumps({"error": "ultralytics not installed."})

    model_path = model_path or str(MODELS_DIR / "train" / "weights" / "best.pt")
    data_yaml = data_yaml or str(YOLO_DATASET_DIR / "data.yaml")
    imgsz = imgsz or DEFAULT_IMGSZ

    if not Path(model_path).exists():
        return json.dumps({"error": f"Model not found at {model_path}. Train a model first."})

    model = YOLO(model_path)
    results = model.val(data=data_yaml, imgsz=imgsz, verbose=True)

    result = {
        "model": model_path,
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
    }

    # Per-class metrics
    if hasattr(results.box, "ap50") and results.box.ap50 is not None:
        per_class = []
        ap50 = results.box.ap50
        names = results.names if hasattr(results, "names") else {}
        for i, ap in enumerate(ap50):
            per_class.append({
                "class_id": i,
                "name": names.get(i, str(i)),
                "ap50": round(float(ap), 4),
            })
        # Sort by AP to find weakest
        per_class.sort(key=lambda x: x["ap50"])
        result["weakest_10_classes"] = per_class[:10]
        result["strongest_10_classes"] = per_class[-10:][::-1]
        result["classes_with_zero_ap"] = [c for c in per_class if c["ap50"] == 0]

        # Save full per-class results
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        with open(ANALYSIS_DIR / "per_class_metrics.json", "w") as f:
            json.dump(per_class, f, indent=2)
        result["full_metrics_saved_to"] = str(ANALYSIS_DIR / "per_class_metrics.json")

    # Estimated competition score
    result["estimated_score"] = round(
        0.7 * result["map50"] + 0.3 * result["map50"],  # simplified
        4,
    )

    return json.dumps(result, indent=2)


@tool
def export_model(model_path: str = "", export_format: str = "onnx") -> str:
    """Export a trained model to ONNX or keep as .pt.

    Args:
        model_path: Path to trained model weights. Uses best.pt if empty.
        export_format: Export format: "onnx", "torchscript", or "pt" (no-op). Default "onnx".

    Returns:
        JSON with export path.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return json.dumps({"error": "ultralytics not installed."})

    model_path = model_path or str(MODELS_DIR / "train" / "weights" / "best.pt")

    if not Path(model_path).exists():
        return json.dumps({"error": f"Model not found at {model_path}"})

    if export_format == "pt":
        return json.dumps({"format": "pt", "path": model_path, "message": "No export needed for .pt format."})

    model = YOLO(model_path)
    export_path = model.export(format=export_format)

    return json.dumps({
        "format": export_format,
        "source": model_path,
        "exported_to": str(export_path),
    }, indent=2)


@tool
def run_inference(
    model_path: str = "",
    source_dir: str = "",
    num_images: int = 10,
    conf_threshold: float = 0.25,
) -> str:
    """Run inference on sample images and save visualized results.

    Args:
        model_path: Path to trained model. Uses best.pt if empty.
        source_dir: Directory with images. Uses validation images if empty.
        num_images: Max number of images to run inference on. Default 10.
        conf_threshold: Confidence threshold. Default 0.25.

    Returns:
        JSON with inference results and saved visualization paths.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return json.dumps({"error": "ultralytics not installed."})

    model_path = model_path or str(MODELS_DIR / "train" / "weights" / "best.pt")
    if not Path(model_path).exists():
        return json.dumps({"error": f"Model not found at {model_path}"})

    # Determine source images
    if source_dir:
        src = Path(source_dir)
    else:
        src = YOLO_DATASET_DIR / "images" / "val"
        if not src.exists():
            src = IMAGES_DIR

    image_files = sorted(src.glob("*.jpg")) + sorted(src.glob("*.png")) + sorted(src.glob("*.jpeg"))
    image_files = image_files[:num_images]

    if not image_files:
        return json.dumps({"error": f"No images found in {src}"})

    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    results = model.predict(
        source=[str(f) for f in image_files],
        conf=conf_threshold,
        save=True,
        project=str(INFERENCE_DIR),
        name="predict",
        exist_ok=True,
    )

    detections_summary = []
    for r in results:
        detections_summary.append({
            "image": Path(r.path).name,
            "num_detections": len(r.boxes),
            "classes_detected": list(set(int(c) for c in r.boxes.cls.tolist())) if len(r.boxes) > 0 else [],
        })

    return json.dumps({
        "num_images": len(image_files),
        "results_dir": str(INFERENCE_DIR / "predict"),
        "detections": detections_summary,
    }, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# HUGGINGFACE TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


@tool
def search_hf_models(query: str, limit: int = 5) -> str:
    """Search HuggingFace for relevant pretrained object detection models.

    Useful for finding pretrained models for grocery/retail detection that
    could be used as starting weights or for ensemble.

    Args:
        query: Search query, e.g. "grocery detection", "retail product yolo".
        limit: Maximum number of results. Default 5.

    Returns:
        JSON with model information.
    """
    try:
        import requests as req

        resp = req.get(
            "https://huggingface.co/api/models",
            params={
                "search": query,
                "filter": "object-detection",
                "sort": "downloads",
                "direction": "-1",
                "limit": limit,
            },
            timeout=15,
        )
        resp.raise_for_status()
        models = resp.json()

        results = []
        for m in models:
            results.append({
                "model_id": m.get("modelId", ""),
                "downloads": m.get("downloads", 0),
                "likes": m.get("likes", 0),
                "pipeline_tag": m.get("pipeline_tag", ""),
                "tags": m.get("tags", [])[:10],
                "url": f"https://huggingface.co/{m.get('modelId', '')}",
            })

        return json.dumps({"query": query, "results": results}, indent=2)

    except Exception as e:
        return json.dumps({"error": f"HuggingFace search failed: {e}"})


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL COLLECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

ANALYZER_TOOLS = [
    analyze_class_distribution,
    analyze_bbox_distribution,
    analyze_image_stats,
    identify_weak_categories,
    visualize_annotations,
    search_hf_models,
]

BOOSTER_TOOLS = [
    analyze_class_distribution,
    identify_weak_categories,
    create_yolo_dataset,
    apply_oversampling,
    generate_augmentation_config,
]

TRAINER_TOOLS = [
    train_yolo_model,
    evaluate_model,
    export_model,
    run_inference,
    analyze_class_distribution,
    identify_weak_categories,
]

ALL_TOOLS = list({id(t): t for tools in [ANALYZER_TOOLS, BOOSTER_TOOLS, TRAINER_TOOLS] for t in tools}.values())
