"""
Step 1: Convert COCO dataset to YOLO format with train/val split.
Run this before training.

Usage:
    uv run python task1_object_detection/experiments/prepare_dataset.py
    uv run python task1_object_detection/experiments/prepare_dataset.py --val-split 0.2
"""

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

# Paths
TASK_ROOT = Path(__file__).resolve().parent.parent
COCO_DIR = TASK_ROOT / "data" / "coco_dataset" / "train"
ANNOTATIONS_FILE = COCO_DIR / "annotations.json"
IMAGES_DIR = COCO_DIR / "images"
OUTPUT_DIR = TASK_ROOT / "output" / "yolo_dataset"


def convert_coco_to_yolo(val_split: float = 0.15, seed: int = 42):
    """Convert COCO annotations to YOLO format."""
    print(f"Loading annotations from {ANNOTATIONS_FILE}")
    with open(ANNOTATIONS_FILE) as f:
        coco = json.load(f)

    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    img_lookup = {img["id"]: img for img in coco["images"]}

    # Group annotations by image
    img_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    # Create directories
    for split in ["train", "val"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Split images
    image_ids = list(img_lookup.keys())
    random.seed(seed)
    random.shuffle(image_ids)
    val_count = max(1, int(len(image_ids) * val_split))
    val_ids = set(image_ids[:val_count])

    # Category ID remapping (YOLO needs 0-indexed contiguous IDs)
    all_cat_ids = sorted(cat_map.keys())
    cat_remap = {old_id: new_id for new_id, old_id in enumerate(all_cat_ids)}

    stats = {"train": 0, "val": 0, "train_anns": 0, "val_anns": 0}

    for img_id in image_ids:
        img_info = img_lookup[img_id]
        split = "val" if img_id in val_ids else "train"

        src_path = IMAGES_DIR / img_info["file_name"]
        if not src_path.exists():
            continue

        # Copy image
        dst_img = OUTPUT_DIR / "images" / split / img_info["file_name"]
        shutil.copy2(src_path, dst_img)

        # Write YOLO label
        iw, ih = img_info["width"], img_info["height"]
        lines = []
        for ann in img_anns.get(img_id, []):
            x, y, w, h = ann["bbox"]
            cx = max(0, min(1, (x + w / 2) / iw))
            cy = max(0, min(1, (y + h / 2) / ih))
            nw = max(0, min(1, w / iw))
            nh = max(0, min(1, h / ih))
            class_id = cat_remap[ann["category_id"]]
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        label_name = Path(img_info["file_name"]).stem + ".txt"
        with open(OUTPUT_DIR / "labels" / split / label_name, "w") as f:
            f.write("\n".join(lines))

        stats[split] += 1
        stats[f"{split}_anns"] += len(lines)

    # Write data.yaml
    names = {cat_remap[old_id]: name for old_id, name in cat_map.items()}
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {OUTPUT_DIR.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(names)}\n")
        f.write("names:\n")
        for idx in sorted(names.keys()):
            # Escape quotes in product names
            f.write(f"  {idx}: \"{names[idx]}\"\n")

    # Save remapping
    with open(OUTPUT_DIR / "category_remap.json", "w") as f:
        json.dump({str(k): v for k, v in cat_remap.items()}, f, indent=2)

    print(f"\nDataset created at {OUTPUT_DIR}")
    print(f"  Train: {stats['train']} images, {stats['train_anns']} annotations")
    print(f"  Val:   {stats['val']} images, {stats['val_anns']} annotations")
    print(f"  Classes: {len(names)}")
    print(f"  data.yaml: {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    convert_coco_to_yolo(args.val_split, args.seed)
