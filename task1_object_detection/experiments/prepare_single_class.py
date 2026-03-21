"""
Convert COCO dataset to YOLO format with ALL categories merged into 1 class ("product").
For training a detection-only model (70% of competition score).

Usage:
    uv run python task1_object_detection/experiments/prepare_single_class.py
"""

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parent.parent
COCO_DIR = TASK_ROOT / "data" / "coco_dataset" / "train"
ANNOTATIONS_FILE = COCO_DIR / "annotations.json"
IMAGES_DIR = COCO_DIR / "images"
OUTPUT_DIR = TASK_ROOT / "output" / "yolo_dataset_1class"

SEED = 42
VAL_SPLIT = 0.15


def convert():
    print(f"Loading annotations from {ANNOTATIONS_FILE}")
    with open(ANNOTATIONS_FILE) as f:
        coco = json.load(f)

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
    random.seed(SEED)
    random.shuffle(image_ids)
    val_count = max(1, int(len(image_ids) * VAL_SPLIT))
    val_ids = set(image_ids[:val_count])

    stats = {"train": 0, "val": 0, "train_anns": 0, "val_anns": 0}

    for img_id in image_ids:
        img_info = img_lookup[img_id]
        split = "val" if img_id in val_ids else "train"

        src_path = IMAGES_DIR / img_info["file_name"]
        if not src_path.exists():
            continue

        shutil.copy2(src_path, OUTPUT_DIR / "images" / split / img_info["file_name"])

        iw, ih = img_info["width"], img_info["height"]
        lines = []
        for ann in img_anns.get(img_id, []):
            x, y, w, h = ann["bbox"]
            cx = max(0, min(1, (x + w / 2) / iw))
            cy = max(0, min(1, (y + h / 2) / ih))
            nw = max(0, min(1, w / iw))
            nh = max(0, min(1, h / ih))
            # All class 0 ("product")
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        label_name = Path(img_info["file_name"]).stem + ".txt"
        with open(OUTPUT_DIR / "labels" / split / label_name, "w") as f:
            f.write("\n".join(lines))

        stats[split] += 1
        stats[f"{split}_anns"] += len(lines)

    # Write data.yaml
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {OUTPUT_DIR.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 1\n")
        f.write("names:\n")
        f.write('  0: "product"\n')

    print(f"\n1-class dataset created at {OUTPUT_DIR}")
    print(f"  Train: {stats['train']} images, {stats['train_anns']} annotations")
    print(f"  Val:   {stats['val']} images, {stats['val_anns']} annotations")
    print(f"  Classes: 1 (product)")
    print(f"  data.yaml: {yaml_path}")


if __name__ == "__main__":
    convert()
