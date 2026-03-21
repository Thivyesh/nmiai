"""
Crop annotated products from shelf images to build a classification dataset.
Also includes product reference images.

Creates:
  output/classifier_dataset/
    train/{category_id}/  — cropped products from shelf images
    val/{category_id}/    — held-out crops for validation
    reference/{category_id}/  — product reference photos

Usage:
    uv run python task1_object_detection/experiments/build_classifier_dataset.py
"""

import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

TASK_ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_FILE = TASK_ROOT / "data" / "coco_dataset" / "train" / "annotations.json"
IMAGES_DIR = TASK_ROOT / "data" / "coco_dataset" / "train" / "images"
PRODUCT_IMAGES_DIR = TASK_ROOT / "data" / "product_images"
OUTPUT_DIR = TASK_ROOT / "output" / "classifier_dataset"

SEED = 42
VAL_SPLIT = 0.15
MIN_CROP_SIZE = 20  # skip tiny crops


def build():
    print("Loading annotations...")
    with open(ANNOTATIONS_FILE) as f:
        coco = json.load(f)

    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    img_lookup = {img["id"]: img for img in coco["images"]}

    # Group annotations by image
    img_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    # Split images into train/val
    image_ids = list(img_lookup.keys())
    random.seed(SEED)
    random.shuffle(image_ids)
    val_count = max(1, int(len(image_ids) * VAL_SPLIT))
    val_ids = set(image_ids[:val_count])

    # Create output dirs
    for split in ["train", "val"]:
        (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)

    stats = {"train_crops": 0, "val_crops": 0, "skipped_small": 0, "skipped_error": 0}
    cat_counts = {"train": Counter(), "val": Counter()}

    # Crop products from shelf images
    print("Cropping products from shelf images...")
    for img_id, anns in img_anns.items():
        img_info = img_lookup[img_id]
        img_path = IMAGES_DIR / img_info["file_name"]
        if not img_path.exists():
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  Error loading {img_path}: {e}")
            stats["skipped_error"] += 1
            continue

        split = "val" if img_id in val_ids else "train"

        for ann in anns:
            cat_id = ann["category_id"]
            x, y, w, h = ann["bbox"]

            # Skip tiny crops
            if w < MIN_CROP_SIZE or h < MIN_CROP_SIZE:
                stats["skipped_small"] += 1
                continue

            # Crop with small padding (5%)
            pad_x = w * 0.05
            pad_y = h * 0.05
            x1 = max(0, int(x - pad_x))
            y1 = max(0, int(y - pad_y))
            x2 = min(img.width, int(x + w + pad_x))
            y2 = min(img.height, int(y + h + pad_y))

            crop = img.crop((x1, y1, x2, y2))

            # Save crop
            cat_dir = OUTPUT_DIR / split / str(cat_id)
            cat_dir.mkdir(parents=True, exist_ok=True)

            crop_name = f"img{img_id}_ann{ann['id']}.jpg"
            crop.save(cat_dir / crop_name, quality=90)

            stats[f"{split}_crops"] += 1
            cat_counts[split][cat_id] += 1

    # Copy product reference images
    print("Copying product reference images...")
    ref_dir = OUTPUT_DIR / "reference"

    # Build product_code -> category_id mapping from annotations
    code_to_cat = {}
    for ann in coco["annotations"]:
        if ann.get("product_code"):
            code_to_cat[str(ann["product_code"])] = ann["category_id"]

    ref_count = 0
    if PRODUCT_IMAGES_DIR.exists():
        for product_dir in PRODUCT_IMAGES_DIR.iterdir():
            if not product_dir.is_dir():
                continue
            code = product_dir.name
            cat_id = code_to_cat.get(code)

            if cat_id is not None:
                out_dir = ref_dir / str(cat_id)
                out_dir.mkdir(parents=True, exist_ok=True)
                for img_file in product_dir.glob("*.jpg"):
                    shutil.copy2(img_file, out_dir / f"ref_{code}_{img_file.name}")
                    ref_count += 1

    # Print stats
    print(f"\n{'='*60}")
    print("CLASSIFIER DATASET BUILT")
    print(f"{'='*60}")
    print(f"  Train crops: {stats['train_crops']}")
    print(f"  Val crops:   {stats['val_crops']}")
    print(f"  Skipped (too small): {stats['skipped_small']}")
    print(f"  Reference images: {ref_count}")
    print(f"  Categories in train: {len(cat_counts['train'])}")
    print(f"  Categories in val:   {len(cat_counts['val'])}")

    # Category distribution in train
    train_counts = sorted(cat_counts["train"].values())
    if train_counts:
        print(f"\n  Train crops per category:")
        print(f"    min={min(train_counts)}, max={max(train_counts)}, "
              f"mean={sum(train_counts)/len(train_counts):.1f}, "
              f"median={train_counts[len(train_counts)//2]}")
        print(f"    <3 crops: {sum(1 for c in train_counts if c < 3)} categories")
        print(f"    <5 crops: {sum(1 for c in train_counts if c < 5)} categories")
        print(f"    <10 crops: {sum(1 for c in train_counts if c < 10)} categories")

    # Save metadata
    meta = {
        "stats": stats,
        "categories": {str(k): v for k, v in cat_map.items()},
        "train_category_counts": {str(k): v for k, v in cat_counts["train"].items()},
        "val_category_counts": {str(k): v for k, v in cat_counts["val"].items()},
        "ref_images": ref_count,
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n  Output: {OUTPUT_DIR}")
    print(f"  Metadata: {OUTPUT_DIR / 'metadata.json'}")


if __name__ == "__main__":
    build()
