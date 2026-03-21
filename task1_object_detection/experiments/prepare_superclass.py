"""
Convert COCO dataset to YOLO format with super-categories (9 groups).
Tests whether grouped detection maintains high mAP while learning product types.

Usage:
    uv run python task1_object_detection/experiments/prepare_superclass.py
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
OUTPUT_DIR = TASK_ROOT / "output" / "yolo_dataset_superclass"

SEED = 42
VAL_SPLIT = 0.15

SUPER_CATEGORIES = {
    "knekkebroed": ["knekkebrød", "knekke", "flatbrød", "wasa", "sigdal", "leksands", "ryvita", "korni"],
    "coffee": ["kaffe", "coffee", "espresso", "nescafe", "evergood", "friele", "ali ", "dolce gusto", "cappuccino", "kapsel"],
    "tea": [" te ", "tea", "twinings", "lipton", "pukka", "urtete"],
    "cereal": ["frokost", "havre", "müsli", "granola", "corn flakes", "cheerios", "cruesli", "puffet", "fras"],
    "eggs": ["egg"],
    "spread": ["smør", "bremykt", "brelett", "ost ", "cream cheese"],
    "cookies": ["kjeks", "cookie", "grissini"],
    "chocolate": ["sjokolade", "nugatti", "regia", "cocoa"],
}

SUPER_NAMES = list(SUPER_CATEGORIES.keys()) + ["other"]
SUPER_NAME_TO_ID = {name: idx for idx, name in enumerate(SUPER_NAMES)}


def get_super_category(product_name: str) -> int:
    name_lower = product_name.lower()
    for group, keywords in SUPER_CATEGORIES.items():
        if any(kw in name_lower for kw in keywords):
            return SUPER_NAME_TO_ID[group]
    return SUPER_NAME_TO_ID["other"]


def convert():
    print(f"Loading annotations from {ANNOTATIONS_FILE}")
    with open(ANNOTATIONS_FILE) as f:
        coco = json.load(f)

    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    img_lookup = {img["id"]: img for img in coco["images"]}

    # Build category_id -> super_category_id mapping
    cat_to_super = {}
    for cat_id, name in cat_map.items():
        cat_to_super[cat_id] = get_super_category(name)

    # Print mapping stats
    from collections import Counter
    super_counts = Counter(cat_to_super.values())
    print(f"\nSuper-category mapping:")
    for name, idx in SUPER_NAME_TO_ID.items():
        print(f"  {idx}: {name} ({super_counts[idx]} original categories)")

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
            super_id = cat_to_super[ann["category_id"]]
            lines.append(f"{super_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

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
        f.write(f"nc: {len(SUPER_NAMES)}\n")
        f.write("names:\n")
        for idx, name in enumerate(SUPER_NAMES):
            f.write(f'  {idx}: "{name}"\n')

    # Save mapping for later use (original cat_id -> super_id)
    mapping = {
        "cat_to_super": {str(k): v for k, v in cat_to_super.items()},
        "super_names": SUPER_NAMES,
        "super_name_to_id": SUPER_NAME_TO_ID,
    }
    with open(OUTPUT_DIR / "super_category_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"\nSuper-class dataset created at {OUTPUT_DIR}")
    print(f"  Train: {stats['train']} images, {stats['train_anns']} annotations")
    print(f"  Val:   {stats['val']} images, {stats['val_anns']} annotations")
    print(f"  Classes: {len(SUPER_NAMES)} ({', '.join(SUPER_NAMES)})")
    print(f"  data.yaml: {yaml_path}")


if __name__ == "__main__":
    convert()
