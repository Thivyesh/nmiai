"""
Analyze what distinguishes products WITHIN each super-category.
Focuses on the hard cases: what signals differentiate similar products?

For each super-category, examines:
- Visual similarity of crops (pixel-level stats)
- Bbox size variation within group
- Co-occurrence patterns (which products appear together)
- How many are truly confusable vs distinct

Usage:
    uv run python task1_object_detection/experiments/analyze_intraclass.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

TASK_ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_FILE = TASK_ROOT / "data" / "coco_dataset" / "train" / "annotations.json"
IMAGES_DIR = TASK_ROOT / "data" / "coco_dataset" / "train" / "images"
CLASSIFIER_DIR = TASK_ROOT / "output" / "classifier_dataset" / "train"
OUTPUT_DIR = TASK_ROOT / "output" / "analysis"

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


def get_super_category(name):
    name_lower = name.lower()
    for group, keywords in SUPER_CATEGORIES.items():
        if any(kw in name_lower for kw in keywords):
            return group
    return "other"


def compute_color_histogram(img_path, bins=16):
    """Compute normalized color histogram for an image crop."""
    try:
        img = Image.open(img_path).convert("RGB").resize((64, 64))
        arr = np.array(img)
        hists = []
        for c in range(3):
            h, _ = np.histogram(arr[:, :, c], bins=bins, range=(0, 256))
            h = h / h.sum()
            hists.append(h)
        return np.concatenate(hists)
    except Exception:
        return None


def analyze():
    print("Loading annotations...")
    with open(ANNOTATIONS_FILE) as f:
        coco = json.load(f)

    cat_map = {c["id"]: c["name"] for c in coco["categories"]}

    # Group categories by super-category
    super_groups = defaultdict(list)
    for cat_id, name in cat_map.items():
        group = get_super_category(name)
        super_groups[group].append({"id": cat_id, "name": name})

    # Count annotations per category
    cat_counts = Counter(ann["category_id"] for ann in coco["annotations"])

    # Co-occurrence: which categories appear in the same image?
    img_cats = defaultdict(set)
    for ann in coco["annotations"]:
        img_cats[ann["image_id"]].add(ann["category_id"])

    print(f"\n{'='*70}")
    print("INTRA-CLASS ANALYSIS BY SUPER-CATEGORY")
    print(f"{'='*70}")

    report = {}

    for group_name in list(SUPER_CATEGORIES.keys()) + ["other"]:
        cats = super_groups[group_name]
        if not cats:
            continue

        print(f"\n{'='*70}")
        print(f"GROUP: {group_name.upper()} ({len(cats)} products)")
        print(f"{'='*70}")

        # Sort by annotation count
        cats_sorted = sorted(cats, key=lambda c: cat_counts.get(c["id"], 0), reverse=True)

        # Annotation distribution within group
        ann_counts = [cat_counts.get(c["id"], 0) for c in cats_sorted]
        total_ann = sum(ann_counts)

        print(f"\n  Annotations: {total_ann} total")
        print(f"  Per-product: min={min(ann_counts)}, max={max(ann_counts)}, "
              f"mean={np.mean(ann_counts):.1f}, median={np.median(ann_counts):.0f}")

        # Show products in this group
        print(f"\n  Products (sorted by frequency):")
        for c in cats_sorted[:10]:
            count = cat_counts.get(c["id"], 0)
            print(f"    [{c['id']:3d}] {count:4d} ann | {c['name'][:55]}")
        if len(cats_sorted) > 10:
            print(f"    ... and {len(cats_sorted) - 10} more")

        # Co-occurrence analysis: how often do products from this group
        # appear together in the same image?
        cooccur = Counter()
        for img_id, img_cat_set in img_cats.items():
            group_cats_in_img = [c["id"] for c in cats if c["id"] in img_cat_set]
            if len(group_cats_in_img) > 1:
                for i, c1 in enumerate(group_cats_in_img):
                    for c2 in group_cats_in_img[i+1:]:
                        pair = tuple(sorted([c1, c2]))
                        cooccur[pair] += 1

        if cooccur:
            print(f"\n  Co-occurrence (products appearing in same image):")
            print(f"    {len(cooccur)} unique pairs co-occur")
            top_pairs = cooccur.most_common(5)
            for (c1, c2), count in top_pairs:
                n1 = cat_map.get(c1, str(c1))[:30]
                n2 = cat_map.get(c2, str(c2))[:30]
                print(f"    {count:3d}x: {n1} <-> {n2}")

        # Color analysis: sample crops and compare color distributions
        print(f"\n  Color analysis (sampling crops):")
        cat_colors = {}
        for c in cats_sorted[:20]:  # top 20 by frequency
            crop_dir = CLASSIFIER_DIR / str(c["id"])
            if not crop_dir.exists():
                continue
            crops = list(crop_dir.glob("*.jpg"))[:10]  # sample 10 crops
            hists = []
            for crop_path in crops:
                h = compute_color_histogram(crop_path)
                if h is not None:
                    hists.append(h)
            if hists:
                cat_colors[c["id"]] = {
                    "mean_hist": np.mean(hists, axis=0),
                    "name": c["name"],
                }

        # Find most similar pairs by color
        if len(cat_colors) >= 2:
            color_similarities = []
            cat_ids = list(cat_colors.keys())
            for i, c1 in enumerate(cat_ids):
                for c2 in cat_ids[i+1:]:
                    h1 = cat_colors[c1]["mean_hist"]
                    h2 = cat_colors[c2]["mean_hist"]
                    # Histogram intersection (higher = more similar)
                    sim = np.minimum(h1, h2).sum()
                    color_similarities.append((c1, c2, sim))

            color_similarities.sort(key=lambda x: -x[2])

            print(f"    Most visually similar (by color):")
            for c1, c2, sim in color_similarities[:5]:
                n1 = cat_colors[c1]["name"][:30]
                n2 = cat_colors[c2]["name"][:30]
                print(f"      sim={sim:.3f}: {n1} <-> {n2}")

            print(f"    Most visually distinct:")
            for c1, c2, sim in color_similarities[-3:]:
                n1 = cat_colors[c1]["name"][:30]
                n2 = cat_colors[c2]["name"][:30]
                print(f"      sim={sim:.3f}: {n1} <-> {n2}")

        # What distinguishes products in this group?
        # Analyze text patterns in product names
        print(f"\n  Distinguishing name features:")
        brands = Counter()
        sizes = Counter()
        for c in cats:
            name = c["name"]
            # Extract brand (last word often)
            parts = name.split()
            if len(parts) >= 2:
                brands[parts[-1]] += 1
            # Extract size (pattern like 300G, 250ML)
            import re
            size_match = re.findall(r'\d+(?:G|ML|KG|STK|POS|PK)', name)
            for s in size_match:
                sizes[s] += 1

        if brands:
            print(f"    Brands: {', '.join(f'{b}({c})' for b, c in brands.most_common(5))}")
        if sizes:
            print(f"    Sizes: {', '.join(f'{s}({c})' for s, c in sizes.most_common(5))}")

        # Store group report
        report[group_name] = {
            "num_products": len(cats),
            "total_annotations": total_ann,
            "annotation_distribution": {
                "min": int(min(ann_counts)),
                "max": int(max(ann_counts)),
                "mean": round(float(np.mean(ann_counts)), 1),
            },
            "co_occurrence_pairs": len(cooccur),
            "color_similar_pairs": len([s for _, _, s in color_similarities if s > 0.8]) if len(cat_colors) >= 2 else 0,
        }

    # Save report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "intraclass_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {OUTPUT_DIR / 'intraclass_report.json'}")


if __name__ == "__main__":
    analyze()
