"""
Diagnose what causes poor performance by correlating per-class AP
with dataset characteristics.

Identifies signals that differentiate well-detected vs missed products.

Usage:
    uv run python task1_object_detection/experiments/diagnose_weakness.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

TASK_ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_FILE = TASK_ROOT / "data" / "coco_dataset" / "train" / "annotations.json"
PRODUCT_IMAGES_DIR = TASK_ROOT / "data" / "product_images"
OUTPUT_DIR = TASK_ROOT / "output" / "analysis"

# Find most recent per-class metrics
MODELS_DIR = TASK_ROOT / "output" / "models"


def load_per_class_metrics():
    """Find and load per-class AP from the most recent training run."""
    candidates = list(MODELS_DIR.rglob("per_class_metrics.json"))
    if not candidates:
        print("No per_class_metrics.json found. Run training + evaluation first.")
        return None
    path = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"Loading per-class metrics from: {path}")
    with open(path) as f:
        return json.load(f)


def analyze():
    print("=" * 70)
    print("WEAKNESS DIAGNOSIS")
    print("=" * 70)

    # Load annotations
    with open(ANNOTATIONS_FILE) as f:
        coco = json.load(f)

    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    img_lookup = {img["id"]: img for img in coco["images"]}

    # Build per-category stats from annotations
    cat_stats = {}
    for cat in coco["categories"]:
        cat_stats[cat["id"]] = {
            "name": cat["name"],
            "cat_id": cat["id"],
            "annotation_count": 0,
            "image_count": 0,
            "images": set(),
            "bbox_areas": [],
            "bbox_widths": [],
            "bbox_heights": [],
            "aspect_ratios": [],
            "corrected_count": 0,
            "positions_x": [],  # relative x position on image
            "positions_y": [],  # relative y position on image
            "has_product_code": 0,
        }

    for ann in coco["annotations"]:
        cid = ann["category_id"]
        if cid not in cat_stats:
            continue
        s = cat_stats[cid]
        s["annotation_count"] += 1
        s["images"].add(ann["image_id"])

        x, y, w, h = ann["bbox"]
        area = w * h
        s["bbox_areas"].append(area)
        s["bbox_widths"].append(w)
        s["bbox_heights"].append(h)
        if h > 0:
            s["aspect_ratios"].append(w / h)

        # Relative position in image
        img = img_lookup.get(ann["image_id"])
        if img:
            s["positions_x"].append((x + w / 2) / img["width"])
            s["positions_y"].append((y + h / 2) / img["height"])

        if ann.get("corrected"):
            s["corrected_count"] += 1
        if ann.get("product_code"):
            s["has_product_code"] += 1

    for s in cat_stats.values():
        s["image_count"] = len(s["images"])
        s["images"] = list(s["images"])

    # Load per-class AP metrics
    per_class = load_per_class_metrics()
    if not per_class:
        return

    # Build AP lookup (class_id from YOLO remapping -> AP)
    # The per_class metrics use remapped IDs, we need to match by name
    ap_by_name = {c["name"]: c["ap50"] for c in per_class}

    # Merge AP into cat_stats
    for cid, s in cat_stats.items():
        s["ap50"] = ap_by_name.get(s["name"], None)

    # Filter to categories that were evaluated (have AP)
    evaluated = [s for s in cat_stats.values() if s["ap50"] is not None]
    zero_ap = [s for s in evaluated if s["ap50"] == 0]
    low_ap = [s for s in evaluated if 0 < s["ap50"] < 0.1]
    mid_ap = [s for s in evaluated if 0.1 <= s["ap50"] < 0.3]
    good_ap = [s for s in evaluated if s["ap50"] >= 0.3]

    print(f"\nTotal categories evaluated: {len(evaluated)}")
    print(f"  Zero AP (0.0):    {len(zero_ap)} ({len(zero_ap)/len(evaluated)*100:.0f}%)")
    print(f"  Low AP (0-0.1):   {len(low_ap)} ({len(low_ap)/len(evaluated)*100:.0f}%)")
    print(f"  Mid AP (0.1-0.3): {len(mid_ap)} ({len(mid_ap)/len(evaluated)*100:.0f}%)")
    print(f"  Good AP (>=0.3):  {len(good_ap)} ({len(good_ap)/len(evaluated)*100:.0f}%)")

    # =====================================================================
    # SIGNAL 1: Annotation count vs AP
    # =====================================================================
    print(f"\n{'='*70}")
    print("SIGNAL 1: Annotation Count vs Performance")
    print(f"{'='*70}")

    for group_name, group in [("Zero AP", zero_ap), ("Low AP", low_ap),
                               ("Mid AP", mid_ap), ("Good AP", good_ap)]:
        counts = [s["annotation_count"] for s in group]
        if counts:
            print(f"\n  {group_name} ({len(group)} classes):")
            print(f"    Annotation count: min={min(counts)}, max={max(counts)}, "
                  f"mean={np.mean(counts):.1f}, median={np.median(counts):.0f}")

    # Correlation
    aps = [s["ap50"] for s in evaluated]
    ann_counts = [s["annotation_count"] for s in evaluated]
    if len(set(aps)) > 1:
        corr = np.corrcoef(ann_counts, aps)[0, 1]
        print(f"\n  Correlation (annotation_count, AP): {corr:.3f}")

    # Threshold analysis
    for threshold in [3, 5, 10, 20, 50]:
        below = [s for s in evaluated if s["annotation_count"] < threshold]
        above = [s for s in evaluated if s["annotation_count"] >= threshold]
        if below and above:
            avg_below = np.mean([s["ap50"] for s in below])
            avg_above = np.mean([s["ap50"] for s in above])
            print(f"  <{threshold} annotations: avg AP={avg_below:.4f} ({len(below)} classes) | "
                  f">={threshold}: avg AP={avg_above:.4f} ({len(above)} classes)")

    # =====================================================================
    # SIGNAL 2: Bbox size vs AP
    # =====================================================================
    print(f"\n{'='*70}")
    print("SIGNAL 2: Bounding Box Size vs Performance")
    print(f"{'='*70}")

    for group_name, group in [("Zero AP", zero_ap), ("Low AP", low_ap),
                               ("Mid AP", mid_ap), ("Good AP", good_ap)]:
        areas = []
        for s in group:
            areas.extend(s["bbox_areas"])
        if areas:
            print(f"\n  {group_name}:")
            print(f"    Bbox area: min={min(areas):.0f}, max={max(areas):.0f}, "
                  f"mean={np.mean(areas):.0f}, median={np.median(areas):.0f}")

    # Per-class mean area vs AP
    mean_areas = [np.mean(s["bbox_areas"]) if s["bbox_areas"] else 0 for s in evaluated]
    if len(set(aps)) > 1 and any(a > 0 for a in mean_areas):
        corr = np.corrcoef(mean_areas, aps)[0, 1]
        print(f"\n  Correlation (mean_bbox_area, AP): {corr:.3f}")

    # Small vs large
    small = [s for s in evaluated if s["bbox_areas"] and np.mean(s["bbox_areas"]) < 10000]
    medium = [s for s in evaluated if s["bbox_areas"] and 10000 <= np.mean(s["bbox_areas"]) < 50000]
    large = [s for s in evaluated if s["bbox_areas"] and np.mean(s["bbox_areas"]) >= 50000]
    for name, grp in [("Small (<10k px²)", small), ("Medium (10k-50k)", medium), ("Large (>50k)", large)]:
        if grp:
            avg_ap = np.mean([s["ap50"] for s in grp])
            print(f"  {name}: avg AP={avg_ap:.4f} ({len(grp)} classes)")

    # =====================================================================
    # SIGNAL 3: Aspect ratio vs AP
    # =====================================================================
    print(f"\n{'='*70}")
    print("SIGNAL 3: Aspect Ratio vs Performance")
    print(f"{'='*70}")

    for group_name, group in [("Zero AP", zero_ap), ("Good AP", good_ap)]:
        ratios = []
        for s in group:
            ratios.extend(s["aspect_ratios"])
        if ratios:
            print(f"\n  {group_name}:")
            print(f"    Aspect ratio (w/h): mean={np.mean(ratios):.2f}, "
                  f"median={np.median(ratios):.2f}, std={np.std(ratios):.2f}")

    # =====================================================================
    # SIGNAL 4: Image count (how many unique images contain this category)
    # =====================================================================
    print(f"\n{'='*70}")
    print("SIGNAL 4: Image Diversity vs Performance")
    print(f"{'='*70}")

    for group_name, group in [("Zero AP", zero_ap), ("Low AP", low_ap),
                               ("Mid AP", mid_ap), ("Good AP", good_ap)]:
        img_counts = [s["image_count"] for s in group]
        if img_counts:
            print(f"\n  {group_name}:")
            print(f"    Unique images: min={min(img_counts)}, max={max(img_counts)}, "
                  f"mean={np.mean(img_counts):.1f}, median={np.median(img_counts):.0f}")

    img_counts_all = [s["image_count"] for s in evaluated]
    if len(set(aps)) > 1:
        corr = np.corrcoef(img_counts_all, aps)[0, 1]
        print(f"\n  Correlation (image_count, AP): {corr:.3f}")

    # =====================================================================
    # SIGNAL 5: Position on shelf vs AP
    # =====================================================================
    print(f"\n{'='*70}")
    print("SIGNAL 5: Position on Shelf vs Performance")
    print(f"{'='*70}")

    for group_name, group in [("Zero AP", zero_ap), ("Good AP", good_ap)]:
        xs, ys = [], []
        for s in group:
            xs.extend(s["positions_x"])
            ys.extend(s["positions_y"])
        if xs:
            print(f"\n  {group_name}:")
            print(f"    X position (0=left, 1=right): mean={np.mean(xs):.2f}, std={np.std(xs):.2f}")
            print(f"    Y position (0=top, 1=bottom): mean={np.mean(ys):.2f}, std={np.std(ys):.2f}")

    # =====================================================================
    # SIGNAL 6: Annotation quality (corrected field)
    # =====================================================================
    print(f"\n{'='*70}")
    print("SIGNAL 6: Annotation Quality (corrected field)")
    print(f"{'='*70}")

    for group_name, group in [("Zero AP", zero_ap), ("Good AP", good_ap)]:
        total_ann = sum(s["annotation_count"] for s in group)
        total_corrected = sum(s["corrected_count"] for s in group)
        if total_ann > 0:
            print(f"  {group_name}: {total_corrected}/{total_ann} corrected "
                  f"({total_corrected/total_ann*100:.1f}%)")

    # =====================================================================
    # SIGNAL 7: Product code availability
    # =====================================================================
    print(f"\n{'='*70}")
    print("SIGNAL 7: Product Code (barcode) Availability")
    print(f"{'='*70}")

    for group_name, group in [("Zero AP", zero_ap), ("Good AP", good_ap)]:
        with_code = sum(1 for s in group if s["has_product_code"] > 0)
        print(f"  {group_name}: {with_code}/{len(group)} have product codes "
              f"({with_code/len(group)*100:.1f}%)")

    # =====================================================================
    # SIGNAL 8: Instances per image (density)
    # =====================================================================
    print(f"\n{'='*70}")
    print("SIGNAL 8: Instances per Image (how many of same product per image)")
    print(f"{'='*70}")

    for group_name, group in [("Zero AP", zero_ap), ("Good AP", good_ap)]:
        densities = []
        for s in group:
            if s["image_count"] > 0:
                densities.append(s["annotation_count"] / s["image_count"])
        if densities:
            print(f"\n  {group_name}:")
            print(f"    Instances/image: mean={np.mean(densities):.1f}, "
                  f"median={np.median(densities):.1f}")

    # =====================================================================
    # SIGNAL 9: Name similarity / product type clustering
    # =====================================================================
    print(f"\n{'='*70}")
    print("SIGNAL 9: Product Type Breakdown")
    print(f"{'='*70}")

    type_keywords = {
        "coffee/tea": ["kaffe", "coffee", "espresso", "cappuccino", "te ", "tea", "nescafe",
                       "evergood", "friele", "ali ", "pukka", "dolce gusto", "kapsel"],
        "bread/crackers": ["brød", "bread", "knekkebrød", "flatbrød", "lefse", "grissini",
                           "kjeks", "kneipp", "wasa", "sigdal"],
        "cereal/breakfast": ["frokost", "havre", "müsli", "granola", "corn flakes",
                             "cheerios", "fras", "puffet"],
        "eggs": ["egg"],
        "spreads": ["smør", "bremykt", "brelett", "nugatti", "syltetøy", "ost"],
    }

    for group_name, group in [("Zero AP", zero_ap), ("Good AP", good_ap)]:
        type_counts = Counter()
        for s in group:
            name_lower = s["name"].lower()
            matched = False
            for type_name, keywords in type_keywords.items():
                if any(kw in name_lower for kw in keywords):
                    type_counts[type_name] += 1
                    matched = True
                    break
            if not matched:
                type_counts["other"] += 1
        print(f"\n  {group_name}:")
        for t, c in type_counts.most_common():
            print(f"    {t}: {c} ({c/len(group)*100:.0f}%)")

    # =====================================================================
    # TOP PERFORMERS - what makes them work?
    # =====================================================================
    print(f"\n{'='*70}")
    print("TOP 15 PERFORMERS (what works)")
    print(f"{'='*70}")

    sorted_by_ap = sorted(evaluated, key=lambda s: s["ap50"], reverse=True)
    for s in sorted_by_ap[:15]:
        mean_area = np.mean(s["bbox_areas"]) if s["bbox_areas"] else 0
        print(f"  AP={s['ap50']:.3f} | ann={s['annotation_count']:3d} | "
              f"imgs={s['image_count']:2d} | area={mean_area:8.0f} | {s['name'][:50]}")

    # =====================================================================
    # WORST WITH ENOUGH DATA (should be doing better)
    # =====================================================================
    print(f"\n{'='*70}")
    print("WORST CLASSES WITH >=20 ANNOTATIONS (should be doing better)")
    print(f"{'='*70}")

    enough_data_zero = [s for s in zero_ap if s["annotation_count"] >= 20]
    enough_data_zero.sort(key=lambda s: s["annotation_count"], reverse=True)
    for s in enough_data_zero[:15]:
        mean_area = np.mean(s["bbox_areas"]) if s["bbox_areas"] else 0
        print(f"  AP={s['ap50']:.3f} | ann={s['annotation_count']:3d} | "
              f"imgs={s['image_count']:2d} | area={mean_area:8.0f} | {s['name'][:50]}")

    # =====================================================================
    # SAVE FULL REPORT
    # =====================================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = []
    for s in sorted_by_ap:
        report.append({
            "name": s["name"],
            "cat_id": s["cat_id"],
            "ap50": s["ap50"],
            "annotation_count": s["annotation_count"],
            "image_count": s["image_count"],
            "mean_bbox_area": round(np.mean(s["bbox_areas"]), 1) if s["bbox_areas"] else 0,
            "mean_aspect_ratio": round(np.mean(s["aspect_ratios"]), 3) if s["aspect_ratios"] else 0,
            "corrected_pct": round(s["corrected_count"] / s["annotation_count"] * 100, 1) if s["annotation_count"] > 0 else 0,
            "has_product_code": s["has_product_code"] > 0,
            "instances_per_image": round(s["annotation_count"] / max(s["image_count"], 1), 1),
        })

    report_path = OUTPUT_DIR / "weakness_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    analyze()
