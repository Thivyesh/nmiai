"""
NorgesGruppen Object Detection Dataset Analysis
Analyzes COCO-format annotations for the grocery store product detection task.
"""

import json
import os
import math
from collections import Counter, defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
ANNOTATIONS_PATH = os.path.join(os.path.dirname(__file__),
    "data/coco_dataset/train/annotations.json")
PRODUCT_IMAGES_DIR = os.path.join(os.path.dirname(__file__),
    "data/product_images/")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "analysis_plots/")

os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
with open(ANNOTATIONS_PATH) as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]
categories = data["categories"]

cat_id_to_name = {c["id"]: c["name"] for c in categories}
cat_id_to_supercat = {c["id"]: c.get("supercategory", "unknown") for c in categories}

print("=" * 80)
print("NORGESGRUPPEN OBJECT DETECTION DATASET ANALYSIS")
print("=" * 80)

# ── 1. Summary Stats ──────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("1. SUMMARY STATISTICS")
print("─" * 60)
print(f"  Total images:       {len(images)}")
print(f"  Total annotations:  {len(annotations)}")
print(f"  Total categories:   {len(categories)}")
print(f"  Supercategories:    {set(cat_id_to_supercat.values())}")

# ── 2. Category Distribution ──────────────────────────────────────────────────
print("\n" + "─" * 60)
print("2. CATEGORY DISTRIBUTION")
print("─" * 60)

cat_counts = Counter(a["category_id"] for a in annotations)

# Sort by count descending
sorted_cats = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)

print(f"\n  Top 20 most annotated categories:")
for cat_id, count in sorted_cats[:20]:
    print(f"    {count:5d}  {cat_id_to_name[cat_id]}")

print(f"\n  Bottom 20 least annotated categories:")
for cat_id, count in sorted_cats[-20:]:
    print(f"    {count:5d}  {cat_id_to_name[cat_id]}")

# Threshold analysis
counts_list = list(cat_counts.values())
cats_1 = [cat_id for cat_id, c in cat_counts.items() if c == 1]
cats_lt5 = [cat_id for cat_id, c in cat_counts.items() if c < 5]
cats_lt10 = [cat_id for cat_id, c in cat_counts.items() if c < 10]
cats_lt20 = [cat_id for cat_id, c in cat_counts.items() if c < 20]

print(f"\n  Categories with exactly 1 annotation:  {len(cats_1)}")
print(f"  Categories with < 5 annotations:       {len(cats_lt5)}")
print(f"  Categories with < 10 annotations:      {len(cats_lt10)}")
print(f"  Categories with < 20 annotations:      {len(cats_lt20)}")
print(f"  Categories with >= 20 annotations:     {len(categories) - len(cats_lt20)}")

print(f"\n  Mean annotations per category:   {np.mean(counts_list):.1f}")
print(f"  Median annotations per category: {np.median(counts_list):.1f}")
print(f"  Std annotations per category:    {np.std(counts_list):.1f}")
print(f"  Min: {min(counts_list)}, Max: {max(counts_list)}")

if cats_1:
    print(f"\n  Categories with only 1 annotation:")
    for cat_id in cats_1:
        print(f"    - {cat_id_to_name[cat_id]}")

# Plot: Category distribution histogram
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].hist(counts_list, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_xlabel("Number of annotations")
axes[0].set_ylabel("Number of categories")
axes[0].set_title("Category Annotation Count Distribution")
axes[0].axvline(np.median(counts_list), color='red', linestyle='--', label=f'Median={np.median(counts_list):.0f}')
axes[0].axvline(np.mean(counts_list), color='orange', linestyle='--', label=f'Mean={np.mean(counts_list):.0f}')
axes[0].legend()

# Log scale version
axes[1].hist(counts_list, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[1].set_xlabel("Number of annotations")
axes[1].set_ylabel("Number of categories (log)")
axes[1].set_title("Category Annotation Count Distribution (log scale)")
axes[1].set_yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "category_distribution.png"), dpi=150)
plt.close()

# Plot: Top 30 categories bar chart
fig, ax = plt.subplots(figsize=(14, 10))
top_n = 30
top_cats = sorted_cats[:top_n]
names = [cat_id_to_name[cid][:40] for cid, _ in top_cats]
vals = [c for _, c in top_cats]
ax.barh(range(top_n), vals, color='steelblue', edgecolor='black', alpha=0.8)
ax.set_yticks(range(top_n))
ax.set_yticklabels(names, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Number of annotations")
ax.set_title(f"Top {top_n} Most Annotated Categories")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "top_categories.png"), dpi=150)
plt.close()

# ── 3. Bounding Box Size Analysis ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("3. BOUNDING BOX SIZE ANALYSIS")
print("─" * 60)

areas = []
widths = []
heights = []
aspect_ratios = []

for a in annotations:
    x, y, w, h = a["bbox"]
    area = w * h
    areas.append(area)
    widths.append(w)
    heights.append(h)
    if h > 0:
        aspect_ratios.append(w / h)

areas = np.array(areas)
aspect_ratios = np.array(aspect_ratios)

# COCO standard size definitions (based on area in pixels)
small = areas[areas < 32**2]
medium = areas[(areas >= 32**2) & (areas < 96**2)]
large = areas[areas >= 96**2]

print(f"  Total bounding boxes: {len(areas)}")
print(f"\n  Area statistics:")
print(f"    Min area:    {areas.min():.0f} px^2")
print(f"    Max area:    {areas.max():.0f} px^2")
print(f"    Mean area:   {areas.mean():.0f} px^2")
print(f"    Median area: {np.median(areas):.0f} px^2")
print(f"\n  Size distribution (COCO thresholds):")
print(f"    Small  (<32x32={32**2}px^2):   {len(small):5d} ({100*len(small)/len(areas):.1f}%)")
print(f"    Medium (32x32 to 96x96):       {len(medium):5d} ({100*len(medium)/len(areas):.1f}%)")
print(f"    Large  (>96x96={96**2}px^2):   {len(large):5d} ({100*len(large)/len(areas):.1f}%)")
print(f"\n  Aspect ratio (width/height):")
print(f"    Mean:   {aspect_ratios.mean():.2f}")
print(f"    Median: {np.median(aspect_ratios):.2f}")
print(f"    Min:    {aspect_ratios.min():.2f}")
print(f"    Max:    {aspect_ratios.max():.2f}")

# Width/height stats
print(f"\n  Width statistics:")
print(f"    Min: {min(widths):.0f}, Max: {max(widths):.0f}, Mean: {np.mean(widths):.0f}, Median: {np.median(widths):.0f}")
print(f"  Height statistics:")
print(f"    Min: {min(heights):.0f}, Max: {max(heights):.0f}, Mean: {np.mean(heights):.0f}, Median: {np.median(heights):.0f}")

# Plot: BBox area distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].hist(areas, bins=100, edgecolor='black', alpha=0.7, color='coral')
axes[0].set_xlabel("Area (px^2)")
axes[0].set_ylabel("Count")
axes[0].set_title("Bounding Box Area Distribution")
axes[0].axvline(32**2, color='green', linestyle='--', label='Small/Med boundary')
axes[0].axvline(96**2, color='blue', linestyle='--', label='Med/Large boundary')
axes[0].legend(fontsize=8)

axes[1].hist(aspect_ratios, bins=80, edgecolor='black', alpha=0.7, color='seagreen')
axes[1].set_xlabel("Aspect Ratio (w/h)")
axes[1].set_ylabel("Count")
axes[1].set_title("Aspect Ratio Distribution")

# Scatter of width vs height
axes[2].scatter(widths, heights, alpha=0.05, s=5, color='purple')
axes[2].set_xlabel("Width (px)")
axes[2].set_ylabel("Height (px)")
axes[2].set_title("BBox Width vs Height")
axes[2].set_aspect('equal')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "bbox_analysis.png"), dpi=150)
plt.close()

# Plot: Size category pie chart
fig, ax = plt.subplots(figsize=(8, 8))
sizes_counts = [len(small), len(medium), len(large)]
labels = [f'Small\n({len(small)})', f'Medium\n({len(medium)})', f'Large\n({len(large)})']
colors = ['#ff9999', '#66b3ff', '#99ff99']
ax.pie(sizes_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90,
       textprops={'fontsize': 12})
ax.set_title("BBox Size Distribution (COCO thresholds)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "bbox_sizes_pie.png"), dpi=150)
plt.close()

# ── 4. Per-Image Statistics ────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("4. PER-IMAGE STATISTICS")
print("─" * 60)

img_ann_counts = Counter(a["image_id"] for a in annotations)
all_img_ids = {img["id"] for img in images}
# Images with 0 annotations
imgs_no_ann = all_img_ids - set(img_ann_counts.keys())

per_img = list(img_ann_counts.values())

print(f"  Images with annotations:    {len(img_ann_counts)}")
print(f"  Images without annotations: {len(imgs_no_ann)}")
print(f"\n  Annotations per image:")
print(f"    Min:    {min(per_img)}")
print(f"    Max:    {max(per_img)}")
print(f"    Mean:   {np.mean(per_img):.1f}")
print(f"    Median: {np.median(per_img):.1f}")
print(f"    Std:    {np.std(per_img):.1f}")

# Percentiles
for p in [10, 25, 75, 90, 95]:
    print(f"    P{p}: {np.percentile(per_img, p):.0f}")

# Extreme images
sorted_img = sorted(img_ann_counts.items(), key=lambda x: x[1], reverse=True)
img_id_to_name = {img["id"]: img["file_name"] for img in images}

print(f"\n  Top 10 images with most annotations:")
for img_id, count in sorted_img[:10]:
    print(f"    {count:4d}  {img_id_to_name.get(img_id, f'id={img_id}')}")

print(f"\n  Bottom 10 images with fewest annotations:")
for img_id, count in sorted_img[-10:]:
    print(f"    {count:4d}  {img_id_to_name.get(img_id, f'id={img_id}')}")

# Images with very few annotations
few_thresh = 10
imgs_few = [(iid, c) for iid, c in img_ann_counts.items() if c < few_thresh]
print(f"\n  Images with < {few_thresh} annotations: {len(imgs_few)}")

# Plot: annotations per image
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(range(len(per_img)), sorted(per_img, reverse=True), color='steelblue', alpha=0.8)
ax.set_xlabel("Image index (sorted by count)")
ax.set_ylabel("Number of annotations")
ax.set_title("Annotations per Image (sorted descending)")
ax.axhline(np.mean(per_img), color='red', linestyle='--', label=f'Mean={np.mean(per_img):.0f}')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "annotations_per_image.png"), dpi=150)
plt.close()

# ── 5. Image Resolution Analysis ──────────────────────────────────────────────
print("\n" + "─" * 60)
print("5. IMAGE RESOLUTION ANALYSIS")
print("─" * 60)

resolutions = Counter((img["width"], img["height"]) for img in images)
print(f"  Unique resolutions: {len(resolutions)}")
for res, count in resolutions.most_common(10):
    print(f"    {res[0]}x{res[1]}: {count} images")

# ── 6. Store Section Analysis ─────────────────────────────────────────────────
print("\n" + "─" * 60)
print("6. STORE SECTION / IMAGE GROUPING ANALYSIS")
print("─" * 60)

# Try to infer sections from image file names or from category names
file_names = [img["file_name"] for img in images]
print(f"  Image file name pattern: {file_names[0]} ... {file_names[-1]}")

# Analyze category names for section clues
category_names = [c["name"] for c in categories]
# Common grocery keywords
section_keywords = {
    "Dairy/Eggs": ["melk", "ost", "egg", "yoghurt", "smør", "fløte", "rømme", "cream", "cheese", "milk", "butter"],
    "Bread/Bakery": ["brød", "knekkebrød", "flatbrød", "rundstykke", "toast", "bread", "baguette", "lompe"],
    "Cereal/Breakfast": ["frokost", "müsli", "havre", "corn", "cereal", "granola", "ris"],
    "Drinks/Coffee": ["kaffe", "coffee", "te ", "espresso", "cappuccino", "cola", "juice", "brus", "vann", "drikke", "saft"],
    "Snacks/Chips": ["chips", "popcorn", "snacks", "nøtt", "peanøtt", "cashew"],
    "Chocolate/Candy": ["sjokolade", "chocolate", "godteri", "drops", "lakris", "candy", "karamell"],
    "Pasta/Rice": ["pasta", "spaghetti", "penne", "ris ", "nudel", "noodle", "lasagne"],
    "Canned/Preserved": ["hermetikk", "boks", "tomat", "bønne", "mais"],
    "Condiments/Sauces": ["ketchup", "sennep", "majones", "dressing", "saus", "sauce", "eddik", "olje"],
    "Spices": ["krydder", "pepper", "salt", "kanel", "karri", "basilikum"],
}

section_counts = defaultdict(list)
unclassified = []
for c in categories:
    name_lower = c["name"].lower()
    found = False
    for section, keywords in section_keywords.items():
        if any(kw in name_lower for kw in keywords):
            section_counts[section].append(c["name"])
            found = True
            break
    if not found:
        unclassified.append(c["name"])

print(f"\n  Inferred product sections (from category names):")
for section, prods in sorted(section_counts.items(), key=lambda x: len(x[1]), reverse=True):
    ann_count = sum(cat_counts.get(
        next(c["id"] for c in categories if c["name"] == p), 0) for p in prods)
    print(f"    {section:25s}: {len(prods):3d} categories, {ann_count:5d} annotations")

print(f"    {'Unclassified':25s}: {len(unclassified):3d} categories")

# Plot section distribution
fig, ax = plt.subplots(figsize=(12, 6))
sec_labels = []
sec_cat_counts = []
sec_ann_counts = []
for section, prods in sorted(section_counts.items(), key=lambda x: len(x[1]), reverse=True):
    sec_labels.append(section)
    sec_cat_counts.append(len(prods))
    ann_count = sum(cat_counts.get(
        next(c["id"] for c in categories if c["name"] == p), 0) for p in prods)
    sec_ann_counts.append(ann_count)
sec_labels.append("Unclassified")
sec_cat_counts.append(len(unclassified))
ann_uncl = sum(cat_counts.get(
    next(c["id"] for c in categories if c["name"] == p), 0) for p in unclassified)
sec_ann_counts.append(ann_uncl)

x = np.arange(len(sec_labels))
width = 0.35
bars1 = ax.bar(x - width/2, sec_cat_counts, width, label='Categories', color='steelblue')
ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, sec_ann_counts, width, label='Annotations', color='coral')
ax.set_xticks(x)
ax.set_xticklabels(sec_labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel("Number of Categories")
ax2.set_ylabel("Number of Annotations")
ax.set_title("Product Sections: Categories vs Annotations")
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "store_sections.png"), dpi=150)
plt.close()

# ── 7. "Corrected" Field Analysis ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("7. 'CORRECTED' FIELD ANALYSIS")
print("─" * 60)

has_corrected = any("corrected" in a for a in annotations)
if has_corrected:
    corrected_vals = Counter(a.get("corrected") for a in annotations)
    print(f"  'corrected' field distribution:")
    for val, count in corrected_vals.most_common():
        print(f"    {val}: {count}")
else:
    print("  The 'corrected' field is NOT present in annotations.")
    print("  Available annotation fields: " + ", ".join(annotations[0].keys()))

# Check for corrected field in images
has_corrected_img = any("corrected" in img for img in images)
if has_corrected_img:
    corrected_img = Counter(img.get("corrected") for img in images)
    print(f"\n  'corrected' field in images:")
    for val, count in corrected_img.most_common():
        print(f"    {val}: {count}")
else:
    print("  The 'corrected' field is also NOT present in images.")

# ── 8. Product Images Cross-Reference ─────────────────────────────────────────
print("\n" + "─" * 60)
print("8. PRODUCT IMAGES CROSS-REFERENCE")
print("─" * 60)

product_image_dirs = set(os.listdir(PRODUCT_IMAGES_DIR))
# Filter out hidden files
product_image_dirs = {d for d in product_image_dirs if not d.startswith('.')}

print(f"  Total product image directories: {len(product_image_dirs)}")
print(f"  Total annotation categories:     {len(categories)}")

# Count how many reference images per product dir
ref_img_counts = {}
for d in product_image_dirs:
    full_path = os.path.join(PRODUCT_IMAGES_DIR, d)
    if os.path.isdir(full_path):
        imgs = [f for f in os.listdir(full_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        ref_img_counts[d] = len(imgs)

print(f"\n  Reference images per product directory:")
ref_counts = list(ref_img_counts.values())
print(f"    Min: {min(ref_counts)}, Max: {max(ref_counts)}, Mean: {np.mean(ref_counts):.1f}")
ref_count_dist = Counter(ref_counts)
for n_imgs, n_dirs in sorted(ref_count_dist.items()):
    print(f"    {n_imgs} images: {n_dirs} products")

# Since categories use product names and product_images use barcodes/EANs,
# there's no direct string match. But we can compare counts.
custom_dirs = [d for d in product_image_dirs if d.startswith('CUSTOM')]
ean_dirs = [d for d in product_image_dirs if not d.startswith('CUSTOM')]

print(f"\n  Product image directory types:")
print(f"    EAN/barcode directories: {len(ean_dirs)}")
print(f"    CUSTOM directories:      {len(custom_dirs)}")
print(f"    CUSTOM dirs: {sorted(custom_dirs)}")

# Category count vs product image count comparison
diff = len(categories) - len(product_image_dirs)
print(f"\n  Category vs product image directory comparison:")
print(f"    Categories:                {len(categories)}")
print(f"    Product image directories: {len(product_image_dirs)}")
print(f"    Difference:                {diff}")
if diff > 0:
    print(f"    => {diff} categories likely have NO reference product images")
elif diff < 0:
    print(f"    => {abs(diff)} product image dirs have no corresponding category")
else:
    print(f"    => Exact 1:1 match in count (though mapping may differ)")

# Assuming 1:1 ordered mapping (category id -> product_image_dir),
# check if that's plausible
print(f"\n  NOTE: Categories use product names (e.g., '{categories[0]['name']}')")
print(f"        Product images use barcodes (e.g., '{list(product_image_dirs)[0]}')")
print(f"        Direct name matching is not possible without a mapping file.")
print(f"        If the 345 product dirs map 1:1 to the first 345 categories,")
print(f"        then {len(categories) - len(product_image_dirs)} categories lack reference images.")

# ── 9. Additional Insights ─────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("9. ADDITIONAL INSIGHTS & DATASET CHALLENGES")
print("─" * 60)

# Class imbalance ratio
max_count = max(counts_list)
min_count = min(counts_list)
print(f"  Class imbalance ratio (max/min): {max_count/min_count:.1f}x")

# Categories that appear in very few images
cat_to_images = defaultdict(set)
for a in annotations:
    cat_to_images[a["category_id"]].add(a["image_id"])

cat_img_counts = {cid: len(imgs) for cid, imgs in cat_to_images.items()}
cats_1img = [cid for cid, n in cat_img_counts.items() if n == 1]
cats_2img = [cid for cid, n in cat_img_counts.items() if n <= 2]
print(f"\n  Categories appearing in only 1 image:  {len(cats_1img)}")
print(f"  Categories appearing in <= 2 images:   {len(cats_2img)}")

# Unique category count per image
img_unique_cats = defaultdict(set)
for a in annotations:
    img_unique_cats[a["image_id"]].add(a["category_id"])
unique_per_img = [len(cats) for cats in img_unique_cats.values()]
print(f"\n  Unique categories per image:")
print(f"    Min: {min(unique_per_img)}, Max: {max(unique_per_img)}")
print(f"    Mean: {np.mean(unique_per_img):.1f}, Median: {np.median(unique_per_img):.1f}")

# Duplicate annotations (same category, same image)
img_cat_pairs = Counter((a["image_id"], a["category_id"]) for a in annotations)
duplicates = {k: v for k, v in img_cat_pairs.items() if v > 1}
print(f"\n  Image-category pairs with multiple annotations: {len(duplicates)}")
if duplicates:
    max_dup = max(duplicates.values())
    print(f"  Max annotations for same product in same image: {max_dup}")
    # This indicates multiple instances of the same product on the shelf
    dup_counts = Counter(duplicates.values())
    for n, cnt in sorted(dup_counts.items()):
        print(f"    {n} annotations: {cnt} image-category pairs")

# iscrowd analysis
iscrowd_counts = Counter(a.get("iscrowd", 0) for a in annotations)
print(f"\n  iscrowd distribution:")
for val, count in iscrowd_counts.most_common():
    print(f"    iscrowd={val}: {count}")

# Plot: category frequency vs num images
fig, ax = plt.subplots(figsize=(10, 6))
cat_anns = []
cat_imgs = []
for cid in cat_counts:
    cat_anns.append(cat_counts[cid])
    cat_imgs.append(cat_img_counts[cid])
ax.scatter(cat_imgs, cat_anns, alpha=0.5, s=20, color='steelblue')
ax.set_xlabel("Number of images category appears in")
ax.set_ylabel("Total annotations for category")
ax.set_title("Category: Number of Images vs Total Annotations")
# Add diagonal reference
max_val = max(max(cat_anns), max(cat_imgs))
ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.3, label='1:1 line')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "category_images_vs_annotations.png"), dpi=150)
plt.close()

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print(f"Plots saved to: {PLOTS_DIR}")
print("=" * 80)
