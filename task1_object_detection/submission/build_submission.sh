#!/bin/bash
# Build the submission zip with all required files.
# Usage: bash task1_object_detection/submission/build_submission.sh
set -e

TASK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUBMISSION_DIR="$TASK_ROOT/submission"
BUILD_DIR="$TASK_ROOT/output/submission_build"
OUTPUT_ZIP="$TASK_ROOT/output/submission.zip"

echo "=== Building submission ==="

# Clean build directory
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/classifiers"

# Copy run.py
cp "$SUBMISSION_DIR/run.py" "$BUILD_DIR/run.py"

# Copy detector weights
# Use 9-class superclass detector (routes to group classifiers)
DETECTOR_WEIGHTS="$TASK_ROOT/output/models/superclass_yolov8m_e50/weights/best.pt"
if [ ! -f "$DETECTOR_WEIGHTS" ]; then
    echo "ERROR: Detector weights not found at $DETECTOR_WEIGHTS"
    exit 1
fi
cp "$DETECTOR_WEIGHTS" "$BUILD_DIR/detector.pt"

# Copy distilled classifiers
for group_dir in "$TASK_ROOT/output/models/distilled"/*/; do
    group_name=$(basename "$group_dir")
    mkdir -p "$BUILD_DIR/classifiers/$group_name"
    cp "$group_dir/best.pt" "$BUILD_DIR/classifiers/$group_name/best.pt"
    cp "$group_dir/classes.json" "$BUILD_DIR/classifiers/$group_name/classes.json"
done

# Generate category mapping
echo "Generating category mapping..."
cd "$TASK_ROOT/.."
uv run python -c "
import json
from pathlib import Path

TASK_ROOT = Path('task1_object_detection')

with open(TASK_ROOT / 'data/coco_dataset/train/annotations.json') as f:
    coco = json.load(f)

cat_map = {c['id']: c['name'] for c in coco['categories']}

SUPER_CATEGORIES = {
    'knekkebroed': ['knekkebrød', 'knekke', 'flatbrød', 'wasa', 'sigdal', 'leksands', 'ryvita', 'korni'],
    'coffee': ['kaffe', 'coffee', 'espresso', 'nescafe', 'evergood', 'friele', 'ali ', 'dolce gusto', 'cappuccino', 'kapsel'],
    'tea': [' te ', 'tea', 'twinings', 'lipton', 'pukka', 'urtete'],
    'cereal': ['frokost', 'havre', 'müsli', 'granola', 'corn flakes', 'cheerios', 'cruesli', 'puffet', 'fras'],
    'eggs': ['egg'],
    'spread': ['smør', 'bremykt', 'brelett', 'ost ', 'cream cheese'],
    'cookies': ['kjeks', 'cookie', 'grissini'],
    'chocolate': ['sjokolade', 'nugatti', 'regia', 'cocoa'],
}

cat_id_to_group = {}
group_classes = {}
for cid, name in cat_map.items():
    name_lower = name.lower()
    matched = False
    for g, kws in SUPER_CATEGORIES.items():
        if any(kw in name_lower for kw in kws):
            cat_id_to_group[str(cid)] = g
            group_classes.setdefault(g, []).append(cid)
            matched = True
            break
    if not matched:
        cat_id_to_group[str(cid)] = 'other'
        group_classes.setdefault('other', []).append(cid)

for g in group_classes:
    group_classes[g] = sorted(group_classes[g])

mapping = {
    'cat_id_to_name': {str(k): v for k, v in cat_map.items()},
    'cat_id_to_group': cat_id_to_group,
    'group_classes': group_classes,
}

out = Path('$BUILD_DIR') / 'category_mapping.json'
with open(out, 'w') as f:
    json.dump(mapping, f, indent=2, ensure_ascii=False)
print(f'Saved category mapping ({len(cat_map)} categories, {len(group_classes)} groups)')
"

# Create zip
cd "$BUILD_DIR"
rm -f "$OUTPUT_ZIP"
zip -r "$OUTPUT_ZIP" . -x ".*" "__MACOSX/*"

echo ""
echo "=== Submission built ==="
echo "  Zip: $OUTPUT_ZIP"
echo "  Size: $(du -h "$OUTPUT_ZIP" | awk '{print $1}')"
echo ""
echo "  Contents:"
unzip -l "$OUTPUT_ZIP" | head -20
echo ""
echo "  Verify: unzip -l $OUTPUT_ZIP | head -20"
echo "  Upload at the competition submit page."
