#!/bin/bash
# Test submission in a Docker container matching the sandbox environment.
# Usage: bash task1_object_detection/submission/test_sandbox/run_test.sh
set -e

TASK_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TEST_DIR="$TASK_ROOT/submission/test_sandbox"
SUBMISSION_ZIP="$TASK_ROOT/output/submission.zip"

if [ ! -f "$SUBMISSION_ZIP" ]; then
    echo "ERROR: submission.zip not found. Run build_submission.sh first."
    exit 1
fi

# Copy submission zip to test dir
cp "$SUBMISSION_ZIP" "$TEST_DIR/submission.zip"

# Copy a few test images
rm -rf "$TEST_DIR/test_images"
mkdir -p "$TEST_DIR/test_images"
ls "$TASK_ROOT/data/coco_dataset/train/images/"*.jpg | head -5 | while read f; do
    cp "$f" "$TEST_DIR/test_images/"
done

echo "=== Building sandbox Docker image ==="
docker build -t nmiai-sandbox-test "$TEST_DIR"

echo ""
echo "=== Running submission in sandbox ==="
mkdir -p "$TEST_DIR/output"
docker run --rm \
    -v "$TEST_DIR/output:/output" \
    --memory=8g \
    --cpus=4 \
    nmiai-sandbox-test

echo ""
echo "=== Results ==="
if [ -f "$TEST_DIR/output/predictions.json" ]; then
    echo "predictions.json created!"
    python3 -c "
import json
with open('$TEST_DIR/output/predictions.json') as f:
    preds = json.load(f)
print(f'  Predictions: {len(preds)}')
if preds:
    cats = set(p['category_id'] for p in preds)
    imgs = set(p['image_id'] for p in preds)
    print(f'  Unique images: {len(imgs)}')
    print(f'  Unique categories: {len(cats)}')
    print(f'  Score range: {min(p[\"score\"] for p in preds):.3f} - {max(p[\"score\"] for p in preds):.3f}')
    print(f'  Sample: {json.dumps(preds[0], indent=2)}')
"
else
    echo "ERROR: predictions.json not created!"
fi

# Cleanup
rm -f "$TEST_DIR/submission.zip"
rm -rf "$TEST_DIR/test_images"
