#!/bin/bash
# Setup Label Studio for reviewing COCO object detection annotations
set -e

DATA_DIR="/Users/thivyeshahilathasan/Documents/GitHub/nmiai/task1_object_detection/data/coco_dataset/train"

echo "=== 1. Creating Label Studio environment ==="
uv venv ~/label-studio-env
source ~/label-studio-env/bin/activate
uv pip install label-studio label-studio-converter

echo "=== 2. Converting COCO annotations to Label Studio format ==="
label-studio-converter import coco \
  -i "$DATA_DIR/annotations.json" \
  -o "$DATA_DIR/ls_annotations.json" \
  --image-root-url "/data/local-files/?d=images/"

echo "=== 3. Starting Label Studio ==="
echo ""
echo "After it opens at http://localhost:8080:"
echo "  1. Create account (local only)"
echo "  2. Create new project"
echo "  3. Settings > Labeling Interface > Code > paste the label config"
echo "  4. Settings > Cloud Storage > Add Source Storage > Local Files"
echo "     Path: $DATA_DIR/images"
echo "  5. Import > upload $DATA_DIR/ls_annotations.json"
echo ""

export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$DATA_DIR"

label-studio start
