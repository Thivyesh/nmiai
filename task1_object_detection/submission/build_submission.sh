#!/bin/bash
# Build submission zip. Only uses .pt, .json, .py files.
set -e

TASK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$TASK_ROOT/output/submission_build"
OUTPUT_ZIP="$TASK_ROOT/output/submission.zip"

echo "=== Building submission ==="
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

cp "$TASK_ROOT/submission/run.py" "$BUILD_DIR/run.py"

# Detector
cp "$TASK_ROOT/output/models/superclass_yolov8m_e50/weights/best.pt" "$BUILD_DIR/detector.pt"

# Merged classifiers (.pt)
cp "$BUILD_DIR/classifiers.pt" "$BUILD_DIR/classifiers.pt" 2>/dev/null || \
cd "$TASK_ROOT/.." && uv run python -c "
import torch, json
from pathlib import Path
_orig = torch.load
torch.load = lambda *a, **kw: _orig(*a, **{**kw, 'weights_only': False})
MODEL_DIR = Path('task1_object_detection/output/models/distilled_v2_effb1')
merged, classes = {}, {}
for gd in sorted(MODEL_DIR.iterdir()):
    if not gd.is_dir(): continue
    mp, cp = gd/'best.pt', gd/'classes.json'
    if not mp.exists(): continue
    merged[gd.name] = torch.load(str(mp), map_location='cpu')
    with open(cp) as f: classes[gd.name] = json.load(f)
torch.save({'models': merged, 'classes': classes}, '$BUILD_DIR/classifiers.pt')
print(f'classifiers.pt: {Path(\"$BUILD_DIR/classifiers.pt\").stat().st_size/1024/1024:.1f}MB')
"

# Category mapping
cd "$TASK_ROOT/.."
uv run python -c "
import json
with open('task1_object_detection/data/coco_dataset/train/annotations.json') as f:
    coco = json.load(f)
cat_map = {c['id']:c['name'] for c in coco['categories']}
SUPERS = {
    'knekkebroed':['knekkebrød','knekke','flatbrød','wasa','sigdal','leksands','ryvita','korni'],
    'coffee':['kaffe','coffee','espresso','nescafe','evergood','friele','ali ','dolce gusto','cappuccino','kapsel'],
    'tea':[' te ','tea','twinings','lipton','pukka','urtete'],
    'cereal':['frokost','havre','müsli','granola','corn flakes','cheerios','cruesli','puffet','fras'],
    'eggs':['egg'],'spread':['smør','bremykt','brelett','ost ','cream cheese'],
    'cookies':['kjeks','cookie','grissini'],'chocolate':['sjokolade','nugatti','regia','cocoa'],
}
c2g,gc={},{}
for cid,name in cat_map.items():
    nl=name.lower(); matched=False
    for g,kws in SUPERS.items():
        if any(kw in nl for kw in kws): c2g[str(cid)]=g; gc.setdefault(g,[]).append(cid); matched=True; break
    if not matched: c2g[str(cid)]='other'; gc.setdefault('other',[]).append(cid)
with open('$BUILD_DIR/category_mapping.json','w') as f:
    json.dump({'cat_id_to_name':{str(k):v for k,v in cat_map.items()},'cat_id_to_group':c2g,'group_classes':{g:sorted(v) for g,v in gc.items()}},f)
print('category_mapping.json saved')
"

# Zip
cd "$BUILD_DIR"
rm -f "$OUTPUT_ZIP"
zip -r "$OUTPUT_ZIP" . -x ".*" "__MACOSX/*"

echo ""
echo "=== Submission ==="
unzip -l "$OUTPUT_ZIP"
echo "Weight files: $(unzip -l "$OUTPUT_ZIP" | grep -cE '\.(pt|pth|onnx|safetensors|npy)$')/3"
echo "Python files: $(unzip -l "$OUTPUT_ZIP" | grep -c '\.py$')/10"
echo "Size: $(du -h "$OUTPUT_ZIP" | awk '{print $1}')"
