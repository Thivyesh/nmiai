#!/bin/bash
# Build submission zip. Uses safetensors for classifiers (no pickle).
# Weight files: detector.pt (1) + classifiers.safetensors (1) = 2/3 limit
set -e

TASK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUBMISSION_DIR="$TASK_ROOT/submission"
BUILD_DIR="$TASK_ROOT/output/submission_build"
OUTPUT_ZIP="$TASK_ROOT/output/submission.zip"

echo "=== Building submission ==="
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

cp "$SUBMISSION_DIR/run.py" "$BUILD_DIR/run.py"

# Detector
DETECTOR="$TASK_ROOT/output/models/superclass_yolov8m_e50/weights/best.pt"
[ ! -f "$DETECTOR" ] && echo "ERROR: detector not found" && exit 1
cp "$DETECTOR" "$BUILD_DIR/detector.pt"

# Classifiers (safetensors + JSON)
cd "$TASK_ROOT/.."
uv run python -c "
import torch, json
from pathlib import Path
from safetensors.torch import save_file

_orig = torch.load
torch.load = lambda *a, **kw: _orig(*a, **{**kw, 'weights_only': False})

MODEL_DIR = Path('task1_object_detection/output/models/distilled_v2_effb1')
tensors, classes = {}, {}
for gd in sorted(MODEL_DIR.iterdir()):
    if not gd.is_dir(): continue
    mp, cp = gd/'best.pt', gd/'classes.json'
    if not mp.exists(): continue
    state = torch.load(str(mp), map_location='cpu')
    with open(cp) as f: classes[gd.name] = json.load(f)
    for k,v in state.items(): tensors[f'{gd.name}##{k}'] = v

save_file(tensors, '$BUILD_DIR/classifiers.safetensors')
with open('$BUILD_DIR/classifier_classes.json','w') as f: json.dump(classes,f)
sz = Path('$BUILD_DIR/classifiers.safetensors').stat().st_size/1024/1024
print(f'Classifiers: {sz:.1f}MB, {len(classes)} groups')
"

# Category mapping
uv run python -c "
import json
from pathlib import Path
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
c2g,gc = {},{}
for cid,name in cat_map.items():
    nl = name.lower()
    matched = False
    for g,kws in SUPERS.items():
        if any(kw in nl for kw in kws): c2g[str(cid)]=g; gc.setdefault(g,[]).append(cid); matched=True; break
    if not matched: c2g[str(cid)]='other'; gc.setdefault('other',[]).append(cid)
with open('$BUILD_DIR/category_mapping.json','w') as f:
    json.dump({'cat_id_to_name':{str(k):v for k,v in cat_map.items()},'cat_id_to_group':c2g,'group_classes':{g:sorted(v) for g,v in gc.items()}},f,indent=2,ensure_ascii=False)
print('Category mapping saved')
"

# Zip
cd "$BUILD_DIR"
rm -f "$OUTPUT_ZIP"
zip -r "$OUTPUT_ZIP" . -x ".*" "__MACOSX/*"

echo ""
echo "=== Submission ==="
echo "  Size: $(du -h "$OUTPUT_ZIP" | awk '{print $1}')"
unzip -l "$OUTPUT_ZIP"
echo "  Weight files: $(unzip -l "$OUTPUT_ZIP" | grep -cE '\.pt|\.safetensors')/3 limit"
echo "  Python files: $(unzip -l "$OUTPUT_ZIP" | grep -c '\.py')/10 limit"
