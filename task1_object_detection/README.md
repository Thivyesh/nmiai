# Task 1: NorgesGruppen Object Detection

Grocery shelf product detection and classification for the NM i AI competition.

## Pipeline

```
Shelf Image
  → 9-class YOLOv8 detector (ONNX) → product boxes + super-category
  → Crop each product
  → Route to group's EfficientNet classifier (.npy weights)
    - 9 groups: knekkebroed, coffee, tea, cereal, eggs, spread, cookies, chocolate, other
    - Trained with knowledge distillation from YOLO backbone + OCR soft labels
  → Output: [image_id, category_id, bbox, score]
```

## Results

| Component | Metric | Value |
|---|---|---|
| Detection (9-class) | mAP@0.5 | 0.90 |
| Detection (1-class) | mAP@0.5 | 0.934 |
| Classification (per-group) | Top-1 accuracy | 93.9% |
| **Competition score** | **Public leaderboard** | **0.8944** |

## Project Structure

```
task1_object_detection/
├── submission/          # Final submission files
│   ├── run.py          # Entry point (ONNX detector + npy classifiers)
│   ├── build_submission.sh
│   └── test_sandbox/   # Docker test matching sandbox env
├── experiments/         # All experiment scripts
│   ├── prepare_dataset.py        # COCO → YOLO conversion
│   ├── prepare_superclass.py     # 9-class super-category dataset
│   ├── prepare_single_class.py   # 1-class detection dataset
│   ├── train_baseline.py         # YOLOv8 training with MLflow
│   ├── train_detector.py         # 1-class detector
│   ├── train_detector_gpu.py     # GPU training script
│   ├── train_distilled.py        # Knowledge distillation classifiers
│   ├── train_distilled_v3.py     # V3 with multi-crop + EfficientNet-b2
│   ├── build_soft_labels.py      # Generate soft labels from backbone+OCR+LogReg
│   ├── build_classifier_dataset.py # Crop products for classification
│   ├── diagnose_weakness.py      # Analyze detection/classification failures
│   ├── analyze_intraclass.py     # Within-group product similarity
│   ├── test_clip.py              # CLIP zero-shot classification test
│   ├── test_ocr.py               # EasyOCR text reading test
│   ├── test_trocr.py             # TrOCR text reading test
│   ├── test_yolo_features.py     # YOLO backbone feature extraction
│   ├── evaluate.py               # Per-class mAP evaluation
│   └── log_analysis.py           # Log dataset stats to MLflow
├── agent/               # LangGraph-based automation agent
│   ├── agent.py         # Analyzer → Booster → Trainer pipeline
│   ├── tools.py         # 14 LangChain tools
│   ├── config.py        # Paths and defaults
│   └── main.py          # CLI entrypoint
├── FINDINGS.md          # Ongoing analysis and results log
├── run.py               # Original baseline submission
└── data/                # Training data (gitignored)
    ├── coco_dataset/    # 248 shelf images + COCO annotations
    └── product_images/  # Reference product photos
```

## Key Findings

1. **Annotation count does NOT predict AP** — correlation: -0.020. The worst classes have 300+ annotations but 0 AP due to visual similarity between same-brand products.

2. **Two-stage pipeline is essential** — 356-class YOLO gets 3.8% mAP. Separating detection (9 groups) + classification (per-group) gets 89.4%.

3. **Knowledge distillation is the breakthrough** — YOLO-cls overfits at <9%. Distilling from backbone+OCR+LogReg soft labels into MobileNet/EfficientNet achieves 93.9%.

4. **OCR helps most on the hardest groups** — +10pp on knekkebroed (WASA variants), +5pp on coffee (EVERGOOD variants).

5. **CLIP zero-shot gets 70% on confusable products** but YOLO backbone+LogReg beats it on the full dataset (33% vs 24%).

## Reproducing

```bash
# Setup
uv sync

# Prepare datasets
uv run python experiments/prepare_superclass.py    # 9-class detector data
uv run python experiments/build_classifier_dataset.py  # Classification crops

# Train detector
uv run python experiments/train_detector_gpu.py    # On GPU

# Build soft labels (requires OCR)
uv run python experiments/build_soft_labels.py

# Train classifiers with distillation
uv run python experiments/train_distilled.py

# Build submission
bash submission/build_submission.sh

# Test in Docker
bash submission/test_sandbox/run_test.sh
```

## Sandbox Compatibility

- Detector: ONNX format (onnxruntime-gpu in sandbox)
- Classifiers: numpy .npy weights (no pickle needed)
- All imports sandbox-safe (no os, pickle, subprocess)
- Docker-tested with exact sandbox package versions
