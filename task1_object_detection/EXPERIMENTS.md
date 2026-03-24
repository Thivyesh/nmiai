# Object Detection Experiments Tracker

## Competition Result
- **Public leaderboard: 0.8944** (9-class YOLOv8m + EfficientNet-b1 distilled classifiers)
- Local eval of submitted model: 0.7676

## Winner Analysis
- 1st place (0.9111): YOLOv8m+l WBF ensemble + EVA02-Small classifier, conf=0.001, score blending
- 2nd place (0.9093): D-FINE-X (non-YOLO) end-to-end, HPT-optimized
- 3rd place: YOLOv8x+l+m WBF ensemble at multi-resolution + CLIP fallback
- 6th place (0.9220): 3-model WBF (YOLOv8x + YOLO26-x + fold model), conf=0.001

## Key Learnings from Winners
1. WBF ensemble of 2-3 models is the single biggest improvement
2. conf=0.001 (ultra-low) maximizes recall → better mAP
3. Score blending: `det * (0.5 + 0.5 * cls)` preserves good detections
4. Train on ALL data (no val split) for final model
5. Architecture diversity in ensemble matters more than individual model quality
6. Two-stage (detect → classify) was used by 1st and 3rd place

## Bug Fix: Super-Category Mapping
- `"ost "` keyword matched `"frokost"` → 12 wrong products in spread group
- Fixed: spread now has 15 real products (was 23 contaminated)
- All v2 models use the fixed mapping

---

## Completed Experiments

### E1: Baseline (356-class YOLOv8s, 30 epochs)
- **Score: 0.038 mAP@0.5**
- 80% of classes had zero AP
- Finding: annotation count does NOT predict AP (correlation: -0.020)

### E2: 1-class detector (YOLOv8m, 50 epochs)
- **mAP@0.5: 0.934** (detection only)
- Detection ceiling: 0.7 × 0.934 = 0.654

### E3: 9-class superclass detector (YOLOv8m, 50 epochs)
- **mAP@0.5: 0.90** (per-class eval)
- Routes detections to group classifiers

### E4: Classification approaches comparison
| Approach | Top-1 Accuracy |
|---|---|
| YOLO-cls (343 classes) | <1% (overfit) |
| YOLO-cls per group | <9% (overfit) |
| YOLO backbone + LogReg (global) | 33% |
| YOLO backbone + LogReg (per group) | 47.3% |
| YOLO backbone + OCR + LogReg (per group) | 55-67% |
| CLIP zero-shot | 69.5% (confusable subset only) |
| CLIP + OCR | 75.2% (confusable subset only) |

### E5: Knowledge distillation (MobileNet-small)
- **94.7% avg accuracy** across 9 groups
- Soft labels from backbone + OCR + LogReg teacher
- Breakthrough: same CNN that got <9% with hard labels gets 94.7% with soft labels

### E6: Classifier model comparison
| Model | Accuracy | Size (9 models) |
|---|---|---|
| MobileNet-small | 92.4% | 54MB |
| MobileNet-large | 93.3% | 189MB |
| **EfficientNet-b1** | **93.9%** | 270MB |

### E7: Score blending + conf tuning
- Old: `det * cls`, conf=0.15 → local 0.7676
- New: `det * (0.5 + 0.5 * cls)`, conf=0.001 → local 0.7768
- Improvement: +0.009

### E8: WBF ensemble (YOLOv8m + YOLOv8x)
- conf=0.001, score blending
- **Local score: 0.7768** (vs 0.7676 single model)
- 2-model ensemble improves detection

### E9: Detection soft labels (refined detector)
- Ensemble of 3 detectors → WBF → refined box coordinates
- 1,475 pseudo-labels added (classifier-verified)
- 22,286 GT boxes refined (70% GT + 30% detector average)
- **Refined single model: 0.7924 local score** (best so far!)
- Beats the 2-model ensemble (0.7768)

### E10: Embedding teacher (embeddinggemma)
- OCR text → embed with embeddinggemma → cosine sim to product names
- Combined [backbone 576d + embedding sims 356d] → LogReg/MLP teacher
- Results vs old backbone-only teacher:

| Group | Old Teacher | New Teacher | Improvement |
|---|---|---|---|
| knekkebroed | 42.1% | **64.2%** | **+22.1pp** |
| cereal | 40.9% | **60.4%** | **+19.5pp** |
| other | 43.4% | **63.1%** | **+19.7pp** |
| eggs | 59.2% | **69.4%** | **+10.2pp** |
| chocolate | 77.9% | **87.5%** | **+9.6pp** |
| tea | 52.6% | **60.3%** | **+7.7pp** |
| coffee | 51.5% | **57.1%** | **+5.6pp** |
| spread | 63.5% | **64.9%** | **+1.4pp** |
| cookies | 48.6% | 48.6% | 0 |

### E11: 3-stage pipeline (detect → group → classify)
- 1-class detector → group classifier (96.3%) → per-group EfficientNet
- Local score: 0.7675 (same as 2-stage, group routing isn't the bottleneck)

### E12: YOLOv8x on A100 (1280px, 50 epochs)
- Trained on A100 GPUs
- Better model but custom ONNX postprocessing lost quality
- Leaderboard: 0.8881 (worse than 0.8944 due to ONNX issues)
- Using .pt through ultralytics: 0.7676 local (same pipeline quality)

### E13: V2 detector (fixed super-category mapping)
- YOLOv8m, 50 epochs, 640px, fixed spread group (18→15 real products)
- **mAP@0.5: 0.88** (vs v1: 0.90 — slightly lower, different class distribution)
- Clean mapping removes cross-contamination

### E14: Embedding teacher (embeddinggemma, fixed mapping)
- OCR text → embeddinggemma (768d) → cosine sim to product names
- Combined [backbone 576d + embedding sims 356d] = 932 features → LogReg
- Per-group feature selection: bbox features only for cookies (+8.1pp)
- **Results vs old backbone-only teacher:**

| Group | Old Teacher | Embedding Teacher | Improvement |
|---|---|---|---|
| knekkebroed | 42.1% | **64.2%** | +22.1pp |
| cereal | 41.5% | **60.9%** | +19.5pp |
| other | 43.8% | **61.6%** | +17.8pp |
| spread | 22.5% | **40.0%** | +17.5pp |
| eggs | 59.2% | **69.4%** | +10.2pp |
| chocolate | 79.9% | **87.8%** | +7.9pp |
| tea | 52.6% | **60.3%** | +7.7pp |
| coffee | 54.5% | **60.0%** | +5.5pp |
| cookies | 48.6% | 48.6% | 0 |

- **MLP vs LogReg comparison**: LogReg wins every group. MLP overfits with 932 features and <2k samples per group.
- Cookies: 0% improvement because problem is data scarcity (149 samples), not feature quality
- Coffee: low improvement because OCR reads brand but not variant text (FILTERMALT vs KOKMALT too small)

### E15: Refined v2 detector (soft labels + pseudo-labels, clean mapping)
- Ensemble of v2 + 1-class detectors → WBF → refined boxes + 1,581 pseudo-labels
- 22,175 GT boxes refined (70% GT + 30% detector average)
- Status: **TRAINING**

### E16: Distill from embedding teacher v2
- EfficientNet-b1 distilled from embedding teacher soft labels
- **Results:**

| Group | V1 (old teacher) | V3 (embedding teacher) |
|---|---|---|
| knekkebroed | 92.9% | **93.7%** (+0.8pp) |
| coffee | 87.7% | **89.9%** (+2.2pp) |
| other | 91.5% | **93.7%** (+2.2pp) |
| cereal | 93.8% | 93.7% (-0.1pp) |
| eggs | 99.1% | 97.7% (-1.4pp) |
| spread | 95.9% | 90.9% (-5.0pp) |
| cookies | 100% | 100% (same) |
| chocolate | 99.3% | 99.3% (same) |
| tea | 92.2% | 91.6% (-0.6pp) |
| **Average** | **94.7%** | **94.5%** (-0.2pp) |

- Marginal change — the better teacher doesn't transfer well through distillation because the student only sees images, not the OCR/embedding features the teacher used.

### E17: V2 pipeline eval (v2 detector + v3 classifiers + blend + conf=0.001)
- **Local score: 0.7759** (vs v1 original: 0.7676, vs refined v1: 0.7924)
- Better than original but worse than refined v1 — v2 detector not as strong

### E18: Fine-tuned CLIP teacher
- Fine-tuned openai/clip-vit-base-patch32 on crop-product name pairs
- Contrastive learning: 14.8M / 151.3M params trainable (10%)
- Froze most of vision encoder, fine-tuned last 2 layers + projections
- **Results:**

| Epoch | Loss | Val Accuracy |
|---|---|---|
| 0 (zero-shot) | — | 24.1% |
| 1 | 0.911 | 43.8% |
| 5 | 0.198 | 60.9% |
| 10 | 0.130 | **67.0%** |

- **Best teacher so far at 67.0%** (vs embedding teacher 63%, backbone LogReg 47%)
- Generated soft labels for all 9468 train + 2977 val crops
- Saved fine-tuned model to output/clip_teacher/best_model/

---

## Running Experiments

### E19: Distill from fine-tuned CLIP teacher
- EfficientNet-b1 per group, using CLIP soft labels
- Status: **TRAINING**
- Expected: better classification than E16 due to stronger teacher (67% vs 63%)

### E15: Refined v2 detector
- Status: **TRAINING**
- Expected: better than refined v1 (0.7924) due to clean mapping

---

## Planned Experiments

### E20: Full pipeline v3
- Refined v2 detector + CLIP-distilled classifiers
- WBF ensemble + score blending + conf=0.001
- End-to-end eval to measure total improvement

### E21: Combined CLIP + embedding teacher
- Merge soft labels from CLIP and embeddinggemma teachers
- Each captures different signals (vision vs OCR text)
- May improve over CLIP alone

### E22: Experiment B — CLIP teacher for detection labels
- Use CLIP teacher predictions as category labels in detection training
- Better category labels → better 9-class detection

### E23: EVA02-Tiny/Small classifiers
- Replace EfficientNet-b1 with EVA02 from timm (winner's choice)
- Better accuracy/size ratio

### E24: YOLOv8l refined on A100
- Train YOLOv8l on soft label dataset at 1280px
- Ensemble refined_m + refined_l with WBF

### E25: Train on all 248 images (no val split)
- Winners all trained on full data for final submission
- Need separate evaluation strategy

### E26: CLIP fine-tuning improvements
- More epochs, larger batch size, different learning rate
- Try CLIP ViT-L for even better teacher
- Fine-tune text encoder too (currently frozen)
