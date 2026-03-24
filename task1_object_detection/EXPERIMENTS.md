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

---

## Running Experiments

### E13: V2 detector (fixed super-category mapping)
- YOLOv8m, 50 epochs, 640px, fixed spread group
- Status: TRAINING
- Next: build refined v2 with soft labels

### E14: Embedding teacher v2 (fixed mapping)
- OCR + embeddinggemma on all training crops
- Status: TRAINING (OCR phase)
- Next: generate soft labels → distill classifiers

---

## Planned Experiments

### E15: Refined v2 detector
- Build detection soft labels using v2 detector ensemble
- Train refined v2 on soft labels + pseudo-labels
- Expected: better than refined v1 (0.7924) due to clean mapping

### E16: Distill from embedding teacher v2
- Use embedding teacher soft labels to train EfficientNet-b1
- Compare with E5 (old backbone-only soft labels)

### E17: Extract backbone from refined v2
- Re-extract 576d features using refined v2 backbone
- Combine with embedding similarities
- Train MLP teacher → generate even better soft labels

### E18: Full pipeline with all improvements
- Refined v2 detector + embedding-distilled classifiers
- WBF ensemble + score blending + conf=0.001
- Expected: significant improvement over 0.8944

### E19 (future): Experiment B — embedding teacher for detection
- Use embedding teacher predictions as category labels in detection training
- Compare with E15 (standard detection soft labels)

### E20 (future): EVA02-Tiny classifiers
- Replace EfficientNet-b1 with EVA02-Tiny from timm
- Winner's classifier choice, better accuracy/size ratio

### E21 (future): YOLOv8l refined on A100
- Train YOLOv8l on soft label dataset
- Ensemble refined_m + refined_l with WBF

### E22 (future): Train on all 248 images (no val split)
- Winners all trained on full data for final submission
- Need separate evaluation strategy (cross-validation or trust leaderboard)
