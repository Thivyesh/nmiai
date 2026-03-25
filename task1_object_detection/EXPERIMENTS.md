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
- **mAP@0.5: 0.970** (vs refined v1: 0.967) — clean mapping slightly better

### E16: Distill from embedding teacher v2
- EfficientNet-b1 distilled from embedding teacher soft labels
- Marginal overall change (-0.2pp avg) — better teacher doesn't transfer fully through distillation

### E17: V2 pipeline eval (v2 detector + v3 classifiers + blend + conf=0.001)
- **Local score: 0.7759** (vs v1 original: 0.7676, vs refined v1: 0.7924)

### E18: Fine-tuned CLIP teachers (3 variants)

| Model | Params | Patches | Best Val Accuracy |
|---|---|---|---|
| CLIP ViT-B/32 | 151M | 49 | 67.0% |
| CLIP ViT-L/14 | 428M | 256 | 67.5% |
| **CLIP ViT-B/16** | **150M** | **196** | **72.9%** |

- B/16 wins — same params as B/32 but 4x finer patches reads text on packaging better
- L/14 barely improves over B/32 — bottleneck is data, not model capacity
- Contrastive learning: freeze most of vision encoder, fine-tune last 2 layers + projections

### E19: Distill from CLIP teachers

| Group | V1 (old) | V4 (CLIP-B/32) | V5 (CLIP-B/16) |
|---|---|---|---|
| coffee | 87.7% | **92.3%** | 89.7% |
| other | 91.5% | 93.7% | 92.3% |
| knekkebroed | 92.9% | 93.6% | 93.3% |
| cereal | 93.8% | 93.7% | **93.9%** |
| spread | **95.9%** | 71.3% | 72.5% |
| cookies | **100%** | 97.3% | 97.3% |
| eggs | **99.1%** | 97.2% | 96.8% |
| chocolate | 99.3% | 99.3% | 99.3% |
| tea | **92.2%** | 89.1% | 90.4% |

- CLIP helps hard groups (coffee +4.6pp) but hurts small groups (spread -24.6pp)
- CLIP soft labels add noise when few training samples

### E20: Per-group best teacher selection
- Pick best classifier per group from all versions
- **Average accuracy: 95.6%** (vs any single version's ~94.5%)

| Group | Best Teacher | Accuracy |
|---|---|---|
| cookies | V1 (old) | 100% |
| chocolate | V1 (old) | 99.3% |
| eggs | V1 (old) | 99.1% |
| cereal | V5 (CLIP-B/16) | 93.9% |
| knekkebroed | V3 (embedding) | 93.7% |
| other | V3 (embedding) | 93.7% |
| coffee | V4 (CLIP-B/32) | 92.3% |
| tea | V1 (old) | 92.2% |
| spread | Clean retrain (label smoothing) | 90.9% |

Pattern: V1 (hard labels) best for small/easy groups; CLIP/embedding for hard groups

### E21: Clean spread retrain
- 15 real spread products (fixed from 23 contaminated)
- Label smoothing (0.1) instead of distillation — more robust for 178 crops
- **Accuracy: 90.9%**

### E22: Full pipeline eval — BEST RESULT
- Refined v2 detector + best-per-group classifiers + score blending + conf=0.001
- **Det=0.798 Cls=0.799 Score=0.7985**
- Best local score, +0.031 over original submission (0.7676)

---

## Results Summary

| # | Config | Det mAP | Cls mAP | Local Score | Leaderboard |
|---|---|---|---|---|---|
| 1 | 356-class YOLOv8s baseline | — | — | 0.038 | — |
| 2 | 9-class det + V1 classifiers (conf=0.15) | 0.760 | 0.757 | 0.7676 | **0.8944** |
| 3 | WBF m+x ensemble + blend + conf=0.001 | 0.780 | 0.769 | 0.7768 | — |
| 4 | Refined v1 (soft labels) + blend + conf=0.001 | 0.792 | 0.793 | 0.7924 | — |
| 5 | V2 det + V3 classifiers + blend | 0.787 | 0.749 | 0.7759 | — |
| **6** | **Refined v2 + best-per-group + blend + conf=0.001** | **0.798** | **0.799** | **0.7985** | — |

### Method Description

**Detection pipeline:**
1. Train YOLOv8m 9-class superclass detector on fixed mapping (9 product groups)
2. Build detection soft labels: run multiple detectors → WBF merge → refine GT box coordinates (70% GT + 30% detector avg) + add pseudo-labels (classifier-verified)
3. Retrain detector on soft labels → "refined" detector with better localization

**Classification pipeline:**
1. Multiple teachers generate soft probability labels per crop:
   - V1: YOLO backbone features (576d) + OCR word-match → LogReg
   - V3: backbone + embeddinggemma OCR text similarity (932d) → LogReg
   - V4/V5: Fine-tuned CLIP (contrastive learning on crop ↔ product name pairs)
2. Per-group best teacher selection: pick whichever teacher gives highest val accuracy per group
3. Knowledge distillation: train EfficientNet-b1 per group on winning teacher's soft labels
4. Small groups (spread, cookies) use hard labels or label smoothing instead

**Inference:**
1. Refined detector (conf=0.001, maximize recall)
2. Crop each detection with 15% padding
3. Route to group's EfficientNet-b1 classifier
4. Score blending: `final_score = det_conf × (0.5 + 0.5 × cls_conf)`

---

## Untested / Planned Experiments

### HIGH PRIORITY — likely to improve score significantly

### E23: Re-extract backbone features with refined v2 detector
- The embedding teacher used backbone features from the ORIGINAL detector
- Refined v2 detector has better features (trained on soft labels)
- Re-extracting → retraining embedding teacher could improve the teacher significantly
- Then distill again → better classifiers → iterate

### E24: Train YOLOv8l on refined soft labels (A100)
- YOLOv8l has 43.7M params vs yolov8m's 25.9M
- Train at 1280px on A100 GPUs
- Ensemble refined_m + refined_l with WBF for even better detection

### E25: WBF ensemble of refined v1 + refined v2 detectors
- Different training data → different errors → WBF should improve
- Both are yolov8m so architecture diversity is limited

### MEDIUM PRIORITY

### E26: CLIP teacher for detection category labels
- Use CLIP predictions as soft category labels in detector training
- Better categories → better 9-class group routing

### E27: Train on all 248 images (no val holdout)
- Winners all used full data for final model
- Need to trust per-group val or cross-validation instead

### E28: EVA02-Tiny/Small classifiers (timm)
- Winner's classifier choice, better than EfficientNet-b1
- Available in sandbox via timm 0.9.12

### E29: Combined CLIP + embedding teacher soft labels
- Merge probability distributions from both teachers
- Different error patterns → ensemble should help

### LOWER PRIORITY

### E30: CLIP fine-tuning improvements
- More epochs, unfreeze text encoder, larger batch
- B/16 + more data augmentation

### E31: Multi-scale detection inference
- Run detector at 640 + 1280, merge with WBF
- Winners used this (1280+flip TTA)

### E32: Classifier TTA
- Horizontal flip on crop at inference, average predictions
- Free +0.5-1% on classification
