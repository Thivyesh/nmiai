# Object Detection Findings

Ongoing analysis of model performance and dataset characteristics.

---

## 2026-03-21: Baseline Diagnosis (YOLOv8s, 30 epochs, 640px)

### Baseline Results

| Metric | Value |
|---|---|
| mAP@0.5 | 0.0375 |
| Precision | 0.187 |
| Recall | 0.057 |
| Est. competition score | 0.038 |
| Zero-AP classes | 198 / 249 (80%) |

### Key Finding: Annotation Count Does NOT Predict AP

Correlation between annotation count and AP: **-0.020** (near zero).

The worst performing classes have hundreds of annotations:
- HAVRE KNEKKEBRØD 300G WASA: **398 annotations, 0.0 AP**
- KNEKKEBRØD 100 FRØ&HAVSALT 245G WASA: **374 annotations, 0.0 AP**
- EVERGOOD CLASSIC FILTERMALT 250G: **368 annotations, 0.0 AP**

Meanwhile some top performers have almost no data:
- COTW LUNGO KAFFEKAPSEL 10STK: **6 annotations, 0.72 AP**
- COTW COLOMBIA EXCELSO KAFFEKAPSEL 10STK: **5 annotations, 0.50 AP**
- HAVREGRYN STORE GLUTENFRI 1KG AXA: **1 annotation, 0.48 AP**

**Conclusion:** The problem is not data quantity — it's inter-class visual similarity.

### Signal Analysis

#### 1. Inter-class confusion is the primary problem
- The worst performers are bread/crackers products (WASA, SIGDAL, LEKSANDS) that look nearly identical on shelves
- Coffee products (EVERGOOD variants) also heavily confused
- Zero AP group breakdown: 31% coffee/tea, 15% bread/crackers, 17% cereal, 10% eggs
- Products with many visually similar siblings fail; visually unique products succeed even with 1 annotation

#### 2. Small bounding boxes are completely missed
- All 15 classes with mean bbox area <10,000 px² have zero AP
- Medium boxes (10k-50k px²) avg AP: 0.047
- Large boxes (>50k px²) avg AP: 0.023 (worse — may indicate annotation quality issues at large sizes)

#### 3. Instances per image correlates with failure
- Zero AP classes: avg 2.6 instances per image
- Good AP classes: avg 1.7 instances per image
- More copies of the same product in one image = more confusion between similar products

#### 4. Aspect ratio consistency matters
- Good AP products have tight, consistent shapes (aspect ratio std = 0.38)
- Zero AP products have variable shapes (std = 0.80)
- More consistent packaging = easier to detect

#### 5. Position and bbox size do NOT matter
- No meaningful difference in shelf position (x, y) between good and bad classes
- Mean bbox area has near-zero correlation with AP (-0.011)

#### 6. Image diversity has minimal effect
- Correlation between unique image count and AP: 0.012
- Zero AP classes appear in 24.7 images on average (plenty of variety)

### Worst Classes With Sufficient Data (should be doing better)

These have >=20 annotations but zero AP — the model is confused, not data-starved:

| Class | Annotations | Images | Mean Area |
|---|---|---|---|
| HAVRE KNEKKEBRØD 300G WASA | 398 | 86 | 30,398 |
| KNEKKEBRØD 100 FRØ&HAVSALT 245G WASA | 374 | 85 | 25,556 |
| EVERGOOD CLASSIC FILTERMALT 250G | 368 | 29 | 23,503 |
| HUSMAN KNEKKEBRØD 260G WASA | 322 | 75 | 29,789 |
| FRUKOST FULLKORN 320G WASA | 300 | 83 | 32,939 |
| LEKSANDS KNEKKE FIBERBIT 240G | 297 | 82 | 29,482 |
| KNEKKEBRØD DIN STUND CHIA&HAVSALT 270G WASA | 265 | 82 | 33,764 |
| LEKSANDS KNEKKE GODT STEKT 200G | 260 | 73 | 23,270 |
| KNEKKEBRØD GODT FOR DEG 235G SIGDAL | 247 | 77 | 19,219 |
| FLATBRØD 275G KORNI | 247 | 80 | 26,407 |

### Top Performers (what works)

| Class | AP | Annotations | Images | Mean Area |
|---|---|---|---|---|
| COTW LUNGO ØKOLOGISK KAFFEKAPSEL 10STK | 0.718 | 6 | 6 | 27,662 |
| FOREST FRUIT TEA 20POS LIPTON | 0.534 | 76 | 35 | 25,455 |
| YELLOW LABEL TEA 50POS LIPTON | 0.531 | 34 | 21 | 41,863 |
| COTW COLOMBIA EXCELSO KAFFEKAPSEL 10STK | 0.497 | 5 | 5 | 16,518 |
| EVERGOOD CLASSIC PRESSMALT 250G | 0.480 | 63 | 28 | 38,329 |
| HAVREKNEKKEBRØD TYNT 265G WASA | 0.320 | 185 | 63 | 23,524 |
| CRUESLI 4 NUTS 450G QUAKER | 0.316 | 139 | 69 | 58,647 |

Common traits: visually distinctive packaging, consistent shape, unique color/branding.

### Recommended Actions

1. **Address inter-class confusion (highest impact)**
   - Use product reference images to create contrastive training signal
   - Consider a two-stage approach: detect generic "product" first, then classify with a separate model
   - Group visually similar categories (all WASA knekkebrød) into super-categories

2. **Increase resolution to 1280px**
   - Small products (<10k px²) are completely missed at 640px
   - Shelf images are high-res (4032x3024) — downscaling to 640 loses too much detail

3. **More training epochs with larger model**
   - 30 epochs is too few for 356 classes
   - YOLOv8m with 100+ epochs
   - But this alone won't fix confusion — the model needs help distinguishing similar products

4. **Explore two-stage detection + classification**
   - Stage 1: Generic product detection (high recall, ignore category)
   - Stage 2: Product classification using reference images
   - This separates the detection problem (easy) from classification (hard)

---

## 2026-03-21: Two-Stage Pipeline & Classification Experiments

### Detection Results (1-class and 9-class)

| Model | mAP@0.5 | Precision | Recall | Score Ceiling |
|---|---|---|---|---|
| **1-class detector** (yolov8m, 50ep) | **0.934** | 0.899 | 0.907 | 0.654 (70%) |
| **9-class superclass** (yolov8m, 50ep) | **0.828** | 0.790 | 0.793 | — |
| 356-class baseline (yolov8s, 30ep) | 0.038 | 0.187 | 0.057 | — |

Detection is solved — 1-class detector at 93.4% mAP. The 9-class superclass detector also works well, grouping products into: knekkebroed, coffee, tea, cereal, eggs, spread, cookies, chocolate, other.

### Classification Experiments

| Approach | Top-1 Accuracy | Notes |
|---|---|---|
| YOLO-cls (343 classes) | <1% | Massive overfit |
| YOLO-cls per group | <9% | Still overfits |
| **YOLO backbone + LogReg (global)** | **33%** | Frozen backbone features, no overfit |
| **YOLO backbone + LogReg (per group)** | **47.3%** | +14pp from grouping |
| **YOLO backbone + OCR + LogReg (per group)** | **55-67%** | OCR boosts hardest groups |
| CLIP zero-shot (confusable subset) | 69.5% | Benchmark, not sandbox-compatible |
| CLIP + OCR (confusable subset) | 75.2% | Best result, needs ONNX for sandbox |
| timm EfficientNet classifier | 0.4% | Complete overfit |

### OCR Comparison

| OCR Model | Avg Word Match |
|---|---|
| EasyOCR | 25% |
| TrOCR | 10% |

### Key Finding: OCR + Backbone Features Combined

OCR word-match features boost classification significantly when added to YOLO backbone features:

| Group | Backbone Only | + OCR | Boost | Top-5 |
|---|---|---|---|---|
| chocolate (13 cls) | 80% | **83%** | +3pp | 99% |
| knekkebroed (55 cls) | 57% | **67%** | +10pp | 90% |
| coffee (66 cls) | 50% | **55%** | +5pp | 89% |
| tea (34 cls) | 53% | **57%** | +4pp | 92% |

The biggest boost is on knekkebroed (+10pp) — exactly where products are most visually similar and text is the main differentiator.

### Planned Pipeline

```
Shelf Image
  → 9-class YOLO detector → product boxes + super-category
  → Crop each box
  → Per-group classification:
    1. YOLO backbone features (576d)
    2. OCR word-match scores
    3. LogReg soft labels → guide YOLO-cls training (knowledge distillation)
    4. Ensemble: YOLO-cls P1 + LogReg+OCR P2 → final category_id
```

### Sandbox Compatibility

| Component | In Sandbox? | Solution |
|---|---|---|
| YOLO detector | Yes (ultralytics) | .pt weights |
| YOLO backbone features | Yes (ultralytics) | Same model |
| sklearn LogReg | Yes (scikit-learn) | .joblib file |
| OCR (EasyOCR) | No | Need ONNX export or skip |
| CLIP | No | Need ONNX export or skip |

Without OCR/CLIP: estimated score ≈ 0.77 (detection 0.654 + classification 0.47 × 0.934 × 0.3)
With OCR boost: estimated score ≈ 0.83
