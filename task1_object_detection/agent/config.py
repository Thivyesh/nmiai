"""Configuration constants for the object detection agent."""

from pathlib import Path

# ── Project paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TASK_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = TASK_ROOT / "data"
COCO_DIR = DATA_DIR / "coco_dataset" / "train"
ANNOTATIONS_FILE = COCO_DIR / "annotations.json"
IMAGES_DIR = COCO_DIR / "images"
PRODUCT_IMAGES_DIR = DATA_DIR / "product_images"

OUTPUT_DIR = TASK_ROOT / "output"
MODELS_DIR = OUTPUT_DIR / "models"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
YOLO_DATASET_DIR = OUTPUT_DIR / "yolo_dataset"
AUGMENTED_DIR = OUTPUT_DIR / "augmented"
INFERENCE_DIR = OUTPUT_DIR / "inference"

# ── Dataset stats ──────────────────────────────────────────────────────────────
NUM_IMAGES = 248
NUM_ANNOTATIONS = 22_731
NUM_CATEGORIES = 356  # IDs 0-355
UNKNOWN_CATEGORY_ID = 356  # "unknown_product"

# ── Training defaults ─────────────────────────────────────────────────────────
DEFAULT_MODEL_SIZE = "yolov8m.pt"  # medium is a good balance
DEFAULT_EPOCHS = 50
DEFAULT_IMGSZ = 640
DEFAULT_BATCH_SIZE = 16
DEFAULT_PATIENCE = 10
DEFAULT_LR0 = 0.01
DEFAULT_OPTIMIZER = "AdamW"

# ── Augmentation defaults ─────────────────────────────────────────────────────
DEFAULT_MOSAIC = 1.0
DEFAULT_MIXUP = 0.1
DEFAULT_COPY_PASTE = 0.1
DEFAULT_HSV_H = 0.015
DEFAULT_HSV_S = 0.7
DEFAULT_HSV_V = 0.4
DEFAULT_DEGREES = 10.0
DEFAULT_TRANSLATE = 0.1
DEFAULT_SCALE = 0.5
DEFAULT_FLIPUD = 0.0
DEFAULT_FLIPLR = 0.5

# ── Scoring weights ───────────────────────────────────────────────────────────
DETECTION_WEIGHT = 0.70  # mAP@0.5 for detection
CLASSIFICATION_WEIGHT = 0.30  # mAP@0.5 for classification

# ── Rare class threshold ──────────────────────────────────────────────────────
RARE_CLASS_THRESHOLD = 10  # categories with fewer than N annotations are "rare"
VERY_RARE_THRESHOLD = 3

# ── Timeouts ──────────────────────────────────────────────────────────────────
ANALYZER_TIMEOUT = 120
BOOSTER_TIMEOUT = 300
TRAINER_TIMEOUT = 3600  # training can take a while
