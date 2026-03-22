"""
Train 9-class superclass detector on GPU.
Use yolov8m (fits in 420MB) at 1280px for best results.

Usage (on GPU machine):
    pip install ultralytics==8.1.0
    python train_detector_gpu.py
"""
import torch
_orig = torch.load
torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})

from pathlib import Path
from ultralytics import YOLO

# Adjust paths for your GPU machine
DATASET = Path(__file__).parent.parent / "output" / "yolo_dataset_superclass" / "data.yaml"
OUT = Path(__file__).parent.parent / "output" / "models"

model = YOLO("yolov8m.pt")
model.train(
    data=str(DATASET),
    epochs=100,
    imgsz=1280,
    batch=8,  # adjust for GPU memory
    patience=20,
    project=str(OUT),
    name="superclass_yolov8m_e100_img1280",
    exist_ok=True,
    verbose=True,
    flipud=0.0, fliplr=0.5,
    mosaic=1.0, mixup=0.15,
    copy_paste=0.2,
    hsv_h=0.02, hsv_s=0.7, hsv_v=0.5,
    degrees=0.0, translate=0.15, scale=0.5,
    erasing=0.3,
)
