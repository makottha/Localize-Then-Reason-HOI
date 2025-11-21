# config.py
"""
Global configuration for VLM-based cyclist detection experiments.
"""

import torch

# YOLO model (COCO-pretrained)
YOLO_MODEL_PATH = "yolo11n.pt"  # or "yolov8n.pt", update as needed

# Detection settings
YOLO_CONF_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.45  # NMS threshold inside YOLO

# Person–bicycle pairing
PAIR_IOU_THRESHOLD = 0.30  # min IoU between person and bicycle boxes to form a candidate pair

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
