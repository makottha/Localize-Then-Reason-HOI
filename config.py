# config.py

import torch

# -----------------------
# YOLO CONFIG
# -----------------------
YOLO_MODEL_PATH = "yolo11n.pt"   # or whatever you use
YOLO_CONF_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.45
PAIR_IOU_THRESHOLD = 0.3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# CROP CONFIG
# -----------------------
CROP_SCALE = 1.8        # expand union box by this factor
CROP_MIN_SIZE = 128     # minimum width/height in pixels

# -----------------------
# CLIP / VLM CONFIG
# -----------------------
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

CLIP_TEXT_PROMPTS = [
    "a person riding a bicycle",              # index 0 → rider
    "a person standing next to a bicycle",    # index 1 → non-rider
]
RIDER_IDX = 0
NON_RIDER_IDX = 1
RIDER_THRESHOLD = 0.5

# -----------------------
# DATA PATHS
# -----------------------
CROPS_DIR = "crops"
CLIP_SCORES_CSV = "clip_rider_scores_v2.csv"
CLIP_BATCH_SIZE = 32
