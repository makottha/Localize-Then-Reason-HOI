# detection_types.py

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Detection:
    """Single detection result."""
    cls_id: int
    cls_name: str
    conf: float
    box_xyxy: Tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass
class PersonBicyclePair:
    """Represents a paired person+bicycle detection and the union crop."""
    person: Detection
    bicycle: Detection
    union_box: Tuple[int, int, int, int]  # integer pixel coords
    crop_path: str = ""  # optional: file path if saved


def compute_iou(
    boxA: Tuple[float, float, float, float],
    boxB: Tuple[float, float, float, float],
) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH

    if interArea <= 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return float(interArea / max(boxAArea + boxBArea - interArea, 1e-6))


def union_box(
    boxA: Tuple[float, float, float, float],
    boxB: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int]:
    """Compute the union bounding box of two boxes and clip to image bounds."""
    x1 = int(max(0, min(boxA[0], boxB[0])))
    y1 = int(max(0, min(boxA[1], boxB[1])))
    x2 = int(min(img_w - 1, max(boxA[2], boxB[2])))
    y2 = int(min(img_h - 1, max(boxA[3], boxB[3])))
    return x1, y1, x2, y2


def expand_box(
    box: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    scale: float,
    min_size: int,
) -> Tuple[int, int, int, int]:
    """
    Expand a box around its center by a given scale factor and enforce a minimum size.
    Clips the result to image boundaries.
    """
    x1, y1, x2, y2 = box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    w_exp = max(w * scale, min_size)
    h_exp = max(h * scale, min_size)

    cx = x1 + w / 2.0
    cy = y1 + h / 2.0

    new_x1 = int(max(0, cx - w_exp / 2.0))
    new_y1 = int(max(0, cy - h_exp / 2.0))
    new_x2 = int(min(img_w, cx + w_exp / 2.0))
    new_y2 = int(min(img_h, cy + h_exp / 2.0))

    return new_x1, new_y1, new_x2, new_y2
