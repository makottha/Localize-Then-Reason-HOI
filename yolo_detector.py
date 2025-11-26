# yolo_pairs.py
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image
from ultralytics import YOLO

from config import (
    YOLO_MODEL_PATH,
    YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    PAIR_IOU_THRESHOLD,
    DEVICE,
)

CROP_SCALE = 1.8       # same as before
CROP_MIN_SIZE = 128    # same as before


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    box_xyxy: Tuple[float, float, float, float]


@dataclass
class PersonBicyclePair:
    person: Detection
    bicycle: Detection
    union_box: Tuple[int, int, int, int]


class YoloPersonBicycleDetector:
    """
    Core YOLO wrapper to detect person+bicycle and build pairs.
    Can work on image paths OR raw frames.
    """

    def __init__(
        self,
        model_path: str = YOLO_MODEL_PATH,
        device: str = DEVICE,
        conf_thres: float = YOLO_CONF_THRESHOLD,
        iou_thres: float = YOLO_IOU_THRESHOLD,
        pair_iou_thres: float = PAIR_IOU_THRESHOLD,
    ) -> None:
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.pair_iou_thres = pair_iou_thres

        print(f"[INFO] Loading YOLO model from '{model_path}' on device: {self.device}")
        self.model = YOLO(model_path)

        self.names = self.model.model.names if hasattr(self.model, "model") else self.model.names
        self.person_id = self._get_class_id("person")
        self.bicycle_id = self._get_class_id("bicycle")
        print(f"[INFO] YOLO classes: person={self.person_id}, bicycle={self.bicycle_id}")

    def _get_class_id(self, class_name: str) -> int:
        if isinstance(self.names, dict):
            for k, v in self.names.items():
                if v == class_name:
                    return int(k)
        else:  # list
            for idx, name in enumerate(self.names):
                if name == class_name:
                    return idx
        raise ValueError(f"Class '{class_name}' not found in YOLO model names.")

    @staticmethod
    def _compute_iou(
        boxA: Tuple[float, float, float, float],
        boxB: Tuple[float, float, float, float],
    ) -> float:
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

    @staticmethod
    def _union_box(
        boxA: Tuple[float, float, float, float],
        boxB: Tuple[float, float, float, float],
        img_w: int,
        img_h: int,
    ) -> Tuple[int, int, int, int]:
        x1 = int(max(0, min(boxA[0], boxB[0])))
        y1 = int(max(0, min(boxA[1], boxB[1])))
        x2 = int(min(img_w - 1, max(boxA[2], boxB[2])))
        y2 = int(min(img_h - 1, max(boxA[3], boxB[3])))
        return x1, y1, x2, y2

    @staticmethod
    def _expand_box(
        box: Tuple[int, int, int, int],
        img_w: int,
        img_h: int,
        scale: float = CROP_SCALE,
        min_size: int = CROP_MIN_SIZE,
    ) -> Tuple[int, int, int, int]:
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

    # ---------- detection on image path (for generate_crops) ----------

    def _run_yolo_on_image_path(self, image_path: str) -> List[Detection]:
        results = self.model(image_path, conf=self.conf_thres, iou=self.iou_thres, verbose=False)
        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return []

        return self._filter_person_bicycle(boxes)

    # ---------- detection on frame (for video) ----------

    def _run_yolo_on_frame(self, frame_bgr: np.ndarray) -> List[Detection]:
        results = self.model(frame_bgr, conf=self.conf_thres, iou=self.iou_thres, verbose=False)
        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return []

        return self._filter_person_bicycle(boxes)

    def _filter_person_bicycle(self, boxes) -> List[Detection]:
        cls_ids = boxes.cls.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        detections: List[Detection] = []
        for cls_id, conf, box in zip(cls_ids, confs, xyxy):
            cls_id_int = int(cls_id)
            if cls_id_int not in (self.person_id, self.bicycle_id):
                continue
            cls_name = (
                self.names[cls_id_int]
                if isinstance(self.names, (dict, list))
                else str(cls_id_int)
            )
            detections.append(
                Detection(
                    cls_id=cls_id_int,
                    cls_name=cls_name,
                    conf=float(conf),
                    box_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                )
            )
        return detections

    # ---------- public helpers ----------

    def detect_pairs_in_image(
        self,
        image_path: str,
    ) -> Tuple[List[Tuple[PersonBicyclePair, Tuple[int, int, int, int]]], Tuple[int, int]]:
        img = Image.open(image_path).convert("RGB")
        img_w, img_h = img.size

        detections = self._run_yolo_on_image_path(image_path)
        return self._build_pairs(detections, img_w, img_h), (img_w, img_h)

    def detect_pairs_in_frame(
        self,
        frame_bgr: np.ndarray,
    ) -> List[Tuple[PersonBicyclePair, Tuple[int, int, int, int]]]:
        h, w = frame_bgr.shape[:2]
        detections = self._run_yolo_on_frame(frame_bgr)
        return self._build_pairs(detections, w, h)

    def _build_pairs(
        self,
        detections: List[Detection],
        img_w: int,
        img_h: int,
    ) -> List[Tuple[PersonBicyclePair, Tuple[int, int, int, int]]]:
        persons = [d for d in detections if d.cls_id == self.person_id]
        bicycles = [d for d in detections if d.cls_id == self.bicycle_id]

        pairs_with_boxes: List[Tuple[PersonBicyclePair, Tuple[int, int, int, int]]] = []

        for p in persons:
            for b in bicycles:
                iou = self._compute_iou(p.box_xyxy, b.box_xyxy)
                if iou >= self.pair_iou_thres:
                    union = self._union_box(p.box_xyxy, b.box_xyxy, img_w, img_h)
                    expanded = self._expand_box(union, img_w, img_h)
                    pairs_with_boxes.append(
                        (PersonBicyclePair(person=p, bicycle=b, union_box=union), expanded)
                    )

        return pairs_with_boxes
