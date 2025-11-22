#!/usr/bin/env python
"""
Annotate a video with cyclist bounding boxes using YOLO + CLIP.

Pipeline per frame:
  1. Run YOLO (COCO-pretrained) to detect 'person' and 'bicycle'.
  2. Form person–bicycle pairs with IoU ≥ PAIR_IOU_THRESHOLD.
  3. Build an expanded union crop around each pair.
  4. Classify each crop with CLIP as 'rider' vs 'non-rider'.
  5. Draw a 'cyclist' box on the frame for crops classified as rider.
  6. Save annotated video.

Requires:
  - config.py with YOLO_MODEL_PATH, YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD, PAIR_IOU_THRESHOLD, DEVICE.
  - clip (OpenAI CLIP), torch, ultralytics, opencv-python, pillow.
"""

import os
import argparse
from dataclasses import dataclass
from typing import Tuple, List

import cv2
import numpy as np
from PIL import Image

import torch
import clip
from ultralytics import YOLO

from config import (
    YOLO_MODEL_PATH,
    YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    PAIR_IOU_THRESHOLD,
    DEVICE,
)

# === Crop expansion hyperparameters ===
CROP_SCALE = 1.8     # expand union box by this factor
CROP_MIN_SIZE = 128  # minimum width/height of crop in pixels

# === CLIP thresholds ===
RIDER_MIN_PROB = 0.55          # minimum rider probability
RIDER_MARGIN = 0.05            # rider_prob - non_rider_prob must exceed this


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


class RiderNonRiderClassifier:
    """
    CLIP-based binary classifier: rider vs non-rider.
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        print(f"[INFO] Loading CLIP model on device: {self.device}")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        # Two prompts: rider vs non-rider
        prompts = [
            "a person riding a bicycle on the street",
            "a person standing next to a bicycle or walking a bicycle",
        ]
        with torch.no_grad():
            self.text_tokens = clip.tokenize(prompts).to(self.device)
            self.text_features = self.model.encode_text(self.text_tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

        self.model.eval()

    @torch.no_grad()
    def predict_probs(self, pil_img: Image.Image) -> Tuple[float, float]:
        """
        Returns (rider_prob, non_rider_prob) for the given crop.
        """
        img_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(img_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Cosine similarity scaled to logits
        logits = (image_features @ self.text_features.T) * 100.0
        probs = logits.softmax(dim=-1).cpu().numpy()[0]
        rider_prob = float(probs[0])
        non_rider_prob = float(probs[1])
        return rider_prob, non_rider_prob


class YoloCyclistAnnotator:
    """
    Uses YOLO to detect person/bicycle, CLIP to classify rider vs non-rider,
    and draws cyclist boxes on video frames.
    """

    def __init__(self) -> None:
        self.device = DEVICE
        print(f"[INFO] Loading YOLO model from '{YOLO_MODEL_PATH}' on device: {self.device}")
        self.yolo = YOLO(YOLO_MODEL_PATH)

        # Resolve class names
        self.names = self.yolo.model.names if hasattr(self.yolo, "model") else self.yolo.names
        self.person_id = self._get_class_id("person")
        self.bicycle_id = self._get_class_id("bicycle")
        print(f"[INFO] YOLO classes: person={self.person_id}, bicycle={self.bicycle_id}")

        # CLIP classifier
        self.clip_classifier = RiderNonRiderClassifier(device=self.device)

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

    def _run_yolo_on_frame(self, frame_bgr: np.ndarray) -> List[Detection]:
        """
        Run YOLO on a BGR frame (OpenCV) and return person/bicycle detections.
        """
        # Ultralytics can take BGR np.ndarray directly
        results = self.yolo(
            frame_bgr,
            conf=YOLO_CONF_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            verbose=False,
        )
        r = results[0]
        boxes = r.boxes

        if boxes is None or len(boxes) == 0:
            return []

        detections: List[Detection] = []
        cls_ids = boxes.cls.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        for cls_id, conf, box in zip(cls_ids, confs, xyxy):
            cls_id_int = int(cls_id)
            if cls_id_int not in (self.person_id, self.bicycle_id):
                continue
            cls_name = self.names[cls_id_int] if isinstance(self.names, (dict, list)) else str(cls_id_int)
            detections.append(
                Detection(
                    cls_id=cls_id_int,
                    cls_name=cls_name,
                    conf=float(conf),
                    box_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                )
            )
        return detections

    def _build_pairs(
        self,
        detections: List[Detection],
        img_w: int,
        img_h: int,
        pair_iou_thres: float,
    ) -> List[PersonBicyclePair]:
        persons = [d for d in detections if d.cls_id == self.person_id]
        bicycles = [d for d in detections if d.cls_id == self.bicycle_id]
        pairs: List[PersonBicyclePair] = []

        for p in persons:
            for b in bicycles:
                iou = self._compute_iou(p.box_xyxy, b.box_xyxy)
                if iou >= pair_iou_thres:
                    union = self._union_box(p.box_xyxy, b.box_xyxy, img_w, img_h)
                    pairs.append(PersonBicyclePair(person=p, bicycle=b, union_box=union))
        return pairs

    def annotate_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Annotate a single frame with cyclist boxes.

        Returns:
            Annotated frame (BGR).
        """
        h, w = frame_bgr.shape[:2]
        detections = self._run_yolo_on_frame(frame_bgr)
        pairs = self._build_pairs(detections, w, h, PAIR_IOU_THRESHOLD)

        if not pairs:
            return frame_bgr

        print("Processing some pairs")

        # For CLIP, we need RGB crops as PIL images
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        for pair in pairs:
            # Expand union box for more context
            ex1, ey1, ex2, ey2 = self._expand_box(pair.union_box, w, h)
            if ex2 <= ex1 or ey2 <= ey1:
                continue

            crop_rgb = frame_rgb[ey1:ey2, ex1:ex2]
            if crop_rgb.size == 0:
                continue

            crop_pil = Image.fromarray(crop_rgb)

            rider_prob, non_rider_prob = self.clip_classifier.predict_probs(crop_pil)

            # Decide if cyclist
            if (
                rider_prob >= RIDER_MIN_PROB
                and (rider_prob - non_rider_prob) >= RIDER_MARGIN
            ):
                print("processing Rider")
                label = f"Rider {rider_prob:.2f}"
                color = (0, 255, 0)  # green in BGR

                cv2.rectangle(frame_bgr, (ex1, ey1), (ex2, ey2), color, 2)
                # Put label above box
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                text_x = ex1
                text_y = max(0, ey1 - 5)
                cv2.rectangle(
                    frame_bgr,
                    (text_x, text_y - th - 4),
                    (text_x + tw + 4, text_y),
                    color,
                    -1,
                )
                cv2.putText(
                    frame_bgr,
                    label,
                    (text_x + 2, text_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

        return frame_bgr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate a video with cyclist boxes using YOLO + CLIP."
    )
    parser.add_argument(
        "--input",
        type=str,
        default='./output_annotated_video.mp4',
        help="Path to input video file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="annotated_cyclist_video.mp4",
        help="Path to output annotated video.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Optional: maximum number of frames to process (-1 = all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input video not found: {args.input}")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Input video: {args.input}")
    print(f"[INFO] Resolution: {width}x{height}, FPS: {fps:.2f}, Frames: {total_frames}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    annotator = YoloCyclistAnnotator()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = annotator.annotate_frame(frame)
        out.write(annotated)

    cap.release()
    out.release()
    print(f"[INFO] Annotated video saved to: {args.output}")


if __name__ == "__main__":
    main()
