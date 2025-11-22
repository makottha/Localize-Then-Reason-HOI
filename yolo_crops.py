#!/usr/bin/env python
"""
YOLO-based crop generator for person+bicycle regions.

Given an input image, this script:
  1. Runs YOLO (COCO-pretrained) to detect objects.
  2. Filters detections for 'person' and 'bicycle'.
  3. Forms person–bicycle pairs based on IoU overlap.
  4. Creates *expanded* union bounding boxes and crops them from the image.
  5. Optionally writes the crops to an output directory.

These crops can then be passed to a CLIP-based classifier to decide
whether the person is riding the bicycle or not.
"""

import argparse
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

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

# === Crop expansion hyperparameters ===
CROP_SCALE = 1.8   # how much to expand the union box (1.5–2.0 is a good range)
CROP_MIN_SIZE = 128  # minimum width/height of crop in pixels


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


class YoloPersonBicycleCropper:
    """
    Runs YOLO on an image, finds person+bicycle pairs, and generates union crops.
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
        # ultralytics handles device inside predict call; we keep device for consistency

        # Resolve class IDs for person and bicycle from model names
        self.names = self.model.model.names if hasattr(self.model, "model") else self.model.names
        self.person_id = self._get_class_id("person")
        self.bicycle_id = self._get_class_id("bicycle")

        print(f"[INFO] Detected class IDs: person={self.person_id}, bicycle={self.bicycle_id}")

    def _get_class_id(self, class_name: str) -> int:
        """
        Find the numeric class id for a given class name in YOLO's names dict.
        """
        if isinstance(self.names, dict):
            for k, v in self.names.items():
                if v == class_name:
                    return int(k)
        else:
            # names is a list
            for idx, name in enumerate(self.names):
                if name == class_name:
                    return idx
        raise ValueError(f"Class '{class_name}' not found in YOLO model names.")

    @staticmethod
    def _compute_iou(boxA: Tuple[float, float, float, float],
                     boxB: Tuple[float, float, float, float]) -> float:
        """
        Compute IoU between two boxes in [x1, y1, x2, y2] format.
        """
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
        iou = interArea / max(boxAArea + boxBArea - interArea, 1e-6)
        return float(iou)

    @staticmethod
    def _union_box(
        boxA: Tuple[float, float, float, float],
        boxB: Tuple[float, float, float, float],
        img_w: int,
        img_h: int,
    ) -> Tuple[int, int, int, int]:
        """
        Compute the union bounding box of two boxes and clip to image bounds.
        """
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
        """
        Expand a box around its center by a given scale factor and enforce a minimum size.
        Clips the result to image boundaries.

        Args:
            box: (x1, y1, x2, y2)
            img_w, img_h: image dimensions
            scale: multiplicative expansion factor for width/height
            min_size: minimum width/height in pixels after expansion

        Returns:
            Expanded box (x1, y1, x2, y2) as ints.
        """
        x1, y1, x2, y2 = box
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        # Expand
        w_exp = max(w * scale, min_size)
        h_exp = max(h * scale, min_size)

        cx = x1 + w / 2.0
        cy = y1 + h / 2.0

        new_x1 = int(max(0, cx - w_exp / 2.0))
        new_y1 = int(max(0, cy - h_exp / 2.0))
        new_x2 = int(min(img_w, cx + w_exp / 2.0))
        new_y2 = int(min(img_h, cy + h_exp / 2.0))

        return new_x1, new_y1, new_x2, new_y2

    def run_detection(self, image_path: str) -> List[Detection]:
        """
        Run YOLO on an image and return filtered detections for person and bicycle.
        """
        # Run model
        results = self.model(
            image_path,
            conf=self.conf_thres,
            iou=self.iou_thres,
            verbose=False,
        )
        r = results[0]

        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return []

        cls_ids = boxes.cls.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        detections: List[Detection] = []
        for cls_id, conf, box in zip(cls_ids, confs, xyxy):
            cls_id_int = int(cls_id)
            cls_name = self.names[cls_id_int] if isinstance(self.names, (dict, list)) else str(cls_id_int)
            if cls_id_int not in (self.person_id, self.bicycle_id):
                continue
            det = Detection(
                cls_id=cls_id_int,
                cls_name=cls_name,
                conf=float(conf),
                box_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            )
            detections.append(det)

        print(f"[DEBUG] Found {len(detections)} person/bicycle detections in '{image_path}'")
        return detections

    def build_pairs(
        self,
        detections: List[Detection],
        img_size: Tuple[int, int],
    ) -> List[PersonBicyclePair]:
        """
        Given detections, build person–bicycle pairs with IoU > threshold.
        """
        img_w, img_h = img_size
        persons = [d for d in detections if d.cls_id == self.person_id]
        bicycles = [d for d in detections if d.cls_id == self.bicycle_id]

        pairs: List[PersonBicyclePair] = []

        for p in persons:
            for b in bicycles:
                iou = self._compute_iou(p.box_xyxy, b.box_xyxy)
                if iou >= self.pair_iou_thres:
                    union = self._union_box(p.box_xyxy, b.box_xyxy, img_w, img_h)
                    pairs.append(
                        PersonBicyclePair(
                            person=p,
                            bicycle=b,
                            union_box=union,
                        )
                    )

        print(f"[DEBUG] Built {len(pairs)} person+bicycle pairs (IoU>={self.pair_iou_thres})")
        return pairs

    def generate_crops(
        self,
        image_path: str,
        output_dir: str = "",
        prefix: str = "crop",
    ) -> List[PersonBicyclePair]:
        """
        Full pipeline for one image:
          - Run YOLO
          - Build person+bicycle pairs
          - Generate *expanded* crops for each pair (optionally save to disk)

        Args:
            image_path: Path to input image.
            output_dir: If non-empty, crops will be saved here.
            prefix: Filename prefix for saved crops.

        Returns:
            List[PersonBicyclePair] with crop_path populated if saved.
        """
        img = Image.open(image_path).convert("RGB")
        img_w, img_h = img.size

        detections = self.run_detection(image_path)
        pairs = self.build_pairs(detections, (img_w, img_h))

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        for idx, pair in enumerate(pairs):
            # Start from the tight union box...
            ux1, uy1, ux2, uy2 = pair.union_box
            # ...then expand it for more context and minimum size
            ex1, ey1, ex2, ey2 = self._expand_box(
                (ux1, uy1, ux2, uy2),
                img_w=img_w,
                img_h=img_h,
            )

            # Safety check: skip degenerate boxes
            if ex2 <= ex1 or ey2 <= ey1:
                print(f"[WARN] Skipping degenerate expanded box for pair {idx} in {image_path}")
                continue

            crop_img = img.crop((ex1, ey1, ex2, ey2))

            if output_dir:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                fname = (
                    f"{prefix}_{base_name}_pair{idx}_"
                    f"p{pair.person.conf:.2f}_b{pair.bicycle.conf:.2f}.jpg"
                )
                out_path = os.path.join(output_dir, fname)
                crop_img.save(out_path)
                pair.crop_path = out_path

        return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate person+bicycle union crops using YOLO."
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default='./images',
        help="Directory containing images to process (jpg/png/jpeg).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="crops",
        help="Directory to save union crops (default: 'crops').",
    )
    parser.add_argument(
        "--pair-iou",
        type=float,
        default=PAIR_IOU_THRESHOLD,
        help="IoU threshold to form person+bicycle pairs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate directory
    if not os.path.isdir(args.image_dir):
        raise NotADirectoryError(
            f"[ERROR] The provided --image-dir does not exist or is not a directory: {args.image_dir}"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # Load YOLO model + class IDs
    cropper = YoloPersonBicycleCropper(
        pair_iou_thres=args.pair_iou,
    )

    # Collect all image files
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [
        os.path.join(args.image_dir, f)
        for f in os.listdir(args.image_dir)
        if os.path.splitext(f.lower())[1] in valid_exts
    ]
    image_files.sort()

    print(f"[INFO] Found {len(image_files)} images in '{args.image_dir}'")

    total_pairs = 0

    # Process in sorted order
    for img_path in image_files:
        try:
            print(f"\n[INFO] Processing {img_path}")
            pairs = cropper.generate_crops(
                image_path=img_path,
                output_dir=args.output_dir,
            )
            print(f"[INFO] → {len(pairs)} cyclist candidate crops")
            total_pairs += len(pairs)
        except Exception as e:
            print(f"[ERROR] Failed on '{img_path}' → {str(e)}")
            continue

    print("\n========== SUMMARY ==========")
    print(f"Processed images: {len(image_files)}")
    print(f"Detected person+bicycle pairs (IoU≥{args.pair_iou}): {total_pairs}")
    print(f"Crops saved in: {args.output_dir}")
    print("================================")


if __name__ == "__main__":
    main()
