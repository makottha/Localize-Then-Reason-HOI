# yolo_detector.py

from typing import List, Tuple

from PIL import Image
from ultralytics import YOLO

from config import (
    YOLO_MODEL_PATH,
    YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    PAIR_IOU_THRESHOLD,
    DEVICE,
    CROP_SCALE,
    CROP_MIN_SIZE,
)
from detection_types import (
    Detection,
    PersonBicyclePair,
    compute_iou,
    union_box,
    expand_box,
)


class YoloPersonBicycleDetector:
    """
    Runs YOLO on an image, finds person+bicycle pairs, and
    (optionally) prepares expanded union boxes.
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

        # Resolve class IDs for person and bicycle from model names
        self.names = self.model.model.names if hasattr(self.model, "model") else self.model.names
        self.person_id = self._get_class_id("person")
        self.bicycle_id = self._get_class_id("bicycle")

        print(f"[INFO] Detected class IDs: person={self.person_id}, bicycle={self.bicycle_id}")

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

    def run_detection(self, image_path: str) -> List[Detection]:
        """
        Run YOLO on an image and return filtered detections for person and bicycle.
        """
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
                iou = compute_iou(p.box_xyxy, b.box_xyxy)
                if iou >= self.pair_iou_thres:
                    ubox = union_box(p.box_xyxy, b.box_xyxy, img_w, img_h)
                    pairs.append(PersonBicyclePair(person=p, bicycle=b, union_box=ubox))
        return pairs

    def generate_expanded_crops(
        self,
        image_path: str,
    ) -> Tuple[Image.Image, List[Tuple[PersonBicyclePair, Tuple[int, int, int, int]]]]:
        """
        Utility used by both crop scripts and video annotator.

        Returns:
            original PIL image,
            list of (pair, expanded_box) where expanded_box=(x1,y1,x2,y2)
        """
        img = Image.open(image_path).convert("RGB")
        img_w, img_h = img.size

        detections = self.run_detection(image_path)
        pairs = self.build_pairs(detections, (img_w, img_h))

        expanded = []
        for pair in pairs:
            ex1, ey1, ex2, ey2 = expand_box(
                pair.union_box,
                img_w=img_w,
                img_h=img_h,
                scale=CROP_SCALE,
                min_size=CROP_MIN_SIZE,
            )
            if ex2 <= ex1 or ey2 <= ey1:
                continue
            expanded.append((pair, (ex1, ey1, ex2, ey2)))

        return img, expanded
