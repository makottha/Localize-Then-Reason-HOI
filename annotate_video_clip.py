# annotate_video_clip.py
#!/usr/bin/env python

import os
import argparse

import cv2
import numpy as np
from PIL import Image

from yolo_detector import YoloPersonBicycleDetector
from clip_rider_classifier import RiderNonRiderClassifier

RIDER_THRESHOLD = 0.5
RIDER_MARGIN = 0.05
DEBUG_NON_RIDER_DIR = "debug_non_rider_crops"
os.makedirs(DEBUG_NON_RIDER_DIR, exist_ok=True)

riders_count = 0

class VideoCyclistAnnotator:
    def __init__(self) -> None:
        self.detector = YoloPersonBicycleDetector()
        self.clip_classifier = RiderNonRiderClassifier()

    def annotate_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        pairs_with_boxes = self.detector.detect_pairs_in_frame(frame_bgr)

        if not pairs_with_boxes:
            return frame_bgr

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        riders_this_frame = 0

        for pair, (x1, y1, x2, y2) in pairs_with_boxes:
            if x2 <= x1 or y2 <= y1:
                continue

            crop_rgb = frame_rgb[y1:y2, x1:x2]
            if crop_rgb.size == 0:
                continue

            print("Found some crops")
            crop_pil = Image.fromarray(crop_rgb)
            rider_prob, non_rider_prob = self.clip_classifier.predict_probs(crop_pil)

            # Debug: print probabilities
            print(f"[DEBUG] CLIP probs: rider={rider_prob:.3f}, non_rider={non_rider_prob:.3f}")

            # Save all non-rider crops for debugging
            if not (rider_prob >= non_rider_prob):
                debug_name = f"frame{np.random.randint(1e8)}_r{rider_prob:.3f}_nr{non_rider_prob:.3f}.jpg"
                debug_path = os.path.join(DEBUG_NON_RIDER_DIR, debug_name)
                crop_pil.save(debug_path)
                print(f"[DEBUG] Saved NON-RIDER crop → {debug_path}")

            # Decide rider
            if rider_prob >= non_rider_prob:
                riders_this_frame += 1
                label = f"Rider {rider_prob:.2f}"
                color = (0, 255, 0)

                global riders_count
                riders_count = riders_count + 1

                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                text_x = x1
                text_y = max(0, y1 - 5)
                cv2.rectangle(frame_bgr,
                              (text_x, text_y - th - 4),
                              (text_x + tw + 4, text_y),
                              color, -1)
                cv2.putText(frame_bgr, label,
                            (text_x + 2, text_y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 0), 1, cv2.LINE_AA)

        if riders_this_frame > 0:
            print(f"[DEBUG] Riders detected in frame: {riders_this_frame}")

        return frame_bgr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate a video with cyclist boxes using YOLO + CLIP."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./downloaded_video.mp4",
        help="Input video path")
    parser.add_argument(
        "--output",
        type=str,
        default="annotated_cyclist_video.mp4",
        help="Output video path",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Max frames to process (-1 = all)",
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

    annotator = VideoCyclistAnnotator()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if args.max_frames > 0 and frame_idx > args.max_frames:
            break

        #print(f"[INFO] Processing frame {frame_idx}/{total_frames}")
        annotated = annotator.annotate_frame(frame)
        out.write(annotated)

    cap.release()
    out.release()
    print(f"[INFO] Annotated video saved to: {args.output}")

    print(f"[INFO] Found {riders_count} raiders")


if __name__ == "__main__":
    main()
