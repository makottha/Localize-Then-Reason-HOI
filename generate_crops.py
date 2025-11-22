# generate_crops.py

#!/usr/bin/env python

import argparse
import os
from typing import List

from yolo_detector import YoloPersonBicycleDetector
from config import CROPS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate expanded person+bicycle union crops using YOLO."
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="./images",
        help="Directory containing images to process (jpg/png/jpeg).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=CROPS_DIR,
        help="Directory to save union crops.",
    )
    return parser.parse_args()


def list_images(image_dir: str) -> List[str]:
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if os.path.splitext(f.lower())[1] in valid_exts
    ]
    files.sort()
    return files


def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.image_dir):
        raise NotADirectoryError(
            f"[ERROR] The provided --image-dir does not exist or is not a directory: {args.image_dir}"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    detector = YoloPersonBicycleDetector()

    image_files = list_images(args.image_dir)
    print(f"[INFO] Found {len(image_files)} images in '{args.image_dir}'")

    total_pairs = 0

    for img_path in image_files:
        try:
            print(f"\n[INFO] Processing {img_path}")
            img, expanded_pairs = detector.generate_expanded_crops(img_path)

            for idx, (pair, (x1, y1, x2, y2)) in enumerate(expanded_pairs):
                crop = img.crop((x1, y1, x2, y2))
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                fname = (
                    f"crop_{base_name}_pair{idx}_"
                    f"p{pair.person.conf:.2f}_b{pair.bicycle.conf:.2f}.jpg"
                )
                out_path = os.path.join(args.output_dir, fname)
                crop.save(out_path)
                pair.crop_path = out_path

            print(f"[INFO] → {len(expanded_pairs)} cyclist candidate crops")
            total_pairs += len(expanded_pairs)

        except Exception as e:
            print(f"[ERROR] Failed on '{img_path}' → {str(e)}")
            continue

    print("\n========== SUMMARY ==========")
    print(f"Processed images: {len(image_files)}")
    print(f"Detected person+bicycle pairs: {total_pairs}")
    print(f"Crops saved in: {args.output_dir}")
    print("================================")


if __name__ == "__main__":
    main()
