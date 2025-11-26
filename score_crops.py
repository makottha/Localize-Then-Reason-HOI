# score_crops.py

#!/usr/bin/env python

import csv
import os

from config import CROPS_DIR, CLIP_SCORES_CSV, CLIP_BATCH_SIZE
from clip_classifier import RiderNonRiderClip, list_image_files


def main():
    if os.path.dirname(CLIP_SCORES_CSV):
        os.makedirs(os.path.dirname(CLIP_SCORES_CSV), exist_ok=True)

    image_files = list_image_files(CROPS_DIR)
    print(f"[INFO] Found {len(image_files)} crop images in '{CROPS_DIR}'")

    if not image_files:
        print("[WARN] No images found. Check CROPS_DIR.")
        return

    clf = RiderNonRiderClip()

    with open(CLIP_SCORES_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["crop_path", "rider_prob", "non_rider_prob", "pred_label"])

        total = len(image_files)
        for start in range(0, total, CLIP_BATCH_SIZE):
            end = min(start + CLIP_BATCH_SIZE, total)
            batch_paths = image_files[start:end]
            print(f"[INFO] Processing {start+1}/{total} → {end}/{total}")

            batch_results = clf.predict_batch(batch_paths)

            for p, rider_p, non_rider_p, label in batch_results:
                writer.writerow([p, f"{rider_p:.4f}", f"{non_rider_p:.4f}", label])

    print(f"\n✅ Done. Scores written to: {CLIP_SCORES_CSV}")


if __name__ == "__main__":
    main()
