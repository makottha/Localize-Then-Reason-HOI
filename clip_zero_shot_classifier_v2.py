import os
import csv
import torch
from PIL import Image
from typing import List, Tuple

import open_clip  # pip install open-clip-torch

# ---------------------------
# CONFIG
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CROPS_DIR = "crops"          # directory with union crops from yolo_crops.py
OUTPUT_CSV = "clip_rider_scores_v2.csv"
BATCH_SIZE = 32

# Exactly two prompts → binary comparison
TEXT_PROMPTS = [
    "a person riding a bicycle",              # index 0 → rider
    "a person standing next to a bicycle"     # index 1 → non-rider
]

RIDER_IDX = 0
NON_RIDER_IDX = 1
RIDER_THRESHOLD = 0.5  # can tune later


# ---------------------------
# Model Loader
# ---------------------------
def load_clip_model():
    """
    Load a CLIP model + preprocess from open_clip.
    Using ViT-B-32 as a good speed/accuracy trade-off.
    """
    print(f"[INFO] Loading CLIP model on device: {DEVICE}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k",
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(DEVICE)
    model.eval()
    return model, preprocess, tokenizer


# ---------------------------
# Utility: collect image files
# ---------------------------
def list_image_files(root_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = []
    for fname in os.listdir(root_dir):
        ext = os.path.splitext(fname.lower())[1]
        if ext in exts:
            files.append(os.path.join(root_dir, fname))
    files.sort()
    return files


# ---------------------------
# Zero-shot scoring per batch
# ---------------------------
def compute_batch_scores(
    model,
    preprocess,
    tokenizer,
    image_paths: List[str],
) -> List[Tuple[str, float, float, str]]:
    """
    For a batch of image paths, returns list of:
        (image_path, rider_prob, non_rider_prob, pred_label)
    where probs are from a softmax over [rider_text, non_rider_text].
    """
    images = []
    valid_paths = []

    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            images.append(preprocess(img))
            valid_paths.append(p)
        except Exception as e:
            print(f"[WARN] Failed to load image {p}: {e}")

    if not images:
        return []

    images = torch.stack(images).to(DEVICE)

    # Two-class text prompts
    text_tokens = tokenizer(TEXT_PROMPTS).to(DEVICE)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
        image_features = model.encode_image(images)
        text_features = model.encode_text(text_tokens)

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Similarity: (batch, 2)
        logits = image_features @ text_features.T
        probs = logits.softmax(dim=-1)  # (batch, 2)

    results = []
    for i, img_path in enumerate(valid_paths):
        rider_prob = probs[i, RIDER_IDX].item()
        non_rider_prob = probs[i, NON_RIDER_IDX].item()

        pred_label = "rider" if rider_prob >= RIDER_THRESHOLD else "non_rider"

        results.append((img_path, rider_prob, non_rider_prob, pred_label))

    return results


# ---------------------------
# Main routine
# ---------------------------
def main():
    if os.path.dirname(OUTPUT_CSV):
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    image_files = list_image_files(CROPS_DIR)
    print(f"[INFO] Found {len(image_files)} crop images in '{CROPS_DIR}'")

    if not image_files:
        print("[WARN] No images found. Check CROPS_DIR.")
        return

    model, preprocess, tokenizer = load_clip_model()

    # CSV header: crop_path, rider_prob, non_rider_prob, pred_label
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["crop_path", "rider_prob", "non_rider_prob", "pred_label"])

        total = len(image_files)
        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            batch_paths = image_files[start:end]
            print(f"[INFO] Processing {start+1}/{total} → {end}/{total}")

            batch_results = compute_batch_scores(
                model,
                preprocess,
                tokenizer,
                batch_paths,
            )

            for (p, rider_p, non_rider_p, label) in batch_results:
                writer.writerow([p, f"{rider_p:.4f}", f"{non_rider_p:.4f}", label])

    print(f"\n✅ Done. Scores written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
