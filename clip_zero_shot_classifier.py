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
OUTPUT_CSV = "clip_rider_scores.csv"
BATCH_SIZE = 32

# Prompts for zero-shot classification
POSITIVE_TEXTS = [
    "a person riding a bicycle",
    "a cyclist on a bike",
    "a person sitting on a moving bicycle"
]

NEGATIVE_TEXTS = [
    "a person standing next to a bicycle",
    "a person walking with a bicycle",
    "a bicycle without a rider",
    "an empty parked bicycle"
]


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
        pretrained="laion2b_s34b_b79k",  # common open_clip checkpoint
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
        (image_path, rider_score, non_rider_score, pred_label)
    where scores are softmax probabilities from CLIP.
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

    # Prepare prompts
    all_texts = POSITIVE_TEXTS + NEGATIVE_TEXTS
    text_tokens = tokenizer(all_texts).to(DEVICE)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
        image_features = model.encode_image(images)
        text_features = model.encode_text(text_tokens)

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Similarity: (batch, num_texts)
        logits = image_features @ text_features.T
        probs = logits.softmax(dim=-1)  # convert to probabilities

    num_pos = len(POSITIVE_TEXTS)
    num_neg = len(NEGATIVE_TEXTS)

    results = []
    for i, img_path in enumerate(valid_paths):
        prob_vec = probs[i]  # (num_texts,)

        pos_prob = prob_vec[:num_pos].sum().item()
        neg_prob = prob_vec[num_pos : num_pos + num_neg].sum().item()

        if pos_prob >= neg_prob:
            pred_label = "rider"
        else:
            pred_label = "non_rider"

        results.append((img_path, pos_prob, neg_prob, pred_label))

    return results


# ---------------------------
# Main routine
# ---------------------------
def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True) if os.path.dirname(OUTPUT_CSV) else None

    image_files = list_image_files(CROPS_DIR)
    print(f"[INFO] Found {len(image_files)} crop images in '{CROPS_DIR}'")

    if not image_files:
        print("[WARN] No images found. Check CROPS_DIR.")
        return

    model, preprocess, tokenizer = load_clip_model()

    # CSV header: crop_path, rider_score, non_rider_score, pred_label
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["crop_path", "rider_score", "non_rider_score", "pred_label"])

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

            for (p, rider_s, non_rider_s, label) in batch_results:
                writer.writerow([p, f"{rider_s:.4f}", f"{non_rider_s:.4f}", label])

    print(f"\n✅ Done. Scores written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
