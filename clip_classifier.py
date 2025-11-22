# clip_classifier.py

import os
from typing import List, Tuple

import torch
import open_clip
from PIL import Image

from config import (
    DEVICE,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    CLIP_TEXT_PROMPTS,
    RIDER_IDX,
    NON_RIDER_IDX,
    RIDER_THRESHOLD,
)


class RiderNonRiderClip:
    """
    CLIP-based zero-shot classifier for rider vs non-rider.
    Uses open_clip ViT-B-32.
    """

    def __init__(self) -> None:
        print(f"[INFO] Loading CLIP model '{CLIP_MODEL_NAME}' on device: {DEVICE}")
        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME,
            pretrained=CLIP_PRETRAINED,
        )
        tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)

        self.model = model.to(DEVICE)
        self.preprocess = preprocess
        self.tokenizer = tokenizer

        self.model.eval()

        # Prepare and cache text features
        with torch.no_grad():
            text_tokens = self.tokenizer(CLIP_TEXT_PROMPTS).to(DEVICE)
            text_features = self.model.encode_text(text_tokens)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def predict_single(self, img: Image.Image) -> Tuple[float, float, str]:
        """
        Predict rider/non-rider for a single PIL image crop.
        Returns (rider_prob, non_rider_prob, pred_label).
        """
        image_tensor = self.preprocess(img).unsqueeze(0).to(DEVICE)

        image_features = self.model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = image_features @ self.text_features.T  # (1, 2)
        probs = logits.softmax(dim=-1)[0]

        rider_prob = probs[RIDER_IDX].item()
        non_rider_prob = probs[NON_RIDER_IDX].item()
        label = "rider" if rider_prob >= RIDER_THRESHOLD else "non_rider"

        return rider_prob, non_rider_prob, label

    @torch.no_grad()
    def predict_batch(
        self,
        image_paths: List[str],
    ) -> List[Tuple[str, float, float, str]]:
        """
        Batch prediction for multiple image file paths.
        Returns list of (path, rider_prob, non_rider_prob, label).
        """
        images = []
        valid_paths = []

        for p in image_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(self.preprocess(img))
                valid_paths.append(p)
            except Exception as e:
                print(f"[WARN] Failed to load image {p}: {e}")

        if not images:
            return []

        batch = torch.stack(images).to(DEVICE)

        image_features = self.model.encode_image(batch)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = image_features @ self.text_features.T  # (N, 2)
        probs = logits.softmax(dim=-1)

        results = []
        for i, path in enumerate(valid_paths):
            rider_prob = probs[i, RIDER_IDX].item()
            non_rider_prob = probs[i, NON_RIDER_IDX].item()
            label = "rider" if rider_prob >= RIDER_THRESHOLD else "non_rider"
            results.append((path, rider_prob, non_rider_prob, label))

        return results


def list_image_files(root_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = []
    for fname in os.listdir(root_dir):
        ext = os.path.splitext(fname.lower())[1]
        if ext in exts:
            files.append(os.path.join(root_dir, fname))
    files.sort()
    return files
