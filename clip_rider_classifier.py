# clip_rider_classifier.py
import torch
import open_clip
from PIL import Image
from typing import List, Tuple

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RIDER_PROMPTS = [
    "a person riding a bicycle on a city street on a cloudy day",
    "a cyclist riding a bike past a bus stop in rainy weather",
    "a person in a jacket riding a bicycle on a wet road",
    "a commuter riding a bicycle near a bus stop in low light",
    "a person sitting on a bicycle and riding it on an overcast day",
    "a cyclist riding a bike in urban traffic during winter"
]


NON_RIDER_PROMPTS = [
    "a person in a coat standing next to a parked bicycle at a bus stop",
    "a person walking while pushing a bicycle on a wet sidewalk",
    "a parked bicycle near people waiting at a bus stop in rainy weather",
    "a person holding a bicycle but not riding it on a city street",
    "a bicycle locked near a bus stop with pedestrians in jackets",
    "a person standing at a bus stop with a bicycle beside them"
]



class RiderNonRiderClassifier:
    """
    CLIP-based binary classifier: rider vs non-rider, with multiple prompts
    per class and averaged text prototypes.
    """

    def __init__(self, device: str = DEVICE) -> None:
        self.device = device
        print(f"[INFO] Loading CLIP (open_clip ViT-B-32) on device: {self.device}")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion2b_s34b_b79k",
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

        # Build text prototypes for rider and non-rider
        with torch.no_grad():
            rider_tokens = self.tokenizer(RIDER_PROMPTS).to(self.device)
            non_rider_tokens = self.tokenizer(NON_RIDER_PROMPTS).to(self.device)

            rider_feats = self.model.encode_text(rider_tokens)   # (Nr, D)
            non_rider_feats = self.model.encode_text(non_rider_tokens)  # (Nn, D)

            # Normalize each feature
            rider_feats = rider_feats / rider_feats.norm(dim=-1, keepdim=True)
            non_rider_feats = non_rider_feats / non_rider_feats.norm(dim=-1, keepdim=True)

            # Average prompts per class → 2 prototypes
            rider_proto = rider_feats.mean(dim=0, keepdim=True)       # (1, D)
            non_rider_proto = non_rider_feats.mean(dim=0, keepdim=True)  # (1, D)

            # Stack: class_text_feats: (2, D) in float32
            class_text_feats = torch.cat([rider_proto, non_rider_proto], dim=0)
            class_text_feats = class_text_feats / class_text_feats.norm(dim=-1, keepdim=True)

        # Store as float32; we'll cast at runtime if image_features are fp16
        self.class_text_feats = class_text_feats  # (2, D) float32

        print(f"[INFO] CLIP text prototypes ready (rider_prompts={len(RIDER_PROMPTS)}, "
              f"non_rider_prompts={len(NON_RIDER_PROMPTS)})")

    @torch.no_grad()
    def predict_probs(self, pil_img: Image.Image) -> Tuple[float, float]:
        """
        Returns (rider_prob, non_rider_prob) for the given crop.

        Uses AMP on CUDA but ensures image_features and class_text_feats
        share the same dtype before the matrix multiply.
        """
        img_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        # Use new autocast API; only for encoding the image
        use_amp = (self.device == "cuda")
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            image_features = self.model.encode_image(img_input)

        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 🔧 DTYPE FIX: make sure text feats match image_feats dtype
        class_text_feats = self.class_text_feats.to(image_features.dtype)

        # Similarity logits: (1, 2)
        logits = (image_features @ class_text_feats.T)
        probs = logits.softmax(dim=-1).cpu().numpy()[0]

        rider_prob = float(probs[0])
        non_rider_prob = float(probs[1])

        return rider_prob, non_rider_prob
