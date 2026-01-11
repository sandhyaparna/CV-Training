# Load a multi label classification models with 5 labels finetuned using pytorch -["efficientnet_v2_s", "EfficientNet_V2_S_Weights"]
# run the model against a folder to filter images with 1&2 labels
# Model is run on the images

import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
import shutil

from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

# -------- CONFIG --------
CHECKPOINT_PATH = "multi_label_classifier_efficientnet_v2_s_224.pt"
INPUT_DIR = "./v1_seg_filter"  # 20000_images
OUTPUT_DIR = "./v1_seg_filter_0.9"  # v1_label5  # v1_label_1_3  # v1_label_1_4 # v1_label_2_3  # v1_label_2_4
NUM_LABELS = 5
THRESHOLD = 0.5  # 0.9
# ------------------------


CLASSES = ["label1", "label2", "label3", "label4", "label5"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(device):
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(weights=weights)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_LABELS)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}

    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()

    return model, weights


def build_preprocess(weights):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0, 0, 0],
            std=[1, 1, 1]
        ),
    ])


def main():
    device = get_device()
    print(f"Using device: {device}")

    model, weights = load_model(device)
    preprocess = build_preprocess(weights)

    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    label1_idx = CLASSES.index("label1")
    label3_idx = CLASSES.index("label3")

    matched = 0
    total = 0

    with torch.no_grad():
        for img_path in input_dir.rglob("*"):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue

            total += 1
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            x = preprocess(img).unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).squeeze(0)

            is_label1 = probs[label1_idx] >= THRESHOLD
            is_label3 = probs[label3_idx] >= THRESHOLD

            if is_label1 and is_label3:
                matched += 1
                print(f"[MATCH] {img_path.name} | "
                      f"label1={probs[label1_idx]:.3f}, "
                      f"label3={probs[label3_idx]:.3f}")

                dest = output_dir / img_path.name
                shutil.copy2(img_path, dest)

    print(f"\nDone. Scanned: {total}, Matched: {matched}")
    print(f"Filtered images saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()




























