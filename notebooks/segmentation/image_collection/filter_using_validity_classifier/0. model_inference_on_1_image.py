# Load a multi label classification models with 5 labels finetuned using pytorch -["efficientnet_v2_s", "EfficientNet_V2_S_Weights"]
# run the model against a folder to filter images with 1&2 labels

import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision import transforms
# -------- CONFIG --------
CHECKPOINT_PATH = "multi_label_classifier_efficientnet_v2_s_224.pt"   # your finetuned checkpoint
IMAGE_PATH = "v1_seg_images/176_493.jpg"        # image to test
NUM_LABELS = 5
THRESHOLD = 0.5
# ------------------------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_model(device):
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(weights=weights)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_LABELS)

    # THIS is what converts CUDA → CPU/MPS safely
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    # Remove DataParallel prefix if present
    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}

    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()

    return model, weights

def main():
    device = get_device()
    print(f"Using device: {device}")

    model, weights = load_model(device)
    #
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0, 0, 0],
            std=[1, 1, 1]
        ),
    ])

    img = Image.open(IMAGE_PATH).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze(0)

    classes = ["label1", "label2", "label3", "label4", "label5"]

    print("\nPredictions:")
    for cls, p in zip(classes, probs):
        flag = "✔" if p >= THRESHOLD else ""
        print(f"{cls:15s}: {p.item():.4f} {flag}")


print(main())





























