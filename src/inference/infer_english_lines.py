import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.vit_bilstm_ctc import ViTBILSTMCTC
from src.utils.image_transforms import ResizeAndPad
from src.utils.text_utils import TextEncoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = os.path.join(
    project_root,
    "output",
    "models",
    "english",
    "vit_lstm_en_lines_best.pth",
)

transform = transforms.Compose([
    ResizeAndPad((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def preprocess_image(img_path):
    image = Image.open(img_path).convert("RGB")
    return transform(image).unsqueeze(0)


def ctc_greedy_decode(logits, blank=0):
    probs = F.log_softmax(logits, dim=2).argmax(2)[:, 0].tolist()

    decoded = []
    prev = None
    for idx in probs:
        if idx != blank and idx != prev:
            decoded.append(idx)
        prev = idx
    return decoded


def predict(model, image_tensor, encoder):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        logits = model(image_tensor)
        path = ctc_greedy_decode(logits, blank=0)
        text = encoder.decode(path)

        probs = torch.softmax(logits, dim=2)
        max_probs, preds = probs.max(2)
        non_blank_mask = preds[:, 0] != 0
        if non_blank_mask.any():
            confidence = max_probs[:, 0][non_blank_mask].mean().item()
        else:
            confidence = 0.0

    return text, confidence


if __name__ == "__main__":
    print("Using device:", DEVICE)

    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    encoder = TextEncoder(lang_code="en")
    model = ViTBILSTMCTC(num_classes=encoder.vocab_size()).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True))

    print("English line OCR model loaded.")

    while True:
        img_path = input("\nEnter line image path (or 'q' to quit): ").strip()
        if img_path.lower() == "q":
            break

        if not os.path.exists(img_path):
            print("Image not found.")
            continue

        image_tensor = preprocess_image(img_path)
        text, confidence = predict(model, image_tensor, encoder)

        print("Prediction:", text)
        print("Confidence:", round(confidence, 4))
