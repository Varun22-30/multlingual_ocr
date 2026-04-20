# src/inference/infer_hindi_safe.py
# --------------------------------------------------
# Hindi OCR Inference (SAFE model + Beam Search)
# --------------------------------------------------

import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from collections import defaultdict

# -------- ADD PROJECT ROOT --------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.vit_bilstm_ctc import ViTBILSTMCTC
from src.utils.text_utils import TextEncoder

# ---------------- DEVICE -----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ---------------- TRANSFORMS -------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- PREPROCESS -------------
def preprocess_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image


# =================================================
# CTC BEAM SEARCH (WITH SCORE)
# =================================================
def ctc_beam_search(log_probs, beam_width=5, blank=0):
    """
    log_probs: (T, C)
    returns: best_path, best_score
    """

    beams = {(): 0.0}

    for t in range(log_probs.size(0)):
        new_beams = defaultdict(lambda: -float("inf"))

        for prefix, score in beams.items():
            for c in range(log_probs.size(1)):
                new_score = score + log_probs[t, c].item()

                if c == blank:
                    new_beams[prefix] = max(new_beams[prefix], new_score)
                else:
                    if prefix and prefix[-1] == c:
                        new_prefix = prefix
                    else:
                        new_prefix = prefix + (c,)
                    new_beams[new_prefix] = max(new_beams[new_prefix], new_score)

        beams = dict(
            sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width]
        )

    best_seq, best_score = max(beams.items(), key=lambda x: x[1])
    return list(best_seq), best_score


# ---------------- PREDICT ----------------
def predict(model, image_tensor, text_encoder):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)

        logits = model(image_tensor)            # (T, N, C)
        log_probs = F.log_softmax(logits, dim=2)

        log_probs = log_probs[:, 0, :]           # (T, C)

        path, score = ctc_beam_search(
            log_probs,
            beam_width=5,
            blank=0
        )

        text = text_encoder.decode(path)

        # Confidence heuristic (average log-prob per timestep)
        confidence = score / max(len(path), 1)

    return text, confidence


# ---------------- MAIN -------------------
if __name__ == "__main__":

    LANG_CODE = "hi"
    CHECKPOINT = "output/checkpoints/vit_lstm_hi_safe.pth"

    text_encoder = TextEncoder(lang_code=LANG_CODE)
    vocab_size = text_encoder.vocab_size()

    model = ViTBILSTMCTC(num_classes=vocab_size).to(DEVICE)

    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    model.load_state_dict(
        torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    )

    print("✅ SAFE Hindi OCR model loaded")

    while True:
        img_path = input("\nEnter image path (or 'q' to quit): ").strip()
        if img_path.lower() == "q":
            break

        if not os.path.exists(img_path):
            print("❌ Image not found")
            continue

        image_tensor = preprocess_image(img_path)
        text, confidence = predict(model, image_tensor, text_encoder)

        # Reject low-confidence garbage
        if confidence < -3.0 or len(text) == 0:
            print("📜 Prediction: <UNCERTAIN / UNREADABLE>")
            print("🔍 Confidence:", round(confidence, 3))
        else:
            print("📜 Prediction:", text)
            print("🔍 Confidence:", round(confidence, 3))