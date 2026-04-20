# src/inference/infer_hindi.py
# --------------------------------------------------
# Hindi OCR Inference (CTC Beam Search)
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
# CTC BEAM SEARCH (NO LANGUAGE MODEL)
# =================================================
def ctc_beam_search(log_probs, beam_width=5, blank=0):
    """
    log_probs: (T, C) log probabilities
    returns best path (list of indices)
    """

    beams = {(): 0.0}  # sequence -> score

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

        # keep top beams
        beams = dict(
            sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width]
        )

    best_seq = max(beams.items(), key=lambda x: x[1])[0]
    return list(best_seq)


# ---------------- PREDICT ----------------
def predict(model, image_tensor, text_encoder):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)

        logits = model(image_tensor)        # (T, N, C)
        log_probs = F.log_softmax(logits, dim=2)

        # Take batch index 0
        log_probs = log_probs[:, 0, :]      # (T, C)

        best_path = ctc_beam_search(
            log_probs,
            beam_width=5,
            blank=0
        )

        text = text_encoder.decode(best_path)

    return text


# ---------------- MAIN -------------------
if __name__ == "__main__":

    LANG_CODE = "hi"
    CHECKPOINT = "output/checkpoints/vit_lstm_hi_finetuned_v2.pth"

    text_encoder = TextEncoder(lang_code=LANG_CODE)
    vocab_size = text_encoder.vocab_size()

    model = ViTBILSTMCTC(num_classes=vocab_size).to(DEVICE)

    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    model.load_state_dict(
        torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    )

    print("✅ Hindi OCR model loaded (Beam Search)")

    while True:
        img_path = input("\nEnter image path (or 'q' to quit): ").strip()
        if img_path.lower() == "q":
            break

        if not os.path.exists(img_path):
            print("❌ Image not found")
            continue

        image_tensor = preprocess_image(img_path)
        prediction = predict(model, image_tensor, text_encoder)

        print("📜 Predicted Text:", prediction)