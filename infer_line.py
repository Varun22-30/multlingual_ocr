# infer_line.py  (debug version)

import os
import sys
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

# --- project root ---
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.vit_bilstm_ctc import ViTBILSTMCTC
from src.utils.text_utils import TextEncoder

LANG_CODE = "te"
MODEL_PATH = r"output\checkpoints\vit_lstm_te_lines_synth.pth"  # line model checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # must match training
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def ctc_greedy_decode(logits, blank=0):
    """
    logits: (T, N, C)
    returns: list of label indices (int)
    """
    log_probs = nn.functional.log_softmax(logits, dim=2)
    # shape: (T, N)
    best_path = log_probs.argmax(2)
    best_path = best_path[:, 0].tolist()  # for N=1 batch

    decoded = []
    prev = None
    for p in best_path:
        if p != blank and p != prev:
            decoded.append(p)
        prev = p
    return decoded, best_path


def load_model():
    text_encoder = TextEncoder(lang_code=LANG_CODE)
    vocab_size = text_encoder.vocab_size()

    print(f"Using device: {device}")
    print(f"Vocab size in TextEncoder: {vocab_size}")

    model = ViTBILSTMCTC(num_classes=vocab_size)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    print("Loading checkpoint:", MODEL_PATH)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, text_encoder


def infer_line(img_path):
    model, text_encoder = load_model()

    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)

    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)  # (1, 3, H, W)

    with torch.no_grad():
        logits = model(tensor)  # expect (T, N, C)
        print("Logits shape:", logits.shape)

    # DEBUG: show a small slice of time-steps and top 3 classes each
    with torch.no_grad():
        log_probs = nn.functional.log_softmax(logits, dim=2)
        lp = log_probs[:, 0, :]  # (T, C) for N=1
        T, C = lp.shape
        print(f"T={T}, C={C}")
        # show 5 evenly spaced time steps
        for t in torch.linspace(0, T - 1, steps=min(5, T)).long():
            vals, idxs = lp[t].topk(5)
            print(f"t={t.item():02d}: top5 indices={idxs.tolist()}  log_probs={[round(v.item(), 2) for v in vals]}")

    ids, raw_path = ctc_greedy_decode(logits, blank=0)
    print("Raw best path indices (first 50):", raw_path[:50])
    print("Collapsed (non-blank, no repeats):", ids)

    text = text_encoder.decode(ids).strip()

    print("================================")
    print(f" Image: {img_path}")
    print(" Decoded chars:", [repr(ch) for ch in text])
    print(f" Predicted line: {text}")
    print("================================")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python infer_line.py path_to_line_image.png")
        sys.exit(1)

    infer_line(sys.argv[1])
