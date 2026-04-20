import os
import sys
from PIL import Image

import torch
from torchvision import transforms

# --- Add project root to path ---
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.vit_bilstm_ctc import ViTBILSTMCTC
from src.utils.text_utils import TextEncoder


MODEL_PATH = "output/checkpoints/vit_lstm_te_examhand.pth"

LANG_CODE = "te"
IMG_SIZE = (224, 224)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Same normalization as training (but without augmentations)
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def load_model():
    print(f"Using device: {device}")
    text_encoder = TextEncoder(lang_code=LANG_CODE)
    vocab_size = text_encoder.vocab_size()

    model = ViTBILSTMCTC(num_classes=vocab_size)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, text_encoder


def preprocess_image(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img)  # (3, H, W)
    return tensor.unsqueeze(0)  # (1, 3, H, W)


def ctc_greedy_decode(logits: torch.Tensor, blank: int = 0):
    """
    logits: (T, N, C)
    returns: list of label indices (after CTC collapse)
    """
    # log_softmax along classes
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)
    # best path
    best_path = log_probs.argmax(2)  # (T, N)
    best_path = best_path[:, 0].tolist()  # batch size = 1

    decoded = []
    prev = None
    for p in best_path:
        if p != blank and p != prev:
            decoded.append(p)
        prev = p
    return decoded


def main():
    if len(sys.argv) < 2:
        print("Usage: python infer.py path/to/image")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        sys.exit(1)

    model, text_encoder = load_model()
    image_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        logits = model(image_tensor)  # (T, N, C)

    pred_indices = ctc_greedy_decode(logits, blank=0)
    predicted_text = text_encoder.decode(pred_indices)

    print("================================")
    print(" Image:", image_path)
    print(" Predicted Text:", predicted_text)
    print("================================")


if __name__ == "__main__":
    main()
