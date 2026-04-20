import os
import sys
import string

import torch
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.vit_bilstm_ctc import ViTBILSTMCTC
from src.utils.text_utils import TextEncoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


TELUGU_MODEL_PATH = "output/models/telugu/vit_lstm_te_best_handwritten_finetuned.pth"
HINDI_MODEL_PATH = "output/models/hindi/vit_lstm_hi_final.pth"
TAMIL_MODEL_PATH = "output/models/tamil/vit_lstm_ta_best.pth"
ENGLISH_MODEL_PATH = "output/models/english/vit_lstm_en_cvl_finetuned.pth"


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def load_model(model_path, lang):
    print(f"Loading {lang} model...")

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    num_classes = checkpoint["classifier.weight"].shape[0]

    encoder = TextEncoder(lang_code=lang)
    model = ViTBILSTMCTC(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()

    return model, encoder


print("\nLoading OCR models...\n")

telugu_model, telugu_encoder = load_model(TELUGU_MODEL_PATH, "te")
hindi_model, hindi_encoder = load_model(HINDI_MODEL_PATH, "hi")
tamil_model, tamil_encoder = load_model(TAMIL_MODEL_PATH, "ta")
english_model, english_encoder = load_model(ENGLISH_MODEL_PATH, "en")

print("\nAll OCR models loaded successfully.\n")


def preprocess(img_path):
    image = Image.open(img_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)
    return image


def predict(model, encoder, image):
    image = image.to(DEVICE)

    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=2)
        max_probs, preds = probs.max(2)
        preds = preds[:, 0].cpu().numpy()
        text = encoder.decode(preds)
        confidence = max_probs.mean().item()

    return text, confidence


def latin_ratio(text):
    chars = [ch for ch in text.strip() if not ch.isspace()]
    if not chars:
        return 0.0
    allowed = set(string.ascii_letters + string.digits)
    return sum(ch in allowed for ch in chars) / len(chars)


def script_ratio(text, start_char, end_char):
    chars = [ch for ch in text.strip() if not ch.isspace()]
    if not chars:
        return 0.0
    return sum(start_char <= ch <= end_char for ch in chars) / len(chars)


def language_validity(text_by_lang):
    text_hi = text_by_lang["Hindi"]
    text_te = text_by_lang["Telugu"]
    text_ta = text_by_lang["Tamil"]
    text_en = text_by_lang["English"]
    return {
        "English": latin_ratio(text_en),
        "Hindi": script_ratio(text_hi, "\u0900", "\u097f"),
        "Telugu": script_ratio(text_te, "\u0c00", "\u0c7f"),
        "Tamil": script_ratio(text_ta, "\u0b80", "\u0bff"),
    }


def adjusted_router_scores(raw_scores, validity_scores):
    adjusted = {}
    for language, (text, confidence) in raw_scores.items():
        validity = validity_scores.get(language, 0.0)
        short_text_penalty = 0.03 if len(text.strip()) <= 1 else 0.0
        score = confidence + (0.20 * validity) - short_text_penalty
        adjusted[language] = score
    return adjusted


def select_multilingual_result(scores):
    validity = language_validity({language: payload["text"] for language, payload in scores.items()})
    raw_scores = {
        language: (payload["text"], payload["confidence"])
        for language, payload in scores.items()
    }
    adjusted_scores = adjusted_router_scores(raw_scores, validity)

    ordered = sorted(adjusted_scores.items(), key=lambda item: item[1], reverse=True)
    best_language, best_adjusted = ordered[0]
    second_adjusted = ordered[1][1] if len(ordered) > 1 else -1.0
    decision_reason = "adjusted_confidence"

    conf_hi = scores["Hindi"]["confidence"]
    conf_te = scores["Telugu"]["confidence"]
    conf_ta = scores["Tamil"]["confidence"]
    conf_en = scores["English"]["confidence"]
    text_en = scores["English"]["text"]

    if abs(conf_hi - conf_te) < 0.01 and conf_hi >= conf_te - 0.002:
        if adjusted_scores["Hindi"] >= adjusted_scores["Telugu"] - 0.01:
            best_language = "Hindi"
            best_adjusted = adjusted_scores["Hindi"]
            decision_reason = "hindi_telugu_tiebreak"

    english_looks_valid = validity["English"] >= 0.7 and len(text_en.strip()) >= 2
    indic_best_conf = max(conf_hi, conf_te, conf_ta)
    indic_best_adjusted = max(
        adjusted_scores["Hindi"],
        adjusted_scores["Telugu"],
        adjusted_scores["Tamil"],
    )

    if (
        english_looks_valid
        and (
            conf_en >= indic_best_conf - 0.03
            or adjusted_scores["English"] >= indic_best_adjusted - 0.03
            or (
                validity["English"] >= 0.9
                and adjusted_scores["English"] >= indic_best_adjusted - 0.06
            )
        )
    ):
        best_language = "English"
        best_adjusted = adjusted_scores["English"]
        decision_reason = "english_rescue"

    confidence_margin = best_adjusted - second_adjusted
    raw_best_confidence = scores[best_language]["confidence"]
    best_text = scores[best_language]["text"].strip()

    if (
        raw_best_confidence < 0.60
        or (not best_text)
        or (
            confidence_margin < 0.005
            and validity.get(best_language, 0.0) < 0.55
        )
    ):
        best_language = "Uncertain"
        decision_reason = "low_margin_or_invalid_script"

    return {
        "selected_language": best_language,
        "selected_text": scores[best_language]["text"] if best_language in scores else "",
        "decision_reason": decision_reason,
        "scores": scores,
        "adjusted_scores": adjusted_scores,
        "validity": {
            "english_looks_valid": validity["English"] >= 0.7 and len(text_en.strip()) >= 2,
            "hindi_looks_valid": validity["Hindi"] >= 0.7,
            "telugu_looks_valid": validity["Telugu"] >= 0.7,
            "tamil_looks_valid": validity["Tamil"] >= 0.7,
        },
    }


def multilingual_predict(image):
    details = multilingual_predict_detailed(image)
    return details["selected_language"], details["selected_text"]


def multilingual_predict_detailed(image):
    text_hi, conf_hi = predict(hindi_model, hindi_encoder, image)
    text_te, conf_te = predict(telugu_model, telugu_encoder, image)
    text_ta, conf_ta = predict(tamil_model, tamil_encoder, image)
    text_en, conf_en = predict(english_model, english_encoder, image)

    scores = {
        "Hindi": {"text": text_hi, "confidence": conf_hi},
        "Telugu": {"text": text_te, "confidence": conf_te},
        "Tamil": {"text": text_ta, "confidence": conf_ta},
        "English": {"text": text_en, "confidence": conf_en},
    }
    return select_multilingual_result(scores)


if __name__ == "__main__":
    while True:
        img_path = input("\nEnter image path (or 'q' to quit): ").strip()

        if img_path.lower() == "q":
            break

        if not os.path.exists(img_path):
            print("Image not found.")
            continue

        image = preprocess(img_path)
        language, text = multilingual_predict(image)

        print("\nDetected language:", language)
        print("Prediction:", text)
