import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.inference.infer_multilingual import (
    DEVICE,
    english_encoder,
    english_model,
    hindi_encoder,
    hindi_model,
    predict,
    select_multilingual_result,
    tamil_encoder,
    tamil_model,
    telugu_encoder,
    telugu_model,
)
from src.models.vit_bilstm_ctc import ViTBILSTMCTC
from src.utils.image_transforms import ResizeAndPad
from src.utils.text_utils import TextEncoder


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int

    def to_list(self) -> List[int]:
        return [self.x, self.y, self.w, self.h]


def build_transform(lang_code: str):
    first_step = ResizeAndPad((224, 224)) if lang_code in {"hi", "ta"} else transforms.Resize((224, 224))
    return transforms.Compose([
        first_step,
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


TRANSFORMS = {
    "en": build_transform("en"),
    "hi": build_transform("hi"),
    "ta": build_transform("ta"),
    "te": build_transform("te"),
}

TELUGU_LINE_MODEL_PATH = os.path.join("output", "models", "telugu", "vit_lstm_te_lines_synth.pth")
TELUGU_LINE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
_TELUGU_LINE_MODEL = None
_TELUGU_LINE_ENCODER = None


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def preprocess_word_image(image: Image.Image, lang_code: str) -> torch.Tensor:
    return TRANSFORMS[lang_code](image).unsqueeze(0)


def preprocess_line_image(image: Image.Image) -> torch.Tensor:
    return TELUGU_LINE_TRANSFORM(image).unsqueeze(0)


def telugu_script_ratio(text: str) -> float:
    chars = [ch for ch in text.strip() if not ch.isspace()]
    if not chars:
        return 0.0
    return sum(0x0C00 <= ord(ch) <= 0x0C7F for ch in chars) / len(chars)


def score_ocr_candidate(text: str, confidence: float, script_ratio: float) -> float:
    length_bonus = min(len(text.strip()), 20) * 0.01
    return confidence + (0.20 * script_ratio) + length_bonus


def generate_telugu_variants(image: Image.Image) -> List[Image.Image]:
    bgr = pil_to_bgr(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    denoised = cv2.bilateralFilter(clahe, 7, 50, 50)
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    variants = [
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB),
    ]
    return [Image.fromarray(variant) for variant in variants]


def get_telugu_line_model():
    global _TELUGU_LINE_MODEL, _TELUGU_LINE_ENCODER
    if _TELUGU_LINE_MODEL is not None and _TELUGU_LINE_ENCODER is not None:
        return _TELUGU_LINE_MODEL, _TELUGU_LINE_ENCODER

    if not os.path.exists(TELUGU_LINE_MODEL_PATH):
        return None, None

    encoder = TextEncoder(lang_code="te")
    checkpoint = torch.load(TELUGU_LINE_MODEL_PATH, map_location=DEVICE, weights_only=True)
    num_classes = checkpoint["classifier.weight"].shape[0]
    model = ViTBILSTMCTC(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()

    _TELUGU_LINE_MODEL = model
    _TELUGU_LINE_ENCODER = encoder
    return _TELUGU_LINE_MODEL, _TELUGU_LINE_ENCODER


def ocr_single_language(image: Image.Image, language_name: str) -> Dict[str, object]:
    language_map = {
        "English": ("en", english_model, english_encoder),
        "Hindi": ("hi", hindi_model, hindi_encoder),
        "Tamil": ("ta", tamil_model, tamil_encoder),
        "Telugu": ("te", telugu_model, telugu_encoder),
    }
    lang_code, model, encoder = language_map[language_name]
    if language_name == "Telugu":
        best_text = ""
        best_confidence = -1.0
        best_score = -1.0
        for variant in generate_telugu_variants(image):
            text, confidence = predict(model, encoder, preprocess_word_image(variant, lang_code))
            script_ratio = telugu_script_ratio(text)
            candidate_score = score_ocr_candidate(text, confidence, script_ratio)
            if candidate_score > best_score:
                best_text = text
                best_confidence = confidence
                best_score = candidate_score
        text, confidence = best_text, best_confidence
    else:
        text, confidence = predict(model, encoder, preprocess_word_image(image, lang_code))
    return {
        "selected_language": language_name,
        "selected_text": text.strip(),
        "decision_reason": "page_language_prior",
        "scores": {
            language_name: {
                "text": text,
                "confidence": confidence,
            }
        },
    }


def ocr_telugu_line(image: Image.Image) -> Dict[str, object]:
    model, encoder = get_telugu_line_model()
    if model is None or encoder is None:
        return ocr_single_language(image, "Telugu")

    best_text = ""
    best_confidence = -1.0
    best_score = -1.0
    for variant in generate_telugu_variants(image):
        text, confidence = predict(model, encoder, preprocess_line_image(variant))
        script_ratio = telugu_script_ratio(text)
        candidate_score = score_ocr_candidate(text, confidence, script_ratio)
        if candidate_score > best_score:
            best_text = text
            best_confidence = confidence
            best_score = candidate_score

    return {
        "selected_language": "Telugu",
        "selected_text": best_text.strip(),
        "decision_reason": "telugu_line_model",
        "scores": {
            "Telugu": {
                "text": best_text,
                "confidence": best_confidence,
            }
        },
    }


def ocr_multilingual_word(image: Image.Image) -> Dict[str, object]:
    text_en, conf_en = predict(english_model, english_encoder, preprocess_word_image(image, "en"))
    text_hi, conf_hi = predict(hindi_model, hindi_encoder, preprocess_word_image(image, "hi"))
    text_ta, conf_ta = predict(tamil_model, tamil_encoder, preprocess_word_image(image, "ta"))
    text_te, conf_te = predict(telugu_model, telugu_encoder, preprocess_word_image(image, "te"))

    scores = {
        "English": {
            "text": text_en,
            "confidence": conf_en,
        },
        "Hindi": {
            "text": text_hi,
            "confidence": conf_hi,
        },
        "Tamil": {
            "text": text_ta,
            "confidence": conf_ta,
        },
        "Telugu": {
            "text": text_te,
            "confidence": conf_te,
        },
    }
    return select_multilingual_result(scores)


def binarize(gray: np.ndarray) -> np.ndarray:
    blur = cv2.bilateralFilter(gray, 7, 50, 50)
    return cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )


def remove_horizontal_lines(binary: np.ndarray) -> np.ndarray:
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(25, binary.shape[1] // 18), 1),
    )
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    cleaned = cv2.subtract(binary, horizontal_lines)
    return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)


def crop_box(image: np.ndarray, box: Box, pad: int = 4) -> np.ndarray:
    x0 = max(0, box.x - pad)
    y0 = max(0, box.y - pad)
    x1 = min(image.shape[1], box.x + box.w + pad)
    y1 = min(image.shape[0], box.y + box.h + pad)
    return image[y0:y1, x0:x1]


def content_box_from_binary(binary: np.ndarray, min_pixels: int = 10):
    ys, xs = np.where(binary > 0)
    if len(xs) < min_pixels or len(ys) < min_pixels:
        return None
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max())
    y1 = int(ys.max())
    return Box(x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def sort_reading_order(boxes: List[Box]) -> List[Box]:
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: (b.y, b.x))
    rows: List[List[Box]] = []
    y_threshold = max(10, int(np.median([b.h for b in boxes]) * 0.6))

    for box in boxes:
        placed = False
        for row in rows:
            row_y = int(np.mean([candidate.y for candidate in row]))
            if abs(box.y - row_y) <= y_threshold:
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])

    ordered: List[Box] = []
    for row in rows:
        row.sort(key=lambda b: b.x)
        ordered.extend(row)
    return ordered


def contours_to_boxes(mask: np.ndarray, min_area: int) -> List[Box]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Box] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < min_area:
            continue
        boxes.append(Box(x, y, w, h))
    return sort_reading_order(boxes)


def segment_paragraphs(image: np.ndarray) -> List[Box]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = remove_horizontal_lines(binarize(gray))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(35, image.shape[1] // 12), max(18, image.shape[0] // 40)),
    )
    merged = cv2.dilate(binary, kernel, iterations=2)
    min_area = max(4000, int(image.shape[0] * image.shape[1] * 0.01))
    return contours_to_boxes(merged, min_area=min_area)


def segment_lines(paragraph_image: np.ndarray) -> List[Box]:
    gray = cv2.cvtColor(paragraph_image, cv2.COLOR_BGR2GRAY)
    binary = remove_horizontal_lines(binarize(gray))
    projection = binary.sum(axis=1)
    threshold = max(int(projection.max() * 0.12), 1)
    active_rows = projection > threshold

    spans = []
    start = None
    for index, is_active in enumerate(active_rows):
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            spans.append((start, index - 1))
            start = None
    if start is not None:
        spans.append((start, len(active_rows) - 1))

    if not spans:
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(40, paragraph_image.shape[1] // 8), max(3, paragraph_image.shape[0] // 70)),
        )
        merged = cv2.dilate(closed, kernel, iterations=1)
        min_area = max(500, int(paragraph_image.shape[0] * paragraph_image.shape[1] * 0.002))
        return contours_to_boxes(merged, min_area=min_area)

    merged_spans = []
    gap_limit = max(6, paragraph_image.shape[0] // 80)
    for start, end in spans:
        if not merged_spans:
            merged_spans.append([start, end])
            continue
        if start - merged_spans[-1][1] <= gap_limit:
            merged_spans[-1][1] = end
        else:
            merged_spans.append([start, end])

    min_height = max(12, paragraph_image.shape[0] // 40)
    boxes: List[Box] = []
    for start, end in merged_spans:
        if end - start + 1 < min_height:
            continue
        line_mask = binary[start:end + 1, :]
        content_box = content_box_from_binary(line_mask)
        if content_box is None:
            continue
        boxes.append(Box(content_box.x, start + content_box.y, content_box.w, content_box.h))

    return sort_reading_order(boxes)


def segment_words(line_image: np.ndarray) -> List[Box]:
    gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    binary = remove_horizontal_lines(binarize(gray))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(10, line_image.shape[1] // 30), max(3, line_image.shape[0] // 5)),
    )
    merged = cv2.dilate(closed, kernel, iterations=1)
    min_area = max(120, int(line_image.shape[0] * line_image.shape[1] * 0.0015))
    return contours_to_boxes(merged, min_area=min_area)


def save_crop(image: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)


def draw_boxes(image: np.ndarray, boxes: List[Box], color: tuple, label_prefix: str = "") -> np.ndarray:
    canvas = image.copy()
    for index, box in enumerate(boxes, start=1):
        cv2.rectangle(canvas, (box.x, box.y), (box.x + box.w, box.y + box.h), color, 2)
        if label_prefix:
            cv2.putText(
                canvas,
                f"{label_prefix}{index}",
                (box.x, max(18, box.y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    return canvas


def infer_page_language(line_crops: List[np.ndarray]) -> str:
    votes = {"English": 0, "Hindi": 0, "Tamil": 0, "Telugu": 0}

    for line_crop in line_crops[:4]:
        word_boxes = segment_words(line_crop)
        candidates = word_boxes[:3] if word_boxes else [Box(0, 0, line_crop.shape[1], line_crop.shape[0])]

        for candidate in candidates:
            word_crop = crop_box(line_crop, candidate, pad=4)
            word_image = Image.fromarray(cv2.cvtColor(word_crop, cv2.COLOR_BGR2RGB))
            result = ocr_multilingual_word(word_image)
            language = result["selected_language"]
            if language in votes:
                votes[language] += 1

    top_language = max(votes, key=votes.get)
    return top_language if votes[top_language] > 0 else "Telugu"


def word_result_to_text(word_results: List[Dict[str, object]]) -> str:
    return " ".join(result["text"] for result in word_results if result["text"])


def process_document(
    image_path: str,
    output_dir: str,
    save_crops: bool = True,
    forced_language: str = "auto",
    mode: str = "auto",
) -> Dict[str, object]:
    page = cv2.imread(image_path)
    if page is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    os.makedirs(output_dir, exist_ok=True)
    page_name = os.path.splitext(os.path.basename(image_path))[0]

    paragraphs = []
    if mode == "handwritten":
        paragraph_boxes = [Box(0, 0, page.shape[1], page.shape[0])]
    else:
        paragraph_boxes = segment_paragraphs(page)
        if not paragraph_boxes:
            paragraph_boxes = [Box(0, 0, page.shape[1], page.shape[0])]

    sample_lines: List[np.ndarray] = []
    for paragraph_box in paragraph_boxes[:2]:
        paragraph_crop = crop_box(page, paragraph_box, pad=8)
        sample_lines.extend(crop_box(paragraph_crop, line_box, pad=6) for line_box in segment_lines(paragraph_crop)[:3])

    page_language = infer_page_language(sample_lines) if sample_lines else "Telugu"
    if forced_language and forced_language.lower() != "auto":
        page_language = forced_language.title()
    elif mode == "handwritten":
        # Handwritten pages in this project are overwhelmingly Telugu-first.
        page_language = "Telugu"

    page_overlay = draw_boxes(page, paragraph_boxes, (0, 180, 255), "P")
    save_crop(page_overlay, os.path.join(output_dir, f"{page_name}_paragraph_boxes.png"))

    for paragraph_index, paragraph_box in enumerate(paragraph_boxes, start=1):
        paragraph_crop = crop_box(page, paragraph_box, pad=8)
        paragraph_dir = os.path.join(output_dir, f"{page_name}_paragraph_{paragraph_index:02d}")

        if save_crops:
            save_crop(paragraph_crop, os.path.join(paragraph_dir, "paragraph.png"))

        lines = []
        line_boxes = segment_lines(paragraph_crop)
        if not line_boxes:
            line_boxes = [Box(0, 0, paragraph_crop.shape[1], paragraph_crop.shape[0])]

        if save_crops:
            line_overlay = draw_boxes(paragraph_crop, line_boxes, (0, 255, 0), "L")
            save_crop(line_overlay, os.path.join(paragraph_dir, "line_boxes.png"))

        for line_index, line_box in enumerate(line_boxes, start=1):
            line_crop = crop_box(paragraph_crop, line_box, pad=6)
            line_image = Image.fromarray(cv2.cvtColor(line_crop, cv2.COLOR_BGR2RGB))

            if save_crops:
                save_crop(line_crop, os.path.join(paragraph_dir, f"line_{line_index:02d}.png"))

            word_results = []
            line_result = ocr_telugu_line(line_image) if page_language == "Telugu" else None
            should_run_word_fallback = (
                page_language != "Telugu"
                or line_result is None
                or not str(line_result["selected_text"]).strip()
            )

            if should_run_word_fallback:
                word_boxes = segment_words(line_crop)
                if save_crops and word_boxes:
                    word_overlay = draw_boxes(line_crop, word_boxes, (255, 120, 0), "W")
                    save_crop(word_overlay, os.path.join(paragraph_dir, f"line_{line_index:02d}_word_boxes.png"))

                for word_index, word_box in enumerate(word_boxes, start=1):
                    word_crop = crop_box(line_crop, word_box, pad=4)
                    word_image = Image.fromarray(cv2.cvtColor(word_crop, cv2.COLOR_BGR2RGB))
                    if page_language in {"English", "Hindi", "Tamil", "Telugu"}:
                        ocr_result = ocr_single_language(word_image, page_language)
                    else:
                        ocr_result = ocr_multilingual_word(word_image)
                    word_text = str(ocr_result["selected_text"]).strip()

                    if save_crops:
                        save_crop(
                            word_crop,
                            os.path.join(paragraph_dir, f"line_{line_index:02d}_word_{word_index:02d}.png"),
                        )

                    word_results.append({
                        "word_index": word_index,
                        "box": word_box.to_list(),
                        "language": ocr_result["selected_language"],
                        "text": word_text,
                        "decision_reason": ocr_result["decision_reason"],
                        "scores": ocr_result["scores"],
                    })

            line_text = word_result_to_text(word_results)
            if line_result is not None and line_result["selected_text"]:
                line_text = str(line_result["selected_text"]).strip()
            lines.append({
                "line_index": line_index,
                "box": line_box.to_list(),
                "language": page_language,
                "decision_reason": line_result["decision_reason"] if line_result is not None else "word_aggregation",
                "text": line_text,
                "words": word_results,
            })

        paragraph_text = "\n".join(line["text"] for line in lines if line["text"])
        paragraphs.append({
            "paragraph_index": paragraph_index,
            "box": paragraph_box.to_list(),
            "text": paragraph_text,
            "lines": lines,
        })

    document_text = "\n\n".join(paragraph["text"] for paragraph in paragraphs if paragraph["text"])
    result = {
        "image_path": image_path,
        "output_dir": output_dir,
        "mode": mode,
        "page_language": page_language,
        "models": {
            "telugu_word": getattr(telugu_model, "__class__", type(telugu_model)).__name__,
            "telugu_line_path": TELUGU_LINE_MODEL_PATH,
        },
        "paragraph_count": len(paragraphs),
        "text": document_text,
        "paragraphs": paragraphs,
    }

    with open(os.path.join(output_dir, f"{page_name}_ocr.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, f"{page_name}_ocr.txt"), "w", encoding="utf-8") as handle:
        handle.write(document_text)

    with open(os.path.join(output_dir, f"{page_name}_debug_summary.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"mode: {mode}\n")
        handle.write(f"page_language: {page_language}\n")
        handle.write(f"paragraph_count: {len(paragraphs)}\n")
        for paragraph in paragraphs:
            handle.write(f"\nparagraph_{paragraph['paragraph_index']:02d}: {paragraph['text']}\n")
            for line in paragraph["lines"]:
                handle.write(
                    f"  line_{line['line_index']:02d} [{line['decision_reason']}]: {line['text']}\n"
                )

    return result


def normalize_input_path(raw_path: str) -> str:
    return raw_path.strip().strip("\"'")


def main():
    parser = argparse.ArgumentParser(description="Segment a page into paragraphs, lines, and words, then run multilingual OCR.")
    parser.add_argument("image_path", nargs="?", help="Path to the input page image")
    parser.add_argument(
        "--output-dir",
        default=os.path.join("output", "document_ocr"),
        help="Directory to store crops and OCR results",
    )
    parser.add_argument(
        "--no-save-crops",
        action="store_true",
        help="Skip saving paragraph, line, and word crops",
    )
    parser.add_argument(
        "--lang",
        default="auto",
        choices=["auto", "telugu", "hindi", "tamil", "english"],
        help="Force a page language instead of auto-routing",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "handwritten", "printed"],
        help="Choose handwritten or printed segmentation behavior",
    )
    args = parser.parse_args()

    image_path = args.image_path
    if not image_path:
        image_path = input("Enter image path: ")

    image_path = normalize_input_path(image_path)

    if not image_path:
        print("No image path provided.")
        sys.exit(1)

    result = process_document(
        image_path=image_path,
        output_dir=args.output_dir,
        save_crops=not args.no_save_crops,
        forced_language=args.lang,
        mode=args.mode,
    )
    print(f"Detected page language: {result['page_language']}")
    print(f"Processing mode: {result['mode']}")
    print(f"Processed {result['paragraph_count']} paragraph(s)")
    print("Detected text:")
    print(result["text"])


if __name__ == "__main__":
    main()
