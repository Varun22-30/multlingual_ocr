import os
import cv2
import numpy as np

# ---------- CONFIG ----------
INPUT_IMAGE = r"C:\Users\sribh\Downloads\WhatsApp Image 2025-12-04 at 9.11.37 AM (1).jpeg"

OUT_DIR     = r"D:\multilingual_ocr\data\handwritten\crops_page1"
PREFIX      = "p1_word_"

MIN_AREA    = 2000   # a bit larger now since whole word
WORD_KERNEL_WIDTH = 35   # increase if it still splits a word
WORD_KERNEL_HEIGHT = 7   # small vertical height
# ----------------------------


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    img = cv2.imread(INPUT_IMAGE)
    if img is None:
        raise FileNotFoundError(f"Could not read {INPUT_IMAGE}")

    h, w = img.shape[:2]

    # 1) Crop to central notebook region
    y0 = int(0.05 * h)
    y1 = int(0.98 * h)
    x0 = int(0.10 * w)
    x1 = int(0.98 * w)
    roi = img[y0:y1, x0:x1]

    # 2) Grayscale + blur
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # 3) Threshold
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31, 15
    )

    # 4) First close small gaps in strokes
    kernel_small = np.ones((3, 3), np.uint8)
    th_closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_small, iterations=1)

    # 5) HORIZONTALLY DILATE to join letters into word blobs
    word_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (WORD_KERNEL_WIDTH, WORD_KERNEL_HEIGHT)
    )
    th_words = cv2.dilate(th_closed, word_kernel, iterations=1)

    # 6) Find contours on the dilated "word-level" mask
    contours, _ = cv2.findContours(th_words, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w_box, h_box = cv2.boundingRect(c)
        if w_box * h_box < MIN_AREA:
            continue
        boxes.append((x, y, w_box, h_box))

    # 7) Sort in reading order: top → bottom, left → right
    boxes.sort(key=lambda b: (b[1], b[0]))
    print(f"Found {len(boxes)} candidate WORDS.")

    count = 1
    for (x, y, w_box, h_box) in boxes:
        pad = 5
        x0_c = max(0, x - pad)
        y0_c = max(0, y - pad)
        x1_c = min(roi.shape[1], x + w_box + pad)
        y1_c = min(roi.shape[0], y + h_box + pad)

        crop = roi[y0_c:y1_c, x0_c:x1_c]

        fname = f"{PREFIX}{count:03d}.jpg"
        out_path = os.path.join(OUT_DIR, fname)
        cv2.imwrite(out_path, crop)
        print("Saved:", out_path)
        count += 1

    print("Done. Word crops saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
