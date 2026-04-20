# generate_synth_telugu_lines.py

import os
import random
import csv

from PIL import Image, ImageDraw, ImageFont, ImageFilter

CORPUS_PATH = r"data\telugu_corpus.txt"
OUTPUT_DIR = r"data\telugu_lines"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
LABELS_CSV = os.path.join(OUTPUT_DIR, "labels_lines.csv")

FONT_PATHS = [
    r"D:\multilingual_ocr\fonts\NotoSansTelugu-VariableFont_wdth,wght.ttf",
    # add more fonts here if you have them
]

IMG_WIDTH = 1600
IMG_HEIGHT = 128

# 🔥 KEY PARAM: how many variations per text line
VARIANTS_PER_LINE = 30   # 30x your corpus size


def load_fonts():
    fonts = []
    for path in FONT_PATHS:
        if os.path.exists(path):
            fonts.append(path)
        else:
            print(f"Warning: Font not found: {path}")
    if not fonts:
        raise RuntimeError("No usable fonts found in FONT_PATHS. Please fix paths.")
    return fonts


def render_line(text, font_path):
    font_size = random.randint(28, 40)
    font = ImageFont.truetype(font_path, font_size)

    img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), "white")
    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = random.randint(20, 60)
    y = max((IMG_HEIGHT - text_h) // 2, 0)

    draw.text((x, y), text, fill="black", font=font)

    # simple augmentations
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.0)))

    if random.random() < 0.3:
        angle = random.uniform(-2.0, 2.0)
        img = img.rotate(angle, expand=False, fillcolor="white")

    return img


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # 🔄 delete old images so we know the new count is correct
    for f in os.listdir(IMAGES_DIR):
        if f.lower().endswith(".png"):
            os.remove(os.path.join(IMAGES_DIR, f))

    fonts = load_fonts()
    print(f"Using {len(fonts)} font(s):")
    for f in fonts:
        print("  ", f)

    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(f"Corpus not found at {CORPUS_PATH}")

    base_lines = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                base_lines.append(line)

    if not base_lines:
        raise RuntimeError("Corpus file is empty. Add some Telugu lines.")

    print(f"Loaded {len(base_lines)} unique corpus lines.")
    total_samples = len(base_lines) * VARIANTS_PER_LINE
    print(f"Will generate ~{total_samples} synthetic line images.")

    with open(LABELS_CSV, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "label"])

        idx = 0
        for text in base_lines:
            for v in range(VARIANTS_PER_LINE):
                font_path = random.choice(fonts)
                img = render_line(text, font_path)

                fname = f"line_{idx:05d}.png"
                out_path = os.path.join(IMAGES_DIR, fname)
                img.save(out_path)

                writer.writerow([fname, text])
                idx += 1

                if idx % 100 == 0:
                    print(f"Rendered {idx}/{total_samples} lines")

    print(f"Done. Generated {idx} images.")
    print(f"Images in: {IMAGES_DIR}")
    print(f"Labels CSV: {LABELS_CSV}")


if __name__ == "__main__":
    main()
