# generate_synth_telugu_lines.py

import os
import random
import csv

from PIL import Image, ImageDraw, ImageFont, ImageFilter

CORPUS_PATH = r"data\telugu_corpus.txt"  # you create this file with lots of Telugu lines
OUTPUT_DIR = r"data\telugu_lines"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
LABELS_CSV = os.path.join(OUTPUT_DIR, "labels_lines.csv")

# Add more fonts here as you download them
FONT_PATHS = [
    r"D:\multilingual_ocr\fonts\NotoSansTelugu-VariableFont_wdth,wght.ttf",
    r"D:\multilingual_ocr\fonts\NotoSansTelugu-Regular.ttf",
    # r"D:\multilingual_ocr\fonts\Gautami.ttf",  # example if you install
]

IMG_WIDTH = 1600   # base render size (will later be resized in training)
IMG_HEIGHT = 128   # one line per image

MAX_LINES = 5000   # limit for now, adjust as you like


def load_fonts():
    fonts = []
    for path in FONT_PATHS:
        if os.path.exists(path):
            # random size, but we’ll also vary later
            fonts.append(path)
        else:
            print(f"Warning: Font not found: {path}")
    if not fonts:
        raise RuntimeError("No usable fonts found in FONT_PATHS. Please fix paths.")
    return fonts


def render_line(text, font_path):
    """Render a single line of Telugu text to an RGB image."""
    # Random font size to add variety
    font_size = random.randint(28, 40)
    font = ImageFont.truetype(font_path, font_size)

    img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), "white")
    draw = ImageDraw.Draw(img)

    # Measure text size to center vertically, left pad a bit
    text_w, text_h = draw.textbbox((0, 0), text, font=font)[2:]
    x = random.randint(20, 60)
    y = max((IMG_HEIGHT - text_h) // 2, 0)

    draw.text((x, y), text, fill="black", font=font)

    # Small random augmentations
    if random.random() < 0.3:
        # slight blur
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.0)))

    if random.random() < 0.3:
        # very slight rotation
        angle = random.uniform(-2.0, 2.0)
        img = img.rotate(angle, expand=False, fillcolor="white")

    return img


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)

    fonts = load_fonts()
    print(f"Using {len(fonts)} font(s):")
    for f in fonts:
        print("  ", f)

    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(f"Corpus not found at {CORPUS_PATH}. Create a Telugu text file with one sentence/line per row.")

    lines = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    if not lines:
        raise RuntimeError("Corpus file is empty after stripping. Please add some Telugu text lines.")

    random.shuffle(lines)
    lines = lines[:MAX_LINES]
    print(f"Preparing to render {len(lines)} synthetic lines...")

    with open(LABELS_CSV, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "label"])

        for idx, text in enumerate(lines):
            font_path = random.choice(fonts)
            img = render_line(text, font_path)

            fname = f"line_{idx:05d}.png"
            out_path = os.path.join(IMAGES_DIR, fname)
            img.save(out_path)

            writer.writerow([fname, text])

            if (idx + 1) % 100 == 0:
                print(f"Rendered {idx+1}/{len(lines)} lines")

    print(f"Done. Images in: {IMAGES_DIR}")
    print(f"Labels CSV: {LABELS_CSV}")


if __name__ == "__main__":
    main()
