import os
import pandas as pd

DATA_DIR = r"D:\multilingual_ocr\data\handwritten"
CSV_PATH = os.path.join(DATA_DIR, "labels_handwritten.csv")
OUT_CSV = os.path.join(DATA_DIR, "labels_handwritten_fixed.csv")

def main():
    df = pd.read_csv(CSV_PATH, encoding="utf-8")

    fixed = 0
    missing = []

    for i, row in df.iterrows():
        fname = row["filename"]
        path = os.path.join(DATA_DIR, fname)

        # if file exists, keep it
        if os.path.exists(path):
            continue

        # if it's .jpg but missing, try .png
        if fname.lower().endswith(".jpg"):
            alt = fname[:-4] + ".png"
            alt_path = os.path.join(DATA_DIR, alt)
            if os.path.exists(alt_path):
                df.at[i, "filename"] = alt
                fixed += 1
            else:
                missing.append(fname)
        else:
            missing.append(fname)

    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"Fixed {fixed} filenames.")
    if missing:
        print("Still missing these files:")
        for m in missing:
            print("  ", m)
    print("Saved:", OUT_CSV)

if __name__ == "__main__":
    main()
