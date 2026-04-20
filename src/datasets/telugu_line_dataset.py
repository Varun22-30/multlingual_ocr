# src/datasets/telugu_line_dataset.py

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class TeluguLineDataset(Dataset):
    """
    Dataset for line-level Telugu OCR.
    CSV: filename,label
    data_dir: directory containing images (e.g., data/telugu_lines/images)
    """
    def __init__(self, data_dir: str, csv_path: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]
        fname = row["filename"]
        label = row["label"]

        img_path = os.path.join(self.data_dir, fname)
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: missing image {img_path}, skipping to next.")
            return self[(idx + 1) % len(self)]

        if self.transform:
            image = self.transform(image)

        return {"image": image, "label": label}
