import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import os


class HindiDataset(Dataset):
    def __init__(self, data_dir, csv_path, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.df = pd.read_csv(csv_path)

        has_absolute_format = {"path", "text"}.issubset(self.df.columns)
        has_filename_format = {"filename", "label"}.issubset(self.df.columns)

        if not has_absolute_format and not has_filename_format:
            raise ValueError(
                "CSV must contain either {'path', 'text'} or "
                f"{{'filename', 'label'}}, found {self.df.columns}"
            )

        self.uses_absolute_paths = has_absolute_format

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.uses_absolute_paths:
            img_path = str(row["path"]).strip()
            label = str(row["text"])
        else:
            img_path = os.path.join(self.data_dir, str(row["filename"]).strip())
            label = str(row["label"])

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": label
        }
