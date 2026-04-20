import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class EnglishDataset(Dataset):

    def __init__(self, data_dir, csv_path, transform=None):

        self.data_dir = data_dir
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img_path = os.path.join(self.data_dir, row["filename"])

        label = str(row["label"])

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": label
        }