import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class TamilDataset(Dataset):

    def __init__(self, data_dir, csv_path, transform=None):

        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img_name = os.path.basename(row["filename"])
        img_path = os.path.join(self.data_dir, img_name)
        label = str(row["label"])

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": label
        }