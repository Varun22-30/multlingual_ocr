import os
from torch.utils.data import Dataset
from PIL import Image


class EnglishLineDataset(Dataset):
    def __init__(self, data_dir, labels_path, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        with open(labels_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue

                filename, text = line.split(",", 1)
                self.samples.append((filename.strip(), text.strip()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        img_path = os.path.join(self.data_dir, filename)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
        }
