import os
from typing import List

from PIL import Image
from torch.utils.data import Dataset


class CVLWordDataset(Dataset):
    def __init__(self, words_root: str, transform=None, allowed_chars: str | None = None):
        self.words_root = words_root
        self.transform = transform
        self.allowed_chars = set(allowed_chars) if allowed_chars is not None else None
        self.samples: List[tuple[str, str]] = []
        self.skipped_unsupported = 0

        if not os.path.exists(words_root):
            raise FileNotFoundError(f"CVL words root not found: {words_root}")

        for writer_id in sorted(os.listdir(words_root)):
            writer_dir = os.path.join(words_root, writer_id)
            if not os.path.isdir(writer_dir):
                continue

            for filename in sorted(os.listdir(writer_dir)):
                if not filename.lower().endswith(".tif"):
                    continue

                label = self.label_from_filename(filename)
                if not label:
                    continue

                if self.allowed_chars is not None and any(ch not in self.allowed_chars for ch in label):
                    self.skipped_unsupported += 1
                    continue

                self.samples.append((os.path.join(writer_dir, filename), label))

        if not self.samples:
            raise RuntimeError(f"No usable CVL word samples found in {words_root}")

    @staticmethod
    def label_from_filename(filename: str) -> str:
        stem = os.path.splitext(filename)[0]
        parts = stem.split("-", 4)
        if len(parts) < 5:
            return ""
        return parts[-1].strip()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
        }
