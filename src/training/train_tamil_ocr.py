import copy
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.datasets.tamil_dataset import TamilDataset
from src.models.vit_bilstm_ctc import ViTBILSTMCTC
from src.utils.image_transforms import ResizeAndPad
from src.utils.text_utils import TextEncoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

TRAIN_DIR = "data/tamil/train"
VAL_DIR = "data/tamil/val"
TEST_DIR = "data/tamil/test"

TRAIN_CSV = "data/tamil/train/labels_train.csv"
VAL_CSV = "data/tamil/val/labels_val.csv"
TEST_CSV = "data/tamil/test/labels_test.csv"

OUTPUT_CKPT = "output/models/tamil/vit_lstm_ta_best.pth"

EPOCHS = 20
BATCH_SIZE = 16
LR = 5e-5
WEIGHT_DECAY = 1e-4
PATIENCE = 5
NUM_WORKERS = 0


train_transform = transforms.Compose([
    ResizeAndPad((224, 224)),
    transforms.RandomRotation(2),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.02, 0.02),
        shear=2,
    ),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

eval_transform = transforms.Compose([
    ResizeAndPad((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    labels = [b["label"] for b in batch]
    return images, labels


train_dataset = TamilDataset(TRAIN_DIR, TRAIN_CSV, train_transform)
val_dataset = TamilDataset(VAL_DIR, VAL_CSV, eval_transform)
test_dataset = TamilDataset(TEST_DIR, TEST_CSV, eval_transform)

pin_memory = DEVICE.type == "cuda"

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory,
)


encoder = TextEncoder(lang_code="ta")
vocab_size = encoder.vocab_size()
model = ViTBILSTMCTC(num_classes=vocab_size).to(DEVICE)

criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


def run_epoch(loader, train):
    model.train() if train else model.eval()
    total_loss = 0.0
    steps = 0

    for images, labels in tqdm(loader, desc="Train" if train else "Val/Test"):
        images = images.to(DEVICE, non_blocking=True)

        targets_list = [
            torch.tensor(encoder.encode(label), dtype=torch.long)
            for label in labels
        ]
        targets = torch.cat(targets_list).to(DEVICE)
        target_lengths = torch.tensor(
            [len(t) for t in targets_list],
            dtype=torch.long,
            device=DEVICE,
        )

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(images)
            log_probs = nn.functional.log_softmax(logits, dim=2)

            timesteps, batch_size, _ = log_probs.size()
            input_lengths = torch.full(
                (batch_size,),
                timesteps,
                dtype=torch.long,
                device=DEVICE,
            )

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(1, steps)


os.makedirs("output/models/tamil", exist_ok=True)

best_val = float("inf")
best_state = None
epochs_without_improvement = 0

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")

    train_loss = run_epoch(train_loader, True)
    val_loss = run_epoch(val_loader, False)

    print("Train Loss:", train_loss)
    print("Val Loss:", val_loss)

    if val_loss < best_val:
        best_val = val_loss
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
        torch.save(best_state, OUTPUT_CKPT)
        print("Tamil model saved")
    else:
        epochs_without_improvement += 1
        print(f"No validation improvement for {epochs_without_improvement} epoch(s)")

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping triggered.")
        break

if best_state is not None:
    model.load_state_dict(best_state)

print("\nRunning final test evaluation...")
test_loss = run_epoch(test_loader, False)
print("Test Loss:", test_loss)
