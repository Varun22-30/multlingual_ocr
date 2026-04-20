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

from src.datasets.english_line_dataset import EnglishLineDataset
from src.models.vit_bilstm_ctc import ViTBILSTMCTC
from src.utils.image_transforms import ResizeAndPad
from src.utils.text_utils import TextEncoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

DATA_DIR = os.path.join("data", "english", "iam_lines", "images")
LABELS_PATH = os.path.join("data", "english", "iam_lines", "labels.txt")
BASE_CKPT = os.path.join("output", "models", "english", "vit_lstm_en_best.pth")
OUTPUT_CKPT = os.path.join("output", "models", "english", "vit_lstm_en_lines_best.pth")

EPOCHS = 12
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
PATIENCE = 3
NUM_WORKERS = 0
VAL_SPLIT = 0.1
SEED = 42
FREEZE_BACKBONE_EPOCHS = 2


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
    images = torch.stack([item["image"] for item in batch])
    labels = [item["label"] for item in batch]
    return images, labels


def set_backbone_trainable(model, trainable):
    if not hasattr(model, "vit"):
        return
    for param in model.vit.parameters():
        param.requires_grad = trainable


def run_epoch(model, loader, encoder, criterion, optimizer, device, is_train):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    steps = 0

    for images, labels in tqdm(loader, desc="Train" if is_train else "Val"):
        images = images.to(device, non_blocking=True)

        targets_list = [
            torch.tensor(encoder.encode(label), dtype=torch.long)
            for label in labels
        ]
        targets = torch.cat(targets_list).to(device)
        target_lengths = torch.tensor(
            [len(t) for t in targets_list],
            dtype=torch.long,
            device=device,
        )

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            log_probs = nn.functional.log_softmax(logits, dim=2)

            timesteps, batch_size, _ = log_probs.size()
            input_lengths = torch.full(
                (batch_size,),
                timesteps,
                dtype=torch.long,
                device=device,
            )

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(1, steps)


if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Line images dir not found: {DATA_DIR}")
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"Line labels file not found: {LABELS_PATH}")
    if not os.path.exists(BASE_CKPT):
        raise FileNotFoundError(f"Base English checkpoint not found: {BASE_CKPT}")

    os.makedirs(os.path.dirname(OUTPUT_CKPT), exist_ok=True)

    full_train_dataset = EnglishLineDataset(DATA_DIR, LABELS_PATH, train_transform)
    full_eval_dataset = EnglishLineDataset(DATA_DIR, LABELS_PATH, eval_transform)

    n_total = len(full_train_dataset)
    n_val = max(1, int(n_total * VAL_SPLIT))
    n_train = n_total - n_val
    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(SEED)).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_eval_dataset, val_indices)

    pin_memory = DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    print(f"English line samples: total={n_total}, train={n_train}, val={n_val}")

    encoder = TextEncoder(lang_code="en")
    model = ViTBILSTMCTC(num_classes=encoder.vocab_size()).to(DEVICE)
    model.load_state_dict(torch.load(BASE_CKPT, map_location=DEVICE, weights_only=True))

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    best_val_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        freeze_backbone = epoch <= FREEZE_BACKBONE_EPOCHS
        set_backbone_trainable(model, not freeze_backbone)

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        phase = "frozen-backbone" if freeze_backbone else "full-finetune"
        print(f"\nEpoch {epoch}/{EPOCHS} | {phase}")

        train_loss = run_epoch(model, train_loader, encoder, criterion, optimizer, DEVICE, True)
        val_loss = run_epoch(model, val_loader, encoder, criterion, optimizer, DEVICE, False)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(best_state_dict, OUTPUT_CKPT)
            epochs_without_improvement = 0
            print(f"Saved improved line model to {OUTPUT_CKPT}")
        else:
            epochs_without_improvement += 1
            print(f"No validation improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping triggered.")
            break

    if best_state_dict is None:
        raise RuntimeError("Training finished without saving a checkpoint.")

    print(f"Best val loss: {best_val_loss:.4f}")
