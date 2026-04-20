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

from src.datasets.hindi_dataset import HindiDataset
from src.models.vit_bilstm_ctc import ViTBILSTMCTC
from src.utils.image_transforms import ResizeAndPad
from src.utils.text_utils import TextEncoder


LANG_CODE = "hi"
DATA_DIR = "data/hindi"
TRAIN_CSV = os.path.join(DATA_DIR, "train", "labels_train.csv")
VAL_CSV = os.path.join(DATA_DIR, "val", "labels_val.csv")
OUTPUT_CKPT = "output/models/hindi/vit_lstm_hi_best.pth"

NUM_EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
PATIENCE = 5


def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    labels = [item["label"] for item in batch]
    return images, labels 


def run_epoch(model, dataloader, text_encoder, criterion, optimizer, device, is_train):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    steps = 0
    desc = "Train" if is_train else "Val"
    progress_bar = tqdm(dataloader, desc=desc)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)

        targets_list = [
            torch.tensor(text_encoder.encode(label), dtype=torch.long)
            for label in labels
        ]
        targets = torch.cat(targets_list).to(device)
        target_lengths = torch.tensor(
            [len(target) for target in targets_list],
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
                size=(batch_size,),
                fill_value=timesteps,
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
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / max(1, steps)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        torch.backends.cudnn.benchmark = True

    os.makedirs(os.path.dirname(OUTPUT_CKPT), exist_ok=True)

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

    val_transform = transforms.Compose([
        ResizeAndPad((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = HindiDataset(
        data_dir=os.path.join(DATA_DIR, "train"),
        csv_path=TRAIN_CSV,
        transform=train_transform,
    )
    val_dataset = HindiDataset(
        data_dir=os.path.join(DATA_DIR, "val"),
        csv_path=VAL_CSV,
        transform=val_transform,
    )

    pin_memory = device.type == "cuda"

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

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size:   {len(val_dataset)}")

    text_encoder = TextEncoder(lang_code=LANG_CODE)
    vocab_size = text_encoder.vocab_size()
    model = ViTBILSTMCTC(num_classes=vocab_size)

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print("Training Hindi model from scratch.")

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---")

        train_loss = run_epoch(
            model=model,
            dataloader=train_loader,
            text_encoder=text_encoder,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            is_train=True,
        )
        val_loss = run_epoch(
            model=model,
            dataloader=val_loader,
            text_encoder=text_encoder,
            criterion=criterion,
            optimizer=None,
            device=device,
            is_train=False,
        )

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(model_to_save.state_dict(), OUTPUT_CKPT)
            print(f"Validation loss improved. Model saved to {OUTPUT_CKPT}")
        else:
            epochs_without_improvement += 1
            print(f"No validation improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping triggered.")
            break
