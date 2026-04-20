# src/training/train_lines_ocr.py
# Line-level OCR training, starting from your good exam-handwritten model.

import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# --- add project root to path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.datasets.telugu_line_dataset import TeluguLineDataset
from src.models.vit_bilstm_ctc import ViTBILSTMCTC
from src.utils.text_utils import TextEncoder


PRETRAINED_PATH = os.path.join(
    project_root,
    "output",
    "checkpoints",
    "vit_lstm_te_examhand.pth",   # your good handwritten checkpoint
)

LANG_CODE = "te"
DATA_ROOT = os.path.join("data", "telugu_lines")
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
LABELS_CSV = os.path.join(DATA_ROOT, "labels_lines.csv")

NUM_EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
NUM_WORKERS = 4


def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    labels = [item["label"] for item in batch]
    return images, labels


def run_epoch(model, dataloader, text_encoder, criterion, optimizer, device, is_train=True):
    if is_train:
        model.train()
        desc = "Train"
    else:
        model.eval()
        desc = "Val"

    total_loss = 0.0
    progress = tqdm(dataloader, desc=desc)

    for images, labels in progress:
        images = images.to(device, non_blocking=True)

        # Encode labels
        targets_list = [torch.tensor(text_encoder.encode(lbl), dtype=torch.long) for lbl in labels]
        targets = torch.cat(targets_list)
        target_lengths = torch.tensor([len(t) for t in targets_list], dtype=torch.long, device=device)
        targets = targets.to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(images)  # (T, N, C)
        log_probs = nn.functional.log_softmax(logits, dim=2)

        T = log_probs.size(0)
        N = log_probs.size(1)
        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long, device=device)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    if not os.path.exists(IMAGES_DIR):
        raise FileNotFoundError(f"Line images dir not found: {IMAGES_DIR}")
    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError(f"Line labels CSV not found: {LABELS_CSV}")

    os.makedirs("output/checkpoints", exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),   # same as your other trainings
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    full_dataset = TeluguLineDataset(
        data_dir=IMAGES_DIR,
        csv_path=LABELS_CSV,
        transform=transform,
    )

    n_total = len(full_dataset)
    n_val = max(int(0.1 * n_total), 1)
    n_train = n_total - n_val
    print(f"Total line samples: {n_total} (train={n_train}, val={n_val})")

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    pin_memory = device.type == "cuda"

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

    text_encoder = TextEncoder(lang_code=LANG_CODE)
    vocab_size = text_encoder.vocab_size()
    print("Vocab size:", vocab_size)

    # --- build model ---
    model = ViTBILSTMCTC(num_classes=vocab_size)

    # --- load pretrained weights from exam-handwritten model ---
    if os.path.exists(PRETRAINED_PATH):
        print(f"Loading pretrained weights from: {PRETRAINED_PATH}")
        state_dict = torch.load(PRETRAINED_PATH, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
    else:
        print(f"⚠ Pretrained model not found at {PRETRAINED_PATH}, training from scratch.")

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    ckpt_path = "output/checkpoints/vit_lstm_te_lines_synth.pth"

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===")

        train_loss = run_epoch(model, train_loader, text_encoder, criterion, optimizer, device, is_train=True)
        val_loss = run_epoch(model, val_loader, text_encoder, criterion, optimizer=None, device=device, is_train=False)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(model_to_save.state_dict(), ckpt_path)
            print(f"✅ Improved val loss, saved to {ckpt_path}")
