import copy
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.datasets.hindi_dataset import HindiDataset
from src.models.vit_bilstm_ctc import ViTBILSTMCTC
from src.utils.image_transforms import ResizeAndPad
from src.utils.text_utils import TextEncoder


LANG_CODE = "hi"
HANDWRITTEN_DIR = os.path.join(project_root, "data", "handwritten_hindi")
HANDWRITTEN_CSV = os.path.join(HANDWRITTEN_DIR, "labels.csv")

BASE_MODEL_PATH = os.path.join(
    project_root,
    "output",
    "models",
    "hindi",
    "vit_lstm_hi_handwritten_finetuned.pth",
)

OUTPUT_CKPT = os.path.join(
    project_root,
    "output",
    "models",
    "hindi",
    "vit_lstm_hi_final.pth",
)

BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2
NUM_WORKERS = 4
SEED = 42
FREEZE_BACKBONE_EPOCHS = 2
PATIENCE = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    ResizeAndPad((224, 224)),
    transforms.RandomRotation(3),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.03, 0.03),
        shear=3,
    ),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
    ),
    transforms.GaussianBlur(
        kernel_size=3,
        sigma=(0.1, 1.0),
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


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    labels = [b["label"] for b in batch]
    return images, labels


def make_dataloaders():
    train_dataset = HindiDataset(
        data_dir=HANDWRITTEN_DIR,
        csv_path=HANDWRITTEN_CSV,
        transform=train_transform,
    )
    val_dataset = HindiDataset(
        data_dir=HANDWRITTEN_DIR,
        csv_path=HANDWRITTEN_CSV,
        transform=val_transform,
    )

    n_total = len(train_dataset)
    n_val = int(n_total * VAL_SPLIT)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(n_total, generator=generator).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    pin_memory = DEVICE.type == "cuda"

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    print(f"Handwritten Hindi samples: total={n_total}, train={n_train}, val={n_val}")
    return train_loader, val_loader


def set_backbone_trainable(model, trainable):
    if not hasattr(model, "vit"):
        return
    for param in model.vit.parameters():
        param.requires_grad = trainable


def run_epoch(model, loader, encoder, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    total_loss = 0.0
    steps = 0

    for images, labels in tqdm(loader, desc="Train" if train else "Val"):
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
                device=device,
            )

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(1, steps)


if __name__ == "__main__":
    torch.manual_seed(SEED)
    os.makedirs(os.path.dirname(OUTPUT_CKPT), exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Base checkpoint: {BASE_MODEL_PATH}")
    print(f"Output checkpoint: {OUTPUT_CKPT}")

    if not os.path.exists(BASE_MODEL_PATH):
        raise FileNotFoundError(f"Base model not found: {BASE_MODEL_PATH}")

    encoder = TextEncoder(lang_code=LANG_CODE)
    model = ViTBILSTMCTC(num_classes=encoder.vocab_size())
    model.load_state_dict(
        torch.load(BASE_MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    model = model.to(DEVICE)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    train_loader, val_loader = make_dataloaders()

    best_val_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        freeze_backbone = epoch <= FREEZE_BACKBONE_EPOCHS
        set_backbone_trainable(model, not freeze_backbone)

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        phase = "frozen-backbone" if freeze_backbone else "full-finetune"
        print(f"\nEpoch {epoch}/{NUM_EPOCHS} | {phase}")

        train_loss = run_epoch(
            model,
            train_loader,
            encoder,
            criterion,
            optimizer,
            DEVICE,
            train=True,
        )
        val_loss = run_epoch(
            model,
            val_loader,
            encoder,
            criterion,
            optimizer,
            DEVICE,
            train=False,
        )

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(best_state_dict, OUTPUT_CKPT)
            print(f"Saved improved checkpoint to {OUTPUT_CKPT}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"No validation improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping triggered.")
            break

    if best_state_dict is None:
        raise RuntimeError("Training finished without saving a checkpoint.")

    print(f"Best val loss: {best_val_loss:.4f}")
