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

from src.datasets.english_dataset import EnglishDataset
from src.models.vit_bilstm_ctc import ViTBILSTMCTC
from src.utils.text_utils import TextEncoder


LANG_CODE = "en"

TRAIN_DIR = "data/english/labeled_images"
VAL_DIR = "data/english/labeled_images"
TEST_DIR = "data/english/labeled_images"

TRAIN_CSV = "data/english/labels_train.csv"
VAL_CSV = "data/english/labels_val.csv"
TEST_CSV = "data/english/labels_test.csv"

PRETRAINED_MODEL = "output/models/english/vit_lstm_en_best.pth"

CONTINUE_EPOCHS = 8
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
PATIENCE = 3
FREEZE_BACKBONE_EPOCHS = 2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
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
    transforms.Resize((224, 224)),
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


train_dataset = EnglishDataset(TRAIN_DIR, TRAIN_CSV, train_transform)
val_dataset = EnglishDataset(VAL_DIR, VAL_CSV, val_transform)
test_dataset = EnglishDataset(TEST_DIR, TEST_CSV, val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
)


text_encoder = TextEncoder(lang_code=LANG_CODE)
vocab_size = text_encoder.vocab_size()
model = ViTBILSTMCTC(num_classes=vocab_size).to(device)

if os.path.exists(PRETRAINED_MODEL):
    print("Loading existing English OCR model...")
    state_dict = torch.load(PRETRAINED_MODEL, map_location=device)
    model.load_state_dict(state_dict)
    print("Model loaded successfully")


def set_backbone_trainable(model, trainable):
    if not hasattr(model, "vit"):
        return
    for param in model.vit.parameters():
        param.requires_grad = trainable


criterion = nn.CTCLoss(blank=0, zero_infinity=True)


def make_optimizer():
    return optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


def run_epoch(loader, optimizer, is_train):
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    steps = 0

    for images, labels in tqdm(loader):
        images = images.to(device, non_blocking=True)

        targets_list = [
            torch.tensor(text_encoder.encode(label), dtype=torch.long)
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

            if torch.isnan(loss):
                print("Skipping batch because loss became NaN")
                continue

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += loss.item()
        steps += 1

    if steps == 0:
        return float("inf")

    return total_loss / steps


best_val_loss = float("inf")
best_state_dict = None
epochs_without_improvement = 0

for epoch in range(1, CONTINUE_EPOCHS + 1):
    freeze_backbone = epoch <= FREEZE_BACKBONE_EPOCHS
    set_backbone_trainable(model, not freeze_backbone)
    optimizer = make_optimizer()

    phase = "frozen-backbone" if freeze_backbone else "full-finetune"
    print(f"\n=== Continue Epoch {epoch}/{CONTINUE_EPOCHS} | {phase} ===")

    train_loss = run_epoch(train_loader, optimizer, True)
    val_loss = run_epoch(val_loader, optimizer=None, is_train=False)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state_dict = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0

        os.makedirs("output/models/english", exist_ok=True)
        torch.save(best_state_dict, PRETRAINED_MODEL)
        print("Saved improved English model")
    else:
        epochs_without_improvement += 1
        print(f"No validation improvement for {epochs_without_improvement} epoch(s)")

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping triggered")
        break


if best_state_dict is not None:
    model.load_state_dict(best_state_dict)

print("\nRunning final test evaluation...")
test_loss = run_epoch(test_loader, optimizer=None, is_train=False)
print("Test Loss:", test_loss)
