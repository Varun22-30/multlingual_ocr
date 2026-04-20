import copy
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.datasets.cvl_word_dataset import CVLWordDataset
from src.models.vit_bilstm_ctc import ViTBILSTMCTC
from src.utils.text_utils import TextEncoder


LANG_CODE = "en"
TRAIN_WORDS_DIR = os.path.join("data", "cvl-database-1-1", "trainset", "words")
TEST_WORDS_DIR = os.path.join("data", "cvl-database-1-1", "testset", "words")

BASE_MODEL_PATH = os.path.join("output", "models", "english", "vit_lstm_en_best.pth")
OUTPUT_CKPT = os.path.join("output", "models", "english", "vit_lstm_en_cvl_finetuned.pth")

NUM_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
PATIENCE = 3
FREEZE_BACKBONE_EPOCHS = 2
VAL_SPLIT = 0.1
SEED = 42
NUM_WORKERS = 0


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

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
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


def make_optimizer(model):
    return optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


def run_epoch(model, loader, text_encoder, criterion, optimizer, is_train):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    steps = 0

    for images, labels in tqdm(loader, desc="Train" if is_train else "Eval"):
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


if __name__ == "__main__":
    text_encoder = TextEncoder(lang_code=LANG_CODE)

    full_train_dataset = CVLWordDataset(
        words_root=TRAIN_WORDS_DIR,
        transform=train_transform,
        allowed_chars=text_encoder.charset,
    )
    full_eval_dataset = CVLWordDataset(
        words_root=TRAIN_WORDS_DIR,
        transform=eval_transform,
        allowed_chars=text_encoder.charset,
    )
    test_dataset = CVLWordDataset(
        words_root=TEST_WORDS_DIR,
        transform=eval_transform,
        allowed_chars=text_encoder.charset,
    )

    n_total = len(full_train_dataset)
    n_val = max(1, int(n_total * VAL_SPLIT))
    n_train = n_total - n_val
    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(SEED)).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_eval_dataset, val_indices)

    print(f"CVL train samples: total={n_total}, train={n_train}, val={n_val}")
    print(f"CVL test samples: {len(test_dataset)}")
    print(f"Skipped unsupported train samples: {full_train_dataset.skipped_unsupported}")
    print(f"Skipped unsupported test samples: {test_dataset.skipped_unsupported}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    model = ViTBILSTMCTC(num_classes=text_encoder.vocab_size()).to(device)

    resume_path = OUTPUT_CKPT if os.path.exists(OUTPUT_CKPT) else BASE_MODEL_PATH
    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"No English checkpoint found at: {resume_path}")

    resumed_from_cvl = resume_path == OUTPUT_CKPT
    if resumed_from_cvl:
        print(f"Resuming CVL fine-tuning from existing checkpoint: {OUTPUT_CKPT}")
    else:
        print(f"Loading base English model from: {BASE_MODEL_PATH}")

    model.load_state_dict(torch.load(resume_path, map_location=device))

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    best_val_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        freeze_backbone = (not resumed_from_cvl) and epoch <= FREEZE_BACKBONE_EPOCHS
        set_backbone_trainable(model, not freeze_backbone)
        optimizer = make_optimizer(model)

        phase = "frozen-backbone" if freeze_backbone else "full-finetune"
        print(f"\n=== CVL Fine-tune Epoch {epoch}/{NUM_EPOCHS} | {phase} ===")

        train_loss = run_epoch(model, train_loader, text_encoder, criterion, optimizer, True)
        val_loss = run_epoch(model, val_loader, text_encoder, criterion, optimizer=None, is_train=False)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            os.makedirs(os.path.dirname(OUTPUT_CKPT), exist_ok=True)
            torch.save(best_state_dict, OUTPUT_CKPT)
            print(f"Saved improved CVL-finetuned English model to {OUTPUT_CKPT}")
        else:
            epochs_without_improvement += 1
            print(f"No validation improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping triggered.")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    print("\nRunning final CVL test evaluation...")
    test_loss = run_epoch(model, test_loader, text_encoder, criterion, optimizer=None, is_train=False)
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"CVL Test Loss: {test_loss:.4f}")
    print(f"Final checkpoint: {OUTPUT_CKPT}")
