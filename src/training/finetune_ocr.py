# src/training/finetune_ocr.py

# --- START: Forcefully Add Project Root to Path ---
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
# --- END: Forcefully Add Project Root to Path ---

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import ConcatDataset

from src.datasets.telugu_dataset import TeluguDataset
from src.models.vit_bilstm_ctc import ViTBILSTMCTC
from src.utils.text_utils import TextEncoder


def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    labels = [item['label'] for item in batch]
    return images, labels


def run_epoch(model, dataloader, text_encoder, criterion, optimizer, device, is_train=True):
    if is_train:
        model.train()
        desc = "Fine-tune Training"
    else:
        model.eval()
        desc = "Validation"

    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=desc)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)

        # Encode labels -> CTC targets
        targets_list = [
            torch.tensor(text_encoder.encode(label), dtype=torch.long)
            for label in labels
        ]

        targets = torch.cat(targets_list).to(device)
        target_lengths = torch.tensor(
            [len(t) for t in targets_list],
            dtype=torch.long,
            device=device
        )

        if is_train:
            optimizer.zero_grad()

        log_probs = model(images)  # (T, N, C)
        log_probs = nn.functional.log_softmax(log_probs, dim=2)

        T = log_probs.size(0)
        N = log_probs.size(1)
        input_lengths = torch.full(
            size=(N,),
            fill_value=T,
            dtype=torch.long,
            device=device
        )

        loss = criterion(log_probs, targets, input_lengths, target_lengths)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


if __name__ == '__main__':
    LANG_CODE = 'te'
    LANG_FOLDER = 'telugu'
    DATA_DIR = f'data/{LANG_FOLDER}/'

    # Fine-tuning hyperparams (smaller LR, fewer epochs)
    FT_NUM_EPOCHS = 5
    BATCH_SIZE = 16
    FT_LEARNING_RATE = 5e-5      # small LR for fine-tuning
    NUM_WORKERS = 4

    # Pretrained checkpoint to start from
    PRETRAINED_CKPT = 'output/checkpoints/vit_lstm_te_best.pth'
    FT_CKPT_OUT = 'output/checkpoints/vit_lstm_te_finetuned.pth'
    FT_LEARNING_RATE = 5e-5
    FT_NUM_EPOCHS = 5


    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        torch.backends.cudnn.benchmark = True

    os.makedirs('output/checkpoints', exist_ok=True)

    # --- TRANSFORMS ---
    # Stronger augmentations for training to improve robustness
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.3, contrast=0.3)
        ], p=0.5),
        transforms.RandomAffine(
            degrees=8,
            translate=(0.05, 0.05),
            scale=(0.9, 1.1),
            shear=5
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Validation: no augmentations, same as original
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

       # --- DATASETS ---

    # Real training data (with strong augmentations)
    train_dataset_real = TeluguDataset(
        data_dir=DATA_DIR,
        csv_path=os.path.join(DATA_DIR, 'labels_train.csv'),
        transform=train_transform
    )

    # Synthetic handwriting-style data
    SYNTH_DATA_DIR = "data/telugu_synth/"
    train_dataset_synth = TeluguDataset(
        data_dir=SYNTH_DATA_DIR,
        csv_path=os.path.join(SYNTH_DATA_DIR, 'labels_synth.csv'),
        transform=train_transform
    )

    # Combine both into one big training set
    train_dataset = ConcatDataset([train_dataset_real, train_dataset_synth])

    # Validation stays as original clean data
    val_dataset = TeluguDataset(
        data_dir=DATA_DIR,
        csv_path=os.path.join(DATA_DIR, 'labels_val.csv'),
        transform=val_transform
    )

    pin_memory = device.type == 'cuda'

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

    print(f"Real train dataset size:   {len(train_dataset_real)}")
    print(f"Synth train dataset size:  {len(train_dataset_synth)}")
    print(f"Total train dataset size:  {len(train_dataset)}")
    print(f"Val dataset size:          {len(val_dataset)}")

    # --- TEXT ENCODER & MODEL ---
    text_encoder = TextEncoder(lang_code=LANG_CODE)
    vocab_size = text_encoder.vocab_size()

    model = ViTBILSTMCTC(num_classes=vocab_size)

    # Load pretrained weights
    print(f"Loading pretrained checkpoint from: {PRETRAINED_CKPT}")
    state_dict = torch.load(PRETRAINED_CKPT, map_location=device)
    model.load_state_dict(state_dict)
    print("Loaded pretrained weights.")

    # Optional: multi-GPU
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)

    # --- LOSS & OPTIMIZER ---
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # Fine-tune with small LR
    optimizer = optim.AdamW(model.parameters(), lr=FT_LEARNING_RATE)

    # --- FINE-TUNING LOOP ---
    best_val_loss = float('inf')
    for epoch in range(1, FT_NUM_EPOCHS + 1):
        print(f"\n=== Fine-tune Epoch {epoch}/{FT_NUM_EPOCHS} ===")

        train_loss = run_epoch(
            model=model,
            dataloader=train_loader,
            text_encoder=text_encoder,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            is_train=True
        )

        val_loss = run_epoch(
            model=model,
            dataloader=val_loader,
            text_encoder=text_encoder,
            criterion=criterion,
            optimizer=None,
            device=device,
            is_train=False
        )

        print(f"[Fine-tune] Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save finetuned checkpoint
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(model_to_save.state_dict(), FT_CKPT_OUT)
            print(f"✅ New best val loss. Finetuned model saved to {FT_CKPT_OUT}")

    print("Fine-tuning finished.")
