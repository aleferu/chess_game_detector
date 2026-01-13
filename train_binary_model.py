#!/usr/bin/env python3


import os
import csv
import argparse
import logging
import random
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from common import setup_logging
import torchinfo


class MicroBinaryCNN(nn.Module):
    """
    Extremely lightweight CNN for Binary Classification (Empty vs Piece).
    Designed for high-speed inference.
    """
    def __init__(self, input_channels=1):
        super().__init__()
        # Input: [B, 1, 64, 64]
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # -> 32x32

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # -> 16x16

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)) # -> [B, 64, 1, 1]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1) # Output logit
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class RandomAugment:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        # Simple augmentations for small grayscale images
        if random.random() < self.p:
            alpha = random.uniform(0.8, 1.2)
            beta = random.uniform(-20, 20)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        return img


class BinaryPieceDataset(Dataset):
    def __init__(self, root_dir, csv_file, img_size=64, train=True, augment=True):
        self.root_dir = root_dir
        self.img_size = img_size
        self.train = train
        self.augment = augment and train
        self.samples = list()

        # Load CSV
        csv_path = os.path.join(root_dir, csv_file)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # We only need file_name and is_piece
                self.samples.append((row["file_name"], int(row["is_piece"])))

        self.augmentor = RandomAugment() if self.augment else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img_path = os.path.join(self.root_dir, "images", fname)

        # Read as Grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Return a black image if missing (robustness)
            raise ValueError("Invalid image! %s", img_path)
        else:
            img = cv2.resize(img, (self.img_size, self.img_size))

        if self.augmentor:
            img = self.augmentor(img)

        # To Tensor [0, 1]
        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        label_tensor = torch.tensor([label], dtype=torch.float32)

        return img_tensor, label_tensor


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": total_loss / max(1, total),
        "acc": total_correct / max(1, total)
    }


def train(args):
    setup_logging()
    logging.info(f"Starting training with config: {vars(args)}")

    if not torch.cuda.is_available():
        logging.error("CUDA device not available. Aborting.")
        return
    device = torch.device("cuda")

    # Data
    full_ds = BinaryPieceDataset(args.data_dir, args.csv_file, img_size=args.img_size, train=True, augment=True)

    # Split
    val_len = int(len(full_ds) * args.val_split)
    train_len = len(full_ds) - val_len
    train_indices, val_indices = random_split(range(len(full_ds)), [train_len, val_len], generator=torch.Generator().manual_seed(args.seed))  # type: ignore

    # Subsets (Augment only on train)
    train_ds = torch.utils.data.Subset(full_ds, train_indices)  # type: ignore

    val_base_ds = BinaryPieceDataset(args.data_dir, args.csv_file, img_size=args.img_size, train=False, augment=False)
    val_ds = torch.utils.data.Subset(val_base_ds, val_indices)  # type: ignore

    num_workers = min(os.cpu_count() or 1, 4)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Model
    model = MicroBinaryCNN().to(device)
    torchinfo.summary(model, input_size=(args.batch_size, 1, args.img_size, args.img_size))

    # Optimization
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20)
    scaler = torch.amp.GradScaler("cuda")  # type: ignore
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    os.makedirs(args.out_dir, exist_ok=True)
    history = list()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):  # type: ignore
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = imgs.size(0)
            train_loss += loss.item() * bs
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            total += bs

            pbar.set_postfix({"loss": f"{train_loss/total:.4f}", "acc": f"{train_correct/total:.4f}"})

        # Validation
        val_metrics = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_metrics["loss"])

        curr_lr = optimizer.param_groups[0]["lr"]
        logging.info(f"Epoch {epoch} (lr={curr_lr:.1e}): "
                     f"Train [Loss: {train_loss/total:.4f}, Acc: {train_correct/total:.4f}] | "
                     f"Val [Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}]")

        # History
        history.append({
            "epoch": epoch,
            "train_loss": train_loss/total,
            "train_acc": train_correct/total,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"]
        })

        # Save History
        with open(os.path.join(args.out_dir, "history.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            for row in history:
                writer.writerow(row)

        # Checkpoints
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_binary_model.pth"))
            logging.info(f"Saved best model (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logging.info(f"Early stopping triggered after {args.patience} epochs.")
                break

        # Save last
        torch.save(model.state_dict(), os.path.join(args.out_dir, "last_binary_model.pth"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="pieces_dataset")
    parser.add_argument("--csv_file", type=str, default="pieces_annotations.csv")
    parser.add_argument("--out_dir", type=str, default="out/binary_model")
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
