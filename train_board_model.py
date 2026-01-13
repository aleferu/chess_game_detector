#!/usr/bin/env python3


import os
import csv
import argparse
import logging
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from common import setup_logging, PIECE_MAP, INV_PIECE_MAP, MicroBoardCNN
import torchinfo
from typing import Any


class RandomAugment:
    """Applies a random augmentation to an image"""

    p: float

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            # Horizontal Flip
            if random.random() < 0.5:
                img = cv2.flip(img, 1)

            # Random Brightness/Contrast
            if random.random() < 0.5:
                alpha = random.uniform(0.7, 1.3)
                beta = random.uniform(-30, 30)
                img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

            # Noise
            if random.random() < 0.2:
                noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
                img = cv2.add(img, noise)

        return img


class BoardDataset(Dataset):
    """torch.Dataset implementation for our use case. Uses the disk"""

    root_dir: str
    img_size: int
    train: bool
    augment: bool
    samples: list[tuple[str, int]]

    def __init__(self, root_dir: str, csv_file: str, img_size: int = 64, train: bool = True, augment: bool = True):
        self.root_dir = root_dir
        self.img_size = img_size
        self.train = train
        self.augment = augment and train
        self.samples = list()

        # Load CSV
        csv_path = os.path.join(root_dir, csv_file)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        # Track counts for weighting
        self.class_counts = {i: 0 for i in range(13)}

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ptype = row["piece_type"]
                if row["is_piece"] == "0":
                    ptype = ""

                if ptype in PIECE_MAP:
                    lbl = PIECE_MAP[ptype]
                    self.samples.append((row["file_name"], lbl))
                    self.class_counts[lbl] += 1

        self.augmentor = RandomAugment() if self.augment else None
        logging.info(f"Loaded {len(self.samples)} samples.")

    def get_class_weights(self, device: torch.device) -> torch.Tensor:
        """
        Calculate class weights using Inverse Class Frequency:
        W_c = N_total / (N_classes * N_c)
        """
        total_samples = len(self.samples)
        num_classes = 13
        weights = list()

        logging.info("Class distribution and weights:")
        for i in range(num_classes):
            count = self.class_counts[i]
            # Avoid division by zero if a class is missing
            w = total_samples / (num_classes * max(count, 1))
            weights.append(w)
            cls_name = INV_PIECE_MAP[i] if INV_PIECE_MAP[i] else "Empty"
            logging.info(f"  Class {i} ({cls_name}): {count} samples -> Weight: {w:.4f}")

        return torch.tensor(weights, dtype=torch.float32).to(device)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        fname, label = self.samples[idx]
        img_path = os.path.join(self.root_dir, "images", fname)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback for missing images, though ideally shouldn't happen
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            logging.warning(f"Warning: Image not found {img_path}")
        else:
            img = cv2.resize(img, (self.img_size, self.img_size))

        if self.augmentor:
            img = self.augmentor(img)

        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, label_tensor


@torch.no_grad()
def evaluate(model: MicroBoardCNN, loader: DataLoader, device: torch.device, criterion: nn.CrossEntropyLoss, num_classes: int = 13) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    # Confusion Matrix: [True, Pred]
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(outputs, dim=1)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Update Confusion Matrix
        for t, p in zip(labels.view(-1), preds.view(-1)):
            conf_matrix[t.long(), p.long()] += 1

    return {
        "loss": total_loss / max(1, total),
        "acc": total_correct / max(1, total),
        "conf_matrix": conf_matrix.cpu().numpy()
    }


def train(args: argparse.Namespace):
    setup_logging()
    logging.info(f"Starting Board Classification training with config: {vars(args)}")

    if not torch.cuda.is_available():
        logging.error("CUDA device not available. Aborting.")
        return
    device = torch.device("cuda")

    # Data
    full_ds = BoardDataset(args.data_dir, args.csv_file, img_size=args.img_size, train=True, augment=True)

    # Calculate Weights based on full dataset
    class_weights = full_ds.get_class_weights(device)

    # Split
    val_len = int(len(full_ds) * args.val_split)
    train_len = len(full_ds) - val_len
    train_indices, val_indices = random_split(range(len(full_ds)), [train_len, val_len], generator=torch.Generator().manual_seed(args.seed))  # type: ignore

    # Subsets
    train_ds = torch.utils.data.Subset(full_ds, train_indices)  # type: ignore

    # Validation Dataset (Disable augmentation)
    val_base_ds = BoardDataset(args.data_dir, args.csv_file, img_size=args.img_size, train=False, augment=False)
    val_ds = torch.utils.data.Subset(val_base_ds, val_indices)  # type: ignore

    num_workers = min(os.cpu_count() or 1, 4)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Model
    model = MicroBoardCNN(num_classes=13).to(device)
    torchinfo.summary(model, input_size=(args.batch_size, 1, args.img_size, args.img_size))

    # Optimization
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20)
    scaler = torch.amp.GradScaler("cuda")  # type: ignore

    # Weighted Loss for Imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)

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
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            total += bs

            pbar.set_postfix({"loss": f"{train_loss/total:.4f}", "acc": f"{train_correct/total:.4f}"})

        # Validation
        val_metrics = evaluate(model, val_loader, device, criterion, num_classes=13)
        scheduler.step(val_metrics["loss"])

        curr_lr = optimizer.param_groups[0]["lr"]
        logging.info(f"Epoch {epoch} (lr={curr_lr:.1e}): "
                     f"Train [Loss: {train_loss/total:.4f}, Acc: {train_correct/total:.4f}] | "
                     f"Val [Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}]")

        # Log Per-Class Accuracy
        cm = val_metrics["conf_matrix"]
        class_accs = cm.diagonal() / cm.sum(axis=1).clip(min=1)

        # Confusion Matrix (Rows=True, Cols=Pred)
        logging.info("  Confusion Matrix (Rows=True, Cols=Pred):")
        # Header
        header = "      " + " ".join([f"{INV_PIECE_MAP.get(i, str(i)):>4}" for i in range(13)])
        logging.info(header)
        for i in range(13):
            row_str = f"{INV_PIECE_MAP.get(i, str(i)):>4}: " + " ".join([f"{cm[i, j]:>4}" for j in range(13)])
            logging.info(row_str)

        # Per-Class Accuracy
        class_accs = cm.diagonal() / cm.sum(axis=1).clip(min=1)
        logging.info("  Per-Class Accuracy:")
        for i, acc in enumerate(class_accs):
            cls_name = INV_PIECE_MAP.get(i, str(i))
            logging.info(f"    {cls_name:<4}: {acc*100:.1f}%")

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
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_board_model.pth"))
            logging.info(f"Saved best model (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logging.info(f"Early stopping triggered after {args.patience} epochs.")
                break

        # Save last
        torch.save(model.state_dict(), os.path.join(args.out_dir, "last_board_model.pth"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="pieces_dataset")
    parser.add_argument("--csv_file", type=str, default="pieces_annotations.csv")
    parser.add_argument("--out_dir", type=str, default="out/board_model")
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
