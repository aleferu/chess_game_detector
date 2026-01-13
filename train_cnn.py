#!/usr/bin/env python3


import os
import csv
import argparse
import logging
from typing import Any, Optional

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split
import torchinfo

from tqdm import tqdm

from common import setup_logging, TinyCNN, get_scaled_sigmoid, ToTensorNormalize


class RandomFlipAndNoise:
    """Applies random H/V flips and simple noise/blur and contrast/brightness jitter.

    Works with 1-channel grayscale NumPy arrays for speed. Bboxes are normalized [ulx,uly,lrx,lry].
    """

    hflip_p: float
    vflip_p: float
    brightness: float
    contrast: float
    noise_p: float
    blur_p: float

    def __init__(self, hflip_p: float = 0.5, vflip_p: float = 0.5,
                 brightness: float = 0.2, contrast: float = 0.2, noise_p: float = 0.3, blur_p: float = 0.2):
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.brightness = brightness
        self.contrast = contrast
        self.noise_p = noise_p
        self.blur_p = blur_p

    def __call__(self, img: np.ndarray, bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # img: HxW (grayscale uint8), bbox in [ul_x, ul_y, lr_x, lr_y] normalized
        ulx, uly, lrx, lry = bbox.tolist()

        # Horizontal flip
        if np.random.rand() < self.hflip_p:
            img = cv2.flip(img, 1)
            ulx, lrx = 1.0 - lrx, 1.0 - ulx

        # Vertical flip
        if np.random.rand() < self.vflip_p:
            img = cv2.flip(img, 0)
            uly, lry = 1.0 - lry, 1.0 - uly

        # Brightness and contrast jitter on grayscale
        arr = img.astype(np.float32)
        if self.brightness > 0:
            factor = 1.0 + np.random.uniform(-self.brightness, self.brightness)
            arr = np.clip(arr * factor, 0, 255)
        if self.contrast > 0:
            mean = np.mean(arr, keepdims=True)
            alpha = 1.0 + np.random.uniform(-self.contrast, self.contrast)
            arr = np.clip((arr - mean) * alpha + mean, 0, 255)

        # Blur
        if np.random.rand() < self.blur_p:
            k = int(np.random.choice([3, 5]))
            arr = cv2.GaussianBlur(arr, (k, k), 0)

        # Add Gaussian noise
        if np.random.rand() < self.noise_p:
            noise = np.random.normal(0, 8.0, size=arr.shape).astype(np.float32)
            arr = np.clip(arr + noise, 0, 255)

        img = arr.astype(np.uint8)
        bbox = np.array([ulx, uly, lrx, lry], dtype=np.float32)
        return img, bbox


class ChessboardDatasetRAM(Dataset):
    """Adapted torch.Dataset for our use case. RAM version"""

    root: str
    img_dir: str
    ann_path: str
    img_size: int
    train: bool
    augment: bool
    records: list[dict[str, Any]]
    images: list[np.ndarray]
    flip_noise: Optional[RandomFlipAndNoise]
    to_tensor: ToTensorNormalize

    def __init__(self, root: str, img_size: int = 256, train: bool = True, augment: bool = True):
        self.root = root
        gray_dir = os.path.join(root, "gray_images")
        self.img_dir = gray_dir if os.path.isdir(gray_dir) else os.path.join(root, "images")
        self.ann_path = os.path.join(root, "annotations.csv")
        self.img_size = img_size
        self.train = train
        self.augment = augment

        if not os.path.isfile(self.ann_path):
            raise FileNotFoundError(f"Annotations file not found: {self.ann_path}")

        # Read CSV
        self.records = list()
        with open(self.ann_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = int(row["image_id"])
                has_board = int(row["has_board"])
                try:
                    ulx = float(row["ul_x"])
                    uly = float(row["ul_y"])
                    lrx = float(row["lr_x"])
                    lry = float(row["lr_y"])
                except Exception:
                    ulx = uly = lrx = lry = 0.0
                self.records.append({
                    "image_id": image_id,
                    "has_board": has_board,
                    "bbox": np.array([ulx, uly, lrx, lry], dtype=np.float32)
                })

        # Load all images into memory
        self.images = list()
        for rec in tqdm(self.records, desc="Loading images"):
            img_path = os.path.join(self.img_dir, f"image_{rec['image_id']:09d}.png")
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Failed to read image: {img_path}")
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            self.images.append(img)

        self.flip_noise = RandomFlipAndNoise() if augment else None
        self.to_tensor = ToTensorNormalize()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        img = self.images[idx]

        bbox = rec["bbox"].copy()
        is_board = float(rec["has_board"])

        # Apply augmentations only in train mode
        if self.train and self.augment and self.flip_noise is not None:
            img, bbox = self.flip_noise(img, bbox)

        x = self.to_tensor(img)
        y_cls = torch.tensor([is_board], dtype=torch.float32)  # shape (1,)
        y_bbox = torch.from_numpy(bbox.astype(np.float32))     # shape (4,)

        return {
            "image": x,
            "target_cls": y_cls,
            "target_bbox": y_bbox,
            "is_positive": torch.tensor([1.0 if is_board > 0.5 else 0.0], dtype=torch.float32)
        }


class ChessboardDatasetDisk(Dataset):
    """Adapted torch.Dataset for our use case. Disk version"""

    root: str
    img_dir: str
    ann_path: str
    img_size: int
    train: bool
    augment: bool
    records: list[dict[str, Any]]
    flip_noise: Optional[RandomFlipAndNoise]
    to_tensor: ToTensorNormalize

    def __init__(self, root: str, img_size: int = 256, train: bool = True, augment: bool = True):
        self.root = root
        gray_dir = os.path.join(root, "gray_images")
        self.img_dir = gray_dir if os.path.isdir(gray_dir) else os.path.join(root, "images")
        self.ann_path = os.path.join(root, "annotations.csv")
        self.img_size = img_size
        self.train = train
        self.augment = augment

        if not os.path.isfile(self.ann_path):
            raise FileNotFoundError(f"Annotations file not found: {self.ann_path}")

        # Read CSV
        self.records = list()
        with open(self.ann_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = int(row["image_id"])
                has_board = int(row["has_board"])
                try:
                    ulx = float(row["ul_x"])
                    uly = float(row["ul_y"])
                    lrx = float(row["lr_x"])
                    lry = float(row["lr_y"])
                except Exception:
                    ulx = uly = lrx = lry = 0.0
                self.records.append({
                    "image_id": image_id,
                    "has_board": has_board,
                    "bbox": np.array([ulx, uly, lrx, lry], dtype=np.float32)
                })

        # Removed: Loading all images into memory (self.images = list())

        self.flip_noise = RandomFlipAndNoise() if augment else None
        self.to_tensor = ToTensorNormalize()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]

        # Load image from disk on demand
        img_path = os.path.join(self.img_dir, f"image_{rec['image_id']:09d}.png")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")

        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        bbox = rec["bbox"].copy()
        is_board = float(rec["has_board"])

        # Apply augmentations only in train mode
        if self.train and self.augment and self.flip_noise is not None:
            img, bbox = self.flip_noise(img, bbox)

        x = self.to_tensor(img)
        y_cls = torch.tensor([is_board], dtype=torch.float32)  # shape (1,)
        y_bbox = torch.from_numpy(bbox.astype(np.float32))     # shape (4,)

        return {
            "image": x,
            "target_cls": y_cls,
            "target_bbox": y_bbox,
            "is_positive": torch.tensor([1.0 if is_board > 0.5 else 0.0], dtype=torch.float32)
        }


def bbox_iou_xyxy(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """IoU for [x1, y1, x2, y2] boxes. Expects (B, 4)."""
    inter_x1 = torch.max(box1[:, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, 3], box2[:, 3])

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area1 = (box1[:, 2] - box1[:, 0]).clamp(0) * (box1[:, 3] - box1[:, 1]).clamp(0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(0) * (box2[:, 3] - box2[:, 1]).clamp(0)
    union = area1 + area2 - inter_area + eps

    return (inter_area / union).clamp(0, 1)


def bbox_ciou_loss_xyxy(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """CIoU loss for boxes in xyxy format (normalized [0,1])."""
    # Centers and sizes
    px = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    py = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    pw = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(eps)
    ph = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(eps)

    gx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    gy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
    gw = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(eps)
    gh = (target_boxes[:, 3] - target_boxes[:, 1]).clamp(eps)

    # IoU
    iou = bbox_iou_xyxy(pred_boxes, target_boxes, eps)

    # Center distance
    c_dist = (px - gx)**2 + (py - gy)**2

    # Enclosing diagonal
    x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    c_diag = (x2 - x1)**2 + (y2 - y1)**2 + eps

    # Aspect ratio term
    v = (4 / (np.pi ** 2)) * torch.pow(torch.atan(gw / gh) - torch.atan(pw / ph), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (c_dist / c_diag + v * alpha)
    return 1 - ciou  # loss


def compute_losses(outputs: torch.Tensor, batch: dict, lambda_reg: float = 1.0, lambda_l1: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    board_logit = outputs[:, 0]
    bbox_raw = outputs[:, 1:5]

    target_cls = batch["target_cls"].squeeze(1)  # [B]
    target_bbox = batch["target_bbox"]           # [B,4]
    pos_mask = batch["is_positive"].squeeze(1)   # [B]

    # Classification
    cls_loss = F.binary_cross_entropy_with_logits(board_logit, target_cls)

    # Bounding box regression (YOLO-style CIoU loss + L1)
    if pos_mask.sum() > 0:
        # Sigmoid makes 0 and 1 unreachable, this is a hacky way of getting around that
        # Ouput in (-0.1, 1.1) range now
        pred_bbox = get_scaled_sigmoid(bbox_raw, clamp=False)

        ciou_loss = bbox_ciou_loss_xyxy(pred_bbox[pos_mask > 0.5], target_bbox[pos_mask > 0.5])
        l1_loss = F.smooth_l1_loss(pred_bbox[pos_mask > 0.5], target_bbox[pos_mask > 0.5])

        reg_loss = ciou_loss.mean() + lambda_l1 * l1_loss
        # Doesn't work, idk why
        # from torchvision.ops import complete_box_iou_loss
        # reg_loss = complete_box_iou_loss(
        #     pred_bbox[pos_mask > 0.5],
        #     target_bbox[pos_mask > 0.5],
        #     reduction="mean"
        # )
    else:
        reg_loss = torch.tensor(0.0, device=outputs.device)

    total = cls_loss + lambda_reg * reg_loss
    return total, cls_loss.detach(), reg_loss.detach()


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) > 0.5).float()
    return (preds == targets).float().mean().item()


def bbox_mae(pred_raw: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.sum() == 0:
        return 0.0
    pred = get_scaled_sigmoid(pred_raw[mask > 0.5], clamp=True)
    tgt = target[mask > 0.5]
    return (pred - tgt).abs().mean().item()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, lambda_reg: float = 1.0, lambda_l1: float = 1.0) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_mae = 0.0
    n = 0
    for batch in loader:
        imgs = batch["image"].to(device)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(imgs)
        loss, _, _ = compute_losses(out, batch, lambda_reg=lambda_reg, lambda_l1=lambda_l1)
        total_loss += loss.item() * imgs.size(0)
        total_acc += accuracy_from_logits(out[:, 0], batch["target_cls"].squeeze(1)) * imgs.size(0)
        total_mae += bbox_mae(out[:, 1:5], batch["target_bbox"], batch["is_positive"].squeeze(1)) * imgs.size(0)
        n += imgs.size(0)
    return {
        "loss": total_loss / max(1, n),
        "acc": total_acc / max(1, n),
        "bbox_mae": total_mae / max(1, n)
    }


def train(args):
    setup_logging()
    logging.info(f"Starting training with config: {vars(args)}")

    # CUDA-only execution
    if not torch.cuda.is_available():
        logging.error("CUDA device not available. Aborting training as requested.")
        return {
            "aborted_no_cuda": True
        }
    device = torch.device("cuda")

    # Auto workers: CPUs - 2, clipped to 1
    try:
        cpu_cnt = os.cpu_count() or 1
    except Exception:
        cpu_cnt = 1
    num_workers = max(1, cpu_cnt - 2)

    # Dataset
    full_ds = ChessboardDatasetRAM(args.data_dir, img_size=args.img_size, train=True, augment=True)
    val_len = int(len(full_ds) * args.val_split)
    train_len = len(full_ds) - val_len
    train_ds, val_indices = random_split(range(len(full_ds)), [train_len, val_len], generator=torch.Generator().manual_seed(args.seed))  # type: ignore

    # Build actual datasets: train uses augmentations; val disables them
    train_ds = torch.utils.data.Subset(full_ds, train_ds.indices)
    val_base = ChessboardDatasetRAM(args.data_dir, img_size=args.img_size, train=False, augment=False)
    val_ds = torch.utils.data.Subset(val_base, val_indices.indices)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Model
    model = TinyCNN(in_channels=1, base=args.base_channels, input_size=args.img_size).to(device)
    torchinfo.summary(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda")  # type: ignore
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=15
    )

    os.makedirs(args.out_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = 0
    history = list()

    max_epochs = getattr(args, "max_epochs")
    patience = getattr(args, "early_stop_patience")
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}")
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_mae = 0.0
        n_seen = 0
        for batch in pbar:
            imgs = batch["image"].to(device, non_blocking=True)
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=True):  # type: ignore
                out = model(imgs)
                loss, _, _ = compute_losses(out, batch, lambda_reg=args.lambda_reg, lambda_l1=args.lambda_l1)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = imgs.size(0)
            epoch_loss += loss.item() * bs
            epoch_acc += accuracy_from_logits(out[:, 0], batch["target_cls"].squeeze(1)) * bs
            epoch_mae += bbox_mae(out[:, 1:5], batch["target_bbox"], batch["is_positive"].squeeze(1)) * bs
            n_seen += bs
            pbar.set_postfix({
                "loss": f"{epoch_loss/max(1, n_seen):.4f}",
                "acc": f"{epoch_acc/max(1, n_seen):.3f}",
                "mae": f"{epoch_mae/max(1, n_seen):.3f}"
            })

        # Validation
        val_metrics = evaluate(model, val_loader, device, args.lambda_reg, args.lambda_l1)  # type: ignore
        scheduler.step(val_metrics["loss"])

        curr_lr = optimizer.param_groups[0]["lr"]
        logging.info(f"Epoch {epoch} with lr {curr_lr}: train_loss={epoch_loss / n_seen:.4f} train_acc={epoch_acc / n_seen:.3f} train_mae={epoch_mae / n_seen:.3f} | "
                     f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.3f} val_mae={val_metrics['bbox_mae']:.3f}")

        # Save history
        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss / n_seen,
            "train_acc": epoch_acc / n_seen,
            "train_mae": epoch_mae / n_seen,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_mae": val_metrics["bbox_mae"],
        })
        with open(os.path.join(args.out_dir, "history.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            for row in history:
                writer.writerow(row)

        # Early stopping on best (lowest) val_loss
        if val_metrics["loss"] < best_val_loss - 1e-6:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_no_improve = 0
            best_path = os.path.join(args.out_dir, "best_model.pth")
            torch.save({"model": model.state_dict(), "args": vars(args), "best_epoch": best_epoch, "best_val_loss": best_val_loss}, best_path)
            logging.info(f"Saved best model to {best_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    last_path = os.path.join(args.out_dir, "last_model.pth")
    torch.save({"model": model.state_dict(), "args": vars(args), "last_epoch": epoch}, last_path)  # type: ignore
    logging.info(f"Saved last model to {last_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a tiny CNN (grayscale) for chessboard presence")
    parser.add_argument("--data_dir", type=str, default="synthetic_chess_dataset", help="Dataset root with annotations.csv and images/ (optionally gray_images/)")
    parser.add_argument("--out_dir", type=str, default="out/exp0")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--early_stop_patience", type=int, default=50, help="Patience for early stopping on val_loss")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--lambda_reg", type=float, default=10.0)
    parser.add_argument("--lambda_l1", type=float, default=2.0)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_channels", type=int, default=32, help="Model width (larger is slower)")
    parser.add_argument("--use_ram", default=True, action="store_true")

    args = parser.parse_args()

    # CUDA-only execution for the whole script
    if not torch.cuda.is_available():
        logging.error("CUDA device not available. Aborting.")
        return

    train(args)


if __name__ == "__main__":
    main()
