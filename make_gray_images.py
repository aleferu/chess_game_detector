#!/usr/bin/env python3


import os
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
from tqdm import tqdm

from common import setup_logging


def convert_one(src_path: str, dst_path: str) -> bool:
    """Reads an image, converts to gray-scale, and saves it"""
    try:
        img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("cv2.imread returned None")
        # Ensure destination directory exists (in case called standalone)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        ok = cv2.imwrite(dst_path, img)
        if not ok:
            raise IOError("cv2.imwrite failed")
        return True
    except Exception as e:
        logging.error(f"Failed to convert {src_path}: {e}")
        return False


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Pre-generate grayscale copies of dataset images")
    parser.add_argument("--data_dir", type=str, default="synthetic_chess_dataset",
                        help="Dataset root containing images/ and annotations.csv")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 2),
                        help="Parallel workers for I/O bound conversion")
    args = parser.parse_args()

    img_dir = os.path.join(args.data_dir, "images")
    gray_dir = os.path.join(args.data_dir, "gray_images")

    if not os.path.isdir(img_dir):
        logging.error(f"Images directory not found: {img_dir}")
        return
    os.makedirs(gray_dir, exist_ok=True)

    # Collect PNG files
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(".png")]
    files.sort()
    if not files:
        logging.warning("No PNG files found to convert.")
        return

    logging.info(f"Converting {len(files)} images to grayscale: {img_dir} -> {gray_dir}")

    tasks = list()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for name in files:
            src = os.path.join(img_dir, name)
            dst = os.path.join(gray_dir, name)
            # Skip if up-to-date (destination exists and is newer or same mtime)
            try:
                if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
                    continue
            except Exception:
                pass
            tasks.append(ex.submit(convert_one, src, dst))

        for fut in tqdm(as_completed(tasks), total=len(tasks), desc="Grayscaling"):
            _ = fut.result()

    logging.info("Grayscale conversion completed.")


if __name__ == "__main__":
    main()
