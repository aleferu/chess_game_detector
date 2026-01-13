#!/usr/bin/env python3


import os
import cv2
import argparse
import logging
import random
import numpy as np
from glob import glob

from common import setup_logging


def clip_uint8(x: np.ndarray) -> np.ndarray:
    """Clips an array to 0-255"""
    return np.clip(x, 0, 255).astype(np.uint8)


def make_black(img_size: int) -> np.ndarray:
    """Returns a black uint8 image"""
    return np.zeros((img_size, img_size, 3), dtype=np.uint8)


def make_white(img_size: int) -> np.ndarray:
    """Returns a white uint8 image"""
    return np.full((img_size, img_size, 3), 255, dtype=np.uint8)


def make_solid(img_size: int, bgr: tuple[int, int, int]) -> np.ndarray:
    """Return a solid image of the specified uint8 bgr color"""
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    img[:, :] = bgr
    return img


def make_rainbow(img_size: int) -> np.ndarray:
    """Horizontal hue sweep, full saturation and value"""
    h = img_size
    w = img_size
    hue = np.linspace(0, 179, w, dtype=np.uint8)
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:, :, 0] = hue[None, :]
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def make_random_rgb(img_size: int) -> np.ndarray:
    """Returns a random uint8 image"""
    return np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)


def make_noise_gaussian(img_size: int, mean: float = 127.0, std: float = 30.0) -> np.ndarray:
    """Returns an image generated with gaussian noise"""
    base = np.full((img_size, img_size, 3), mean, dtype=np.float32)
    noise = np.random.normal(0.0, std, base.shape).astype(np.float32)
    img = base + noise
    return clip_uint8(img)


def make_noise_salt_pepper(img_size: int, amount: float = 0.02, s_vs_p: float = 0.5) -> np.ndarray:
    """Returns an image generated with S&P noise"""
    # Start with mid-gray and add salt & pepper
    img = np.full((img_size, img_size, 3), 127, dtype=np.uint8)
    num_pixels = img_size * img_size
    num_salt = int(amount * num_pixels * s_vs_p)
    num_pepper = int(amount * num_pixels * (1.0 - s_vs_p))

    # Salt
    coords = (np.random.randint(0, img_size, num_salt), np.random.randint(0, img_size, num_salt))
    img[coords[0], coords[1]] = 255

    # Pepper
    coords = (np.random.randint(0, img_size, num_pepper), np.random.randint(0, img_size, num_pepper))
    img[coords[0], coords[1]] = 0
    return img


def _fbm_noise_single_channel(img_size: int, octaves: int = 5, persistence: float = 0.5) -> np.ndarray:
    """Fractal Brownian Motion on a single channel using resized white noise layers"""
    h = w = img_size
    result = np.zeros((h, w), dtype=np.float32)
    amplitude = 1.0
    total_amp = 0.0
    size = 8  # Starting low-res noise grid

    for _ in range(octaves):
        noise_small = np.random.rand(size, size).astype(np.float32)
        noise_large = cv2.resize(noise_small, (w, h), interpolation=cv2.INTER_CUBIC)
        result += amplitude * noise_large
        total_amp += amplitude
        amplitude *= persistence
        size = min(max(2, size * 2), img_size)
        if size >= img_size:
            # Final octave at full resolution
            noise_full = np.random.rand(h, w).astype(np.float32)
            result += amplitude * noise_full
            total_amp += amplitude
            break

    result /= (total_amp + 1e-6)
    result = (result * 255.0)
    return result.astype(np.uint8)


def make_noise_fbm(img_size: int, octaves: int = 5) -> np.ndarray:
    """Generate smooth fractal noise and colorize by stacking 3 shifted channels"""
    c0 = _fbm_noise_single_channel(img_size, octaves)
    c1 = _fbm_noise_single_channel(img_size, octaves)
    c2 = _fbm_noise_single_channel(img_size, octaves)
    img = np.stack([c0, c1, c2], axis=2)
    # Optional gentle contrast boost
    img = img.astype(np.float32)
    alpha = 1.2
    img = clip_uint8(alpha * (img - 127.0) + 127.0)
    return img


def _center_crop_to_square(img: np.ndarray) -> np.ndarray:
    """Squares image"""
    h, w = img.shape[:2]
    if h == w:
        return img
    if h > w:
        top = (h - w) // 2
        return img[top:top + w, :]
    else:
        left = (w - h) // 2
        return img[:, left:left + h]


def make_collage(img_size: int, source_paths: list[str]) -> np.ndarray:
    """Returns a collage of the given images"""
    if not source_paths:
        # fallback to random RGB if no sources
        return make_random_rgb(img_size)

    grid = random.choice([2, 3, 4])
    tile = img_size // grid
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    for gy in range(grid):
        for gx in range(grid):
            p = random.choice(source_paths)
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = _center_crop_to_square(img)
            img = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)
            y0 = gy * tile
            x0 = gx * tile
            canvas[y0:y0 + tile, x0:x0 + tile] = img

    # In case img_size not divisible by grid, stretch to full size
    canvas = cv2.resize(canvas, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return canvas


def load_background_sources(bg_dir: str) -> list[str]:
    """Loads the paths of the images located in the given dir"""
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*bmp")
    paths = list()
    for e in exts:
        paths.extend(glob(os.path.join(bg_dir, e)))
    paths = sorted(paths)
    return paths


def save_img(path: str, img: np.ndarray) -> None:
    """Saves an image using cv2"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


def main():
    parser = argparse.ArgumentParser(description="Generate various backgrounds for dataset synthesis")
    parser.add_argument("--output_dir", type=str, default="backgrounds_generated")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--per_type", type=int, default=10, help="Images to create for each non-collage type")
    parser.add_argument("--n_collages", type=int, default=100, help="Number of collage images to create")
    parser.add_argument("--background_dir", type=str, default="backgrounds", help="Source images for collages")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    setup_logging()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    logging.info(f"Generating backgrounds with config: {args}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Solid colors
    solids = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "yellow": (0, 255, 255),
        "magenta": (255, 0, 255),
        "cyan": (255, 255, 0),
    }

    # Standard types to generate per_type images for
    generators = {
        "rainbow": lambda: make_rainbow(args.img_size),
        "random_rgb": lambda: make_random_rgb(args.img_size),
        "gaussian": lambda: make_noise_gaussian(args.img_size),
        "salt_pepper": lambda: make_noise_salt_pepper(args.img_size, amount=0.03, s_vs_p=0.5),
        "fbm_noise": lambda: make_noise_fbm(args.img_size, octaves=5),
    }

    # Write solids
    for name, bgr in solids.items():
        img = make_solid(args.img_size, bgr) if name not in ("black", "white") else (
            make_black(args.img_size) if name == "black" else make_white(args.img_size)
        )
        path = os.path.join(args.output_dir, f"{name}.png")
        save_img(path, img)
        logging.info(f"Created {args.per_type} {name} backgrounds")

    # Write other generators
    for name, fn in generators.items():
        for i in range(1, args.per_type + 1):
            img = fn()
            path = os.path.join(args.output_dir, f"{name}_{i:04d}.png")
            save_img(path, img)
            if "rainbow" in name:
                break
        logging.info(f"Created {args.per_type} {name} backgrounds")

    # Collages
    source_paths = load_background_sources(args.background_dir)
    if not source_paths:
        logging.warning(f"No source images found in {args.background_dir}; collage generation will fallback to random colors")

    for i, source_path in enumerate(source_paths):
        img = cv2.imread(source_path)
        if img is None: continue
        res_img = cv2.resize(img, (args.img_size, args.img_size), interpolation=cv2.INTER_AREA)
        path = os.path.join(args.output_dir, f"old_background_{i}.png")
        save_img(path, res_img)
    logging.info(f"Created versions of the old backgrounds")

    for i in range(1, args.n_collages + 1):
        img = make_collage(args.img_size, source_paths)
        path = os.path.join(args.output_dir, f"collage_{i:04d}.png")
        save_img(path, img)
    logging.info(f"Created {args.n_collages} collage backgrounds from {len(source_paths)} sources")

    logging.info(f"Done. Backgrounds saved to {args.output_dir}")


if __name__ == "__main__":
    main()
