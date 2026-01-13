#!/usr/bin/env python3


import os
import cv2
import csv
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
import chess
import chess.svg
import cairosvg
from io import BytesIO
import multiprocessing as mp

from common import setup_logging
from render_position import render_position, PIECE_NAMES, BOARD_NAMES


def get_board_state(board: chess.Board, flipped: bool) -> list[str]:
    """
    Returns 64 strings.
    If flipped=False: a8, b8 ... h1 (White perspective)
    If flipped=True:  h1, g1 ... a8 (Black perspective)
    """
    state = list()
    # Perspective determines the order of ranks and files
    ranks = range(8) if flipped else range(7, -1, -1)
    files = range(7, -1, -1) if flipped else range(8)

    for rank in ranks:
        for file in files:
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece is None:
                state.append("")
            else:
                color = "w" if piece.color == chess.WHITE else "b"
                state.append(f"{color}{piece.symbol().lower()}")
    return state


def flip_board_state(state: list[str], mode: str) -> list[str]:
    """Flips the visual board state (8x8 grid flattened).

    mode="h": horizontal flip (columns reversed)
    mode="v": vertical flip (rows reversed)
    """
    arr = np.array(state).reshape(8, 8)
    if mode == "h":
        arr = np.fliplr(arr)
    elif mode == "v":
        arr = np.flipud(arr)
    return arr.flatten().tolist()


def svg_to_cv2_image(svg_str: bytes, size: int) -> np.ndarray:
    """Render SVG board to an RGBA OpenCV image."""
    png_data = BytesIO()
    cairosvg.svg2png(bytestring=svg_str, write_to=png_data, output_width=size, output_height=size)
    img_array = np.frombuffer(png_data.getvalue(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)  # RGBA
    assert img is not None
    return img


def _get_new_board(existing_fens: dict[str, bool]) -> tuple[chess.Board, str]:
    """Generates a never-seen board and fen"""
    while True:
        board = chess.Board()
        for _ in range(random.randint(2, 30)):
            if board.is_game_over():
                break
            move = random.choice(list(board.legal_moves))
            board.push(move)
        fen = board.fen()
        # Check against shared dictionary
        if fen not in existing_fens:
            existing_fens[fen] = True
            return board, fen


def _generate_unique_board_with_chess(board: chess.Board, board_size: int) -> tuple[np.ndarray, float, bool]:
    """Generates an RGBA OpenCV image using chess-python"""
    show_coords = random.choice([True, False])
    flipped = random.choice([True, False])
    svg_data = chess.svg.board(board, size=board_size, coordinates=show_coords, flipped=flipped)
    board_img = svg_to_cv2_image(svg_data.encode("utf-8"), board_size)
    # Add correction ratio: empirically, coords add ~3% padding each side.
    coord_pad_ratio = 0.03 if show_coords else 0.0
    return board_img, coord_pad_ratio, flipped


def _generate_unique_board_with_images(fen: str, board_size: int) -> tuple[np.ndarray, bool]:
    """Generates an RGBA OpenCV image using github.com/GiorgioMegrelli/chess.com-boards-and-pieces"""
    side = random.choice(["white", "black"])
    flipped = (side == "black")
    square_size = board_size // 8
    pieces_name = random.choice(PIECE_NAMES)
    board_name = random.choice(BOARD_NAMES)
    img = render_position(fen, board_name, pieces_name, square_size, side)
    return img, flipped


def generate_unique_board(existing_fens: dict[str, bool], board_size: int) -> tuple[np.ndarray, float, list[str]]:
    """Generate a unique random chess position as BGR OpenCV image and return square data."""
    board, fen = _get_new_board(existing_fens)

    if random.random() > 0.5:
        board_img, pad, flipped = _generate_unique_board_with_chess(board, board_size)
    else:
        board_img, flipped = _generate_unique_board_with_images(fen, board_size)
        pad = 0.0

    # Generate the 64 fields based on whether the visual board is flipped
    board_state = get_board_state(board, flipped)

    return board_img, pad, board_state


def apply_random_noise(img: np.ndarray) -> tuple[np.ndarray, int]:
    """Apply augmentation using OpenCV. Returns (image, noise_type)."""
    max_noise_type = 9
    noise_type = np.random.randint(0, max_noise_type + 1)
    img = img.copy()
    if noise_type == 0:
        pass
    elif noise_type == 1:  # Blur
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    elif noise_type == 2:  # Brightness
        factor = random.uniform(0.7, 1.3)
        hsv = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2HSV)
        hsv[..., 2] = np.clip(hsv[..., 2] * factor, 0, 255)
        img[:, :, :3] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    elif noise_type == 3:  # Contrast
        alpha = random.uniform(0.7, 1.3)
        img[:, :, :3] = np.clip(alpha * img[:, :, :3], 0, 255).astype(np.uint8)
    elif noise_type == 4:  # LAB Equalization
        lab = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        img[:, :, :3] = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    elif noise_type == 5:  # Flip horizontally
        img = cv2.flip(img, 1)
    elif noise_type == 6:  # Flip vertically
        img = cv2.flip(img, 0)
    elif noise_type == 7:  # Sharpen
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        img[:, :, :3] = cv2.filter2D(img[:, :, :3], -1, kernel)
    elif noise_type == 8:  # Gaussian noise
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
    elif noise_type == 9:  # Slight blur + contrast mix
        img = cv2.GaussianBlur(img, (3, 3), 0)
        alpha = random.uniform(1.1, 1.3)
        img[:, :, :3] = np.clip(alpha * img[:, :, :3], 0, 255).astype(np.uint8)
    return img, noise_type


def random_axis_squish(img: np.ndarray, min_scale: float = 0.6, max_scale: float = 1.0) -> np.ndarray:
    """Squish the image along either width or height by a random factor in [min_scale, max_scale].

    - Preserves alpha channel if present (expects RGBA for boards).
    - Does not pad back to original size; returns the resized image.
    """
    assert 0 < min_scale <= max_scale <= 1.0, "min_scale and max_scale must be in (0,1] and min<=max"
    h, w = img.shape[:2]
    if w <= 1 or h <= 1:
        return img
    scale = random.uniform(min_scale, max_scale)
    axis = random.choice(["w", "h"])  # Which axis to squish
    if axis == "w":
        new_w = max(1, int(round(w * scale)))
        new_h = h
    else:
        new_w = w
        new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    return resized


def load_backgrounds(bg_dir: str, img_size: int, max_bg: int) -> list[np.ndarray]:
    """Load and resize backgrounds."""
    paths = sorted(
        [os.path.join(bg_dir, f) for f in os.listdir(bg_dir)
         if f.lower().endswith(("png", "jpg", "jpeg"))]
    )
    if not paths:
        raise FileNotFoundError(f"No valid images found in {bg_dir}")
    backgrounds = [
        cv2.resize(cv2.imread(p), (img_size, img_size))  # type: ignore
        for p in paths[:max_bg]
    ]
    logging.info(f"Loaded {len(backgrounds)} background(s).")
    return backgrounds


def overlay_rgba(bg: np.ndarray, fg: np.ndarray, x: int, y: int) -> np.ndarray:
    """Overlay RGBA image onto BGR background at (x, y)."""
    h, w = fg.shape[:2]
    if fg.shape[2] == 4:
        alpha = fg[:, :, 3] / 255.0
        for c in range(3):
            bg[y:y + h, x:x + w, c] = (
                alpha * fg[:, :, c] + (1 - alpha) * bg[y:y + h, x:x + w, c]
            )
    else:
        bg[y:y + h, x:x + w] = fg
    return bg


def random_place_board(bg: np.ndarray, board_img: np.ndarray, coord_pad_ratio: float) -> tuple[np.ndarray, list[float], list[float]]:
    """Place a non-rotated board randomly and return normalized UL/LR corners only.

    coord_pad_ratio trims the outer label margins from the board image if coordinates are drawn.
    """
    img = bg.copy()
    h_bg, w_bg = img.shape[:2]
    h_b, w_b = board_img.shape[:2]

    # Ensure it fits (guard against negatives)
    max_x = max(0, w_bg - w_b)
    max_y = max(0, h_bg - h_b)
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    # Paste directly
    img = overlay_rgba(img, board_img, x, y)

    # Adjust corners to exclude coordinate margin if present (use current squished size)
    dx = w_b * coord_pad_ratio
    dy = h_b * coord_pad_ratio
    ul = [(x + dx) / w_bg, (y + dy) / h_bg]
    lr = [(x + w_b - dx) / w_bg, (y + h_b - dy) / h_bg]

    return img, ul, lr


def process_background_task(bg: np.ndarray, start_id: int, args: argparse.Namespace, shared_fens: dict[str, bool]) -> tuple[list[list], int, int]:
    """Worker function to process a batch of images for a single background."""
    # Re-seed random generators for process safety
    random.seed()
    np.random.seed()

    results = list()
    current_positive_count = 0
    current_partial_count = 0
    current_id = start_id

    for _ in range(args.images_per_background):
        noisy_bg, _ = apply_random_noise(bg)

        # Positive
        ratio = random.uniform(args.min_ratio, args.max_ratio)
        board_size = int(args.img_size * ratio)
        board_img, coord_pad_ratio, board_state = generate_unique_board(shared_fens, board_size)
        # Randomly squish board to simulate non-square aspect from resizing
        if random.random() < args.squish_prob:
            board_img = random_axis_squish(board_img, args.squish_min, args.squish_max)

        # Apply noise (potentially flips)
        board_img, noise_type = apply_random_noise(board_img)

        # Update board_state if flipped
        if noise_type == 5:  # H-flip
            board_state = flip_board_state(board_state, "h")
        elif noise_type == 6:  # V-flip
            board_state = flip_board_state(board_state, "v")

        composed, ul, lr = random_place_board(noisy_bg, board_img, coord_pad_ratio)

        current_id += 1
        path = os.path.join(args.output_dir, "images", f"image_{current_id:09d}.png")
        cv2.imwrite(path, composed)
        results.append([current_id, 1, f"{ul[0]:.6f}", f"{ul[1]:.6f}", f"{lr[0]:.6f}", f"{lr[1]:.6f}"] + board_state)
        current_positive_count += 1

        # Negative
        neg = noisy_bg.copy()
        is_partial = random.random() > 0.5
        if is_partial:
            current_partial_count += 1

            board_img, _, _ = generate_unique_board(shared_fens, board_size)
            # Optionally squish before cropping to create rectangular fragments too
            if random.random() < args.squish_prob:
                board_img = random_axis_squish(board_img, args.squish_min, args.squish_max)

            # Apply noise to negative fragment (flips don't matter for labels here)
            board_img, _ = apply_random_noise(board_img)

            h, w = board_img.shape[:2]
            # Significant crop: remove 30%-70% from left/top
            crop_x = int(w * random.uniform(0.3, 0.7))
            crop_y = int(h * random.uniform(0.3, 0.7))
            crop_x = max(1, min(crop_x, w - 1))
            crop_y = max(1, min(crop_y, h - 1))
            board_img = board_img[crop_y:h, crop_x:w]
            # Place cropped fragment somewhere on background
            h2, w2 = board_img.shape[:2]
            if 0 < h2 <= neg.shape[0] and 0 < w2 <= neg.shape[1]:
                x = random.randint(0, neg.shape[1] - w2)
                y = random.randint(0, neg.shape[0] - h2)
                neg = overlay_rgba(neg, board_img, x, y)

        current_id += 1
        path = os.path.join(args.output_dir, "images", f"image_{current_id:09d}.png")
        cv2.imwrite(path, neg)
        # Negatives: no corners
        results.append([current_id, 0, 0, 0, 0, 0] + [""] * 64)

    return results, current_positive_count, current_partial_count


def main():
    parser = argparse.ArgumentParser(description="Synthetic Chessboard Dataset Generator")
    parser.add_argument("--background_dir", type=str, default="backgrounds_generated")
    parser.add_argument("--output_dir", type=str, default="synthetic_chess_dataset")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--n_backgrounds", type=int, default=1000)
    parser.add_argument("--images_per_background", type=int, default=50)
    parser.add_argument("--min_ratio", type=float, default=0.2)
    parser.add_argument("--max_ratio", type=float, default=1.0)
    parser.add_argument("--squish_prob", type=float, default=0.75, help="Probability to apply axis squish to the board image.")
    parser.add_argument("--squish_min", type=float, default=0.3, help="Minimum scale for the squished axis (0,1].")
    parser.add_argument("--squish_max", type=float, default=0.8, help="Maximum scale for the squished axis (<=1.0).")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 2), help="Number of worker processes")
    args = parser.parse_args()

    setup_logging()
    logging.info(f"Starting dataset generation with config: {args}")

    assert 0 < args.min_ratio < args.max_ratio <= 1, "Invalid board size ratio bounds"
    assert 0.0 <= args.squish_prob <= 1.0, "squish_prob must be in [0,1]"
    assert 0.0 < args.squish_min <= args.squish_max <= 1.0, "squish_min/max must be in (0,1] and min<=max"
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)

    backgrounds = load_backgrounds(args.background_dir, args.img_size, args.n_backgrounds)

    # Manager to share FENs across processes
    manager = mp.Manager()
    shared_fens = manager.dict()

    # Prepare inputs for parallel processing
    # Each task processes 1 background and produces 2 * images_per_background images (positive + negative)
    # Calculate starting IDs for each task to ensure uniqueness
    tasks = list()
    images_per_task = args.images_per_background * 2
    for i, bg in enumerate(backgrounds):
        start_id = i * images_per_task
        tasks.append((bg, start_id, args, shared_fens))

    logging.info(f"Processing with {args.workers} workers...")

    # Open CSV and write as results come in
    ann_path = os.path.join(args.output_dir, "annotations.csv")

    total_images = 0
    positive_count = 0
    partial_count = 0

    with open(ann_path, "w", newline="") as csvfile:
        sq_headers = [f"sq{i}" for i in range(64)]
        writer = csv.writer(csvfile)
        writer.writerow(["image_id", "has_board", "ul_x", "ul_y", "lr_x", "lr_y"] + sq_headers)  # header

        with mp.Pool(processes=args.workers) as pool:
            # starmap_async allows us to track progress
            results_iter = pool.starmap(process_background_task, tasks)

            for rows, p_count, part_count in tqdm(results_iter, total=len(tasks), desc="Backgrounds"):
                for row in rows:
                    writer.writerow(row)

                # Update stats
                positive_count += p_count
                partial_count += part_count
                total_images += len(rows)

    logging.info(f"Dataset created at {args.output_dir}")
    logging.info(f"Total images: {total_images} ({positive_count} positives)")
    logging.info(f"Partial images: {partial_count}")


if __name__ == "__main__":
    main()
