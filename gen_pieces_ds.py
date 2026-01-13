#!/usr/bin/env python3


import os
import csv
import argparse
import logging
import cv2
import numpy as np
from tqdm import tqdm
import random

from common import setup_logging, TinyCNNAdapter


class SquareModifier:
    """Modifies an image in-place by adding typical elements found when playing in chess.com"""

    LINE_PROPORTION: float = 0.2
    LINE_COLOR: tuple[float, float, float] = (0, 165, 255)
    LINE_ALPHA: float = 0.5
    CIRCLE_PROPORTION: float = 0.18
    CIRCLE_COLOR: tuple[float, float, float] = (60, 60, 60)
    CIRCLE_ALPHA: float = 0.5
    CIRCUMFERENCE_RADIUS_PROPORTION: float = 0.1
    CIRCUMFERENCE_COLOR: tuple[float, float, float] = (60, 60, 60)
    CIRCUMFERENCE_ALPHA: float = 0.5
    TRIANGLE_PROPORTION: float = 0.2
    TRIANGLE_COLOR: tuple[float, float, float] = (0, 165, 255)
    TRIANGLE_ALPHA: float = 0.5
    TRIANGLE_OFFSET: float = 1.4

    p_circle: float
    p_line: float
    p_arrow: float
    p_circum: float

    def __init__(self, p_circle: float, p_line: float, p_arrow: float, p_circum: float):
        assert p_circle + p_line + p_arrow + p_circum <= 1.0
        assert p_circle >= 0
        assert p_line >= 0
        assert p_arrow >= 0
        assert p_circum >= 0
        self.p_circle = p_circle
        self.p_line = p_line + p_circle
        self.p_arrow = p_arrow + p_line + p_circle
        self.p_circum = p_circum + p_arrow + p_line + p_circle

    def __call__(self, image: np.ndarray):
        image_shape = image.shape
        if len(image_shape) == 3:
            image_height, image_width, _ = image_shape
        elif len(image_shape) == 2:
            image_height, image_width = image_shape
        else:
            raise ValueError("Invalid image")

        r = random.random()
        if r < self.p_circle:
            self.add_circle(image, image_height, image_width)
        elif r < self.p_line:
            self.add_line(image, image_height, image_width)
        elif r < self.p_arrow:
            self.add_triangle(image, image_height, image_width)
        elif r < self.p_circum:
            self.add_circumference(image, image_height, image_width)

    def add_circle(self, image: np.ndarray, image_height: int, image_width: int):
        """Adds a circle as if you selected a piece to move"""
        overlay = image.copy()
        center = (image_width // 2, image_height // 2)
        radius = int(image_width * SquareModifier.CIRCLE_PROPORTION)
        cv2.circle(overlay, center, radius, SquareModifier.CIRCLE_COLOR, -1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, SquareModifier.CIRCLE_ALPHA, image, 1 - SquareModifier.CIRCLE_ALPHA, 0, image)

    def add_line(self, image: np.ndarray, image_height: int, image_width: int):
        """Adds the line of an arrow"""
        overlay = image.copy()
        r = random.random()
        if r < 0.25:  # horizontal line
            x1 = 0
            x2 = image_width
            image_height_half = image_height // 2
            y1 = image_height_half + int(image_height_half * SquareModifier.LINE_PROPORTION)
            y2 = image_height_half - int(image_height_half * SquareModifier.LINE_PROPORTION)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), SquareModifier.LINE_COLOR, -1, lineType=cv2.LINE_AA)
        elif r < 0.5:  # vertical line
            y1 = 0
            y2 = image_height
            image_width_half = image_width // 2
            x1 = image_width_half + int(image_width_half * SquareModifier.LINE_PROPORTION)
            x2 = image_width_half - int(image_width_half * SquareModifier.LINE_PROPORTION)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), SquareModifier.LINE_COLOR, -1, lineType=cv2.LINE_AA)
        elif r < 0.625:  # diagonal line tl-br
            x1 = 0
            y1 = 0
            x2 = image_width
            y2 = image_height
            thickness = int((image_width + image_height) / 2.0 * SquareModifier.LINE_PROPORTION)
            cv2.line(overlay, (x1, y1), (x2, y2), SquareModifier.LINE_COLOR, thickness, lineType=cv2.LINE_AA)
        elif r < 0.75:  # diagonal line tr-bl
            x1 = image_width
            y1 = 0
            x2 = 0
            y2 = image_height
            thickness = int((image_width + image_height) / 2.0 * SquareModifier.LINE_PROPORTION)
            cv2.line(overlay, (x1, y1), (x2, y2), SquareModifier.LINE_COLOR, thickness, lineType=cv2.LINE_AA)
        else:
            cy = image_height // 2
            cx = image_width // 2
            dy = int(cy * SquareModifier.LINE_PROPORTION)
            dx = int(cx * SquareModifier.LINE_PROPORTION)
            if r < 0.8125:  # top left
                cv2.rectangle(overlay, (cx - dx, 0), (cx + dx, cy + dy), SquareModifier.LINE_COLOR, -1, lineType=cv2.LINE_AA)
                cv2.rectangle(overlay, (0, cy - dy), (cx + dx, cy + dy), SquareModifier.LINE_COLOR, -1, lineType=cv2.LINE_AA)
            elif r < 0.875:  # top right
                cv2.rectangle(overlay, (cx - dx, 0), (cx + dx, cy + dy), SquareModifier.LINE_COLOR, -1, lineType=cv2.LINE_AA)
                cv2.rectangle(overlay, (cx - dx, cy - dy), (image_width, cy + dy), SquareModifier.LINE_COLOR, -1, lineType=cv2.LINE_AA)
            elif r < 0.9375:  # bottom right
                cv2.rectangle(overlay, (cx - dx, cy - dy), (cx + dx, image_height), SquareModifier.LINE_COLOR, -1, lineType=cv2.LINE_AA)
                cv2.rectangle(overlay, (cx - dx, cy - dy), (image_width, cy + dy), SquareModifier.LINE_COLOR, -1, lineType=cv2.LINE_AA)
            else:  # bottom left
                cv2.rectangle(overlay, (cx - dx, cy - dy), (cx + dx, image_height), SquareModifier.LINE_COLOR, -1, lineType=cv2.LINE_AA)
                cv2.rectangle(overlay, (0, cy - dy), (cx + dx, cy + dy), SquareModifier.LINE_COLOR, -1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, SquareModifier.LINE_ALPHA, image, 1 - SquareModifier.LINE_ALPHA, 0, image)

    def add_triangle(self, image: np.ndarray, image_height: int, image_width: int):
        """Adds the head of the arrow"""
        overlay = image.copy()
        r = random.random()

        cx = image_width // 2
        cy = image_height // 2

        base = int((image_width + image_height) / 2.0 * SquareModifier.TRIANGLE_PROPORTION)
        dx = int(cx * SquareModifier.TRIANGLE_PROPORTION)
        dy = int(cy * SquareModifier.TRIANGLE_PROPORTION)
        diag = int(base * SquareModifier.TRIANGLE_OFFSET)

        pt1 = (cx, cy)
        # RIGHT
        if r < 0.125:
            pt2 = (dx, cy - base)
            pt3 = (dx, cy + base)
            cv2.rectangle(
                overlay,
                (0, cy - dy),
                (dx, cy + dy),
                SquareModifier.TRIANGLE_COLOR,
                -1,
                lineType=cv2.LINE_AA,
            )
        # LEFT
        elif r < 0.25:
            pt2 = (image_width - dx, cy - base)
            pt3 = (image_width - dx, cy + base)
            cv2.rectangle(
                overlay,
                (image_width - dx, cy - dy),
                (image_width, cy + dy),
                SquareModifier.TRIANGLE_COLOR,
                -1,
                lineType=cv2.LINE_AA,
            )
        # DOWN
        elif r < 0.375:
            pt2 = (cx - base, dy)
            pt3 = (cx + base, dy)
            cv2.rectangle(
                overlay,
                (cx - dx, 0),
                (cx + dx, dy),
                SquareModifier.TRIANGLE_COLOR,
                -1,
                lineType=cv2.LINE_AA,
            )
        # UP
        elif r < 0.5:
            pt2 = (cx - base, image_height - dy)
            pt3 = (cx + base, image_height - dy)
            cv2.rectangle(
                overlay,
                (cx - dx, image_height - dy),
                (cx + dx, image_height),
                SquareModifier.TRIANGLE_COLOR,
                -1,
                lineType=cv2.LINE_AA,
            )
        # BR
        elif r < 0.625:
            pt2 = (diag, 0)
            pt3 = (0, diag)
            cv2.line(
                overlay,
                (0, 0),
                ((pt2[0] + pt3[0]) // 2, (pt2[1] + pt3[1]) // 2),
                SquareModifier.TRIANGLE_COLOR,
                int(base),
                lineType=cv2.LINE_AA,
            )
        # BL
        elif r < 0.75:
            pt2 = (image_width - diag, 0)
            pt3 = (image_width, diag)
            cv2.line(
                overlay,
                (image_width, 0),
                ((pt2[0] + pt3[0]) // 2, (pt2[1] + pt3[1]) // 2),
                SquareModifier.TRIANGLE_COLOR,
                int(base),
                lineType=cv2.LINE_AA,
            )
        # TR
        elif r < 0.875:
            pt2 = (0, image_height - diag)
            pt3 = (diag, image_height)
            cv2.line(
                overlay,
                (0, image_height),
                ((pt2[0] + pt3[0]) // 2, (pt2[1] + pt3[1]) // 2),
                SquareModifier.TRIANGLE_COLOR,
                int(base),
                lineType=cv2.LINE_AA,
            )
        # TL
        else:
            pt2 = (image_width - diag, image_height)
            pt3 = (image_width, image_height - diag)
            cv2.line(
                overlay,
                (image_width, image_height),
                ((pt2[0] + pt3[0]) // 2, (pt2[1] + pt3[1]) // 2),
                SquareModifier.TRIANGLE_COLOR,
                int(base),
                lineType=cv2.LINE_AA,
            )

        points = np.array([pt1, pt2, pt3], dtype=np.int32)
        cv2.fillPoly(overlay, [points], SquareModifier.TRIANGLE_COLOR, lineType=cv2.LINE_AA)
        cv2.addWeighted(
            overlay,
            SquareModifier.TRIANGLE_ALPHA,
            image,
            1 - SquareModifier.TRIANGLE_ALPHA,
            0,
            image,
        )


    def add_circumference(self, image: np.ndarray, image_height: int, image_width: int):
        overlay = image.copy()
        center = (image_width // 2, image_height // 2)
        radius = min(center)
        thickness = int(radius * SquareModifier.CIRCUMFERENCE_RADIUS_PROPORTION)
        radius -= thickness // 2
        cv2.circle(overlay, center, radius, SquareModifier.CIRCUMFERENCE_COLOR, thickness, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, SquareModifier.CIRCUMFERENCE_ALPHA, image, 1 - SquareModifier.CIRCUMFERENCE_ALPHA, 0, image)


def extract_squares(img: np.ndarray, bbox: tuple[float, float, float, float]) -> list[np.ndarray]:
    """
    Extracts 64 squares from the image based on the normalized bbox [ul_x, ul_y, lr_x, lr_y].
    Returns a list of 64 images (numpy arrays), ordered row by row (a8...h1 visually).
    """
    h_img, w_img = img.shape[:2]
    ul_x, ul_y, lr_x, lr_y = bbox

    # Denormalize coordinates
    x1 = int(ul_x * w_img)
    y1 = int(ul_y * h_img)
    x2 = int(lr_x * w_img)
    y2 = int(lr_y * h_img)

    # Ensure coordinates are within bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_img, x2)
    y2 = min(h_img, y2)

    board_w = x2 - x1
    board_h = y2 - y1

    if board_w <= 0 or board_h <= 0:
        return list()

    # Square dimensions
    sq_w = board_w / 8.0
    sq_h = board_h / 8.0

    squares = list()
    for row in range(8):
        for col in range(8):
            # Calculate integer coordinates for the square
            sx1 = int(x1 + col * sq_w)
            sy1 = int(y1 + row * sq_h)
            sx2 = int(x1 + (col + 1) * sq_w)
            sy2 = int(y1 + (row + 1) * sq_h)

            # Clamp
            sx1 = max(0, min(w_img, sx1))
            sy1 = max(0, min(h_img, sy1))
            sx2 = max(0, min(w_img, sx2))
            sy2 = max(0, min(h_img, sy2))

            if sx2 <= sx1 or sy2 <= sy1:
                # Fallback for degenerate squares (should not happen often)
                logging.error("Fallback happened!")
                squares.append(np.zeros((32, 32, 3), dtype=np.uint8))
            else:
                crop = img[sy1:sy2, sx1:sx2]
                squares.append(crop)

    return squares


def main():
    parser = argparse.ArgumentParser(description="Generate Pieces Dataset from Board Dataset using Trained Model")
    parser.add_argument("--input_ds", type=str, default="dataset_720/", help="Path to the input dataset (generated by gen_ds.py)")
    parser.add_argument("--model_path", type=str, default="out/base16/best_model.pth", help="Path to the trained TinyCNN model (.pth)")
    parser.add_argument("--output_dir", type=str, default="pieces_dataset", help="Output directory for the new dataset")
    parser.add_argument("--piece_size", type=int, default=64, help="Fixed size (WxH) for output piece images")

    args = parser.parse_args()

    setup_logging()
    logging.info(f"Starting piece dataset generation with config: {args}")

    if not os.path.exists(args.input_ds):
        logging.error(f"Input dataset not found: {args.input_ds}")
        return

    # Initialize Adapter
    try:
        adapter = TinyCNNAdapter(args.model_path)
    except Exception as e:
        logging.error(f"Failed to load model from {args.model_path}: {e}")
        return

    # Prepare output directories
    imgs_out_dir = os.path.join(args.output_dir, "images")
    os.makedirs(imgs_out_dir, exist_ok=True)

    ann_path = os.path.join(args.input_ds, "annotations.csv")
    if not os.path.exists(ann_path):
        logging.error(f"Annotations file not found: {ann_path}")
        return

    out_csv_path = os.path.join(args.output_dir, "pieces_annotations.csv")

    # Counters
    processed_boards = 0
    skipped_boards = 0
    total_pieces = 0
    total_empties = 0

    # Modifiers
    empty_modifier = SquareModifier(
        p_circle=0.166,
        p_line=0.166,
        p_arrow=0.166,
        p_circum=0.0
    )
    piece_modifier = SquareModifier(
        p_circle=0.0,
        p_line=0.166,
        p_arrow=0.166,
        p_circum=0.166
    )

    with open(ann_path, "r") as f_in, open(out_csv_path, "w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)
        writer.writerow(["file_name", "is_piece", "piece_type"])

        for row in tqdm(reader, desc="Processing Images"):
            # Only process if ground truth says there is a board
            if row["has_board"] != "1":
                continue

            img_id = row["image_id"]
            img_filename = f"image_{int(img_id):09d}.png"
            img_path = os.path.join(args.input_ds, "images", img_filename)

            if not os.path.exists(img_path):
                logging.warning(f"Image not found: {img_path}")
                continue

            # Read image
            img_orig = cv2.imread(img_path)
            if img_orig is None:
                logging.warning(f"Failed to read {img_path}")
                continue

            # Detect
            try:
                found, bbox = adapter.eval_from_img(img_orig)
            except Exception as e:
                logging.warning(f"Inference failed for {img_id}: {e}")
                continue

            if not found:
                skipped_boards += 1
                continue

            # Extract squares
            squares = extract_squares(img_orig, bbox)
            if len(squares) != 64:
                logging.warning(f"Failed to extract 64 squares for {img_id}")
                continue

            # Process each square
            for i, sq_img in enumerate(squares):
                label_str = row[f"sq{i}"]

                # Determine class
                is_piece = 1 if label_str else 0
                piece_type = label_str if label_str else "empty"

                sq_img = cv2.resize(sq_img, (args.piece_size, args.piece_size), interpolation=cv2.INTER_LINEAR)

                if piece_type == "empty":
                    empty_modifier(sq_img)
                else:
                    piece_modifier(sq_img)

                # Save crop
                out_filename = f"{int(img_id):09d}_{i:02d}.png"
                out_path = os.path.join(imgs_out_dir, out_filename)

                cv2.imwrite(out_path, sq_img)

                # Write annotation
                writer.writerow([out_filename, is_piece, piece_type])

                if is_piece:
                    total_pieces += 1
                else:
                    total_empties += 1

            processed_boards += 1

    logging.info(f"Done. Processed {processed_boards} boards. Skipped {skipped_boards} (not detected).")
    logging.info(f"Extracted {total_pieces} pieces and {total_empties} empty squares.")
    logging.info(f"Dataset saved to {args.output_dir}")


if __name__ == "__main__":
    main()
