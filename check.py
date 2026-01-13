#!/usr/bin/env python3


import pandas as pd
import cv2
import os
import argparse
import logging

from common import setup_logging


def verify_annotations(csv_path: str, img_dir: str, num_samples: int = 100):
    df = pd.read_csv(csv_path)
    # Only check images where a board actually exists
    df_positive = df[df["has_board"] == 1]

    samples = df_positive.sample(n=min(num_samples, len(df_positive)))

    i = 1
    for _, row in samples.iterrows():
        logging.info(f"Sample #{i}/{num_samples}")
        img_id = int(row["image_id"])
        img_path = os.path.join(img_dir, f"image_{img_id:09d}.png")

        img = cv2.imread(img_path)
        if img is None:
            logging.warning(f"Skipping {img_path}, not found.")
            continue

        h, w = img.shape[:2]

        # Convert percentages to pixel coordinates
        x1 = int(row["ul_x"] * w)
        y1 = int(row["ul_y"] * h)
        x2 = int(row["lr_x"] * w)
        y2 = int(row["lr_y"] * h)

        # Draw the box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"ID: {img_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Verification", img)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

        i += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Generate Pieces Dataset from Board Dataset using Trained Model")
    parser.add_argument("--csv_path", type=str, default="synthetic_chess_dataset/annotations.csv", help="Path to the input dataset CSV")
    parser.add_argument("--images_dir_path", type=str, default="synthetic_chess_dataset/images", help="Path to the input dataset images directory")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of images to check")

    args = parser.parse_args()

    verify_annotations(args.csv_path, args.images_dir_path, args.num_samples)
