#!/usr/bin/env python3


import os
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Avoid cv2 conflicts
import matplotlib.pyplot as plt
import logging

from common import setup_logging


def plot_individual_metric(
    df: pd.DataFrame,
    metric_name: str,
    train_col: str,
    val_col: str,
    goal: str,
    experiment_name: str,
    output_folder: str
) -> None:
    """
    Generates and saves a single plot for a specific metric.
    """
    plt.figure(figsize=(8, 6))

    # Plot training and validation lines
    plt.plot(df["epoch"], df[train_col], label=f"Train {metric_name.upper()}", color="blue")
    plt.plot(df["epoch"], df[val_col], label=f"Val {metric_name.upper()}", color="orange")

    # Determine best values for the dotted reference lines
    best_t: float = df[train_col].min() if goal == "min" else df[train_col].max()  # type: ignore
    best_v: float = df[val_col].min() if goal == "min" else df[val_col].max()  # type: ignore

    # Add horizontal dotted lines with alpha 0.5
    plt.axhline(best_t, color="blue", linestyle="--", alpha=0.5, label=f"Best Train: {best_t:.4f}")
    plt.axhline(best_v, color="orange", linestyle="--", alpha=0.5, label=f"Best Val: {best_v:.4f}")

    plt.title(f"{experiment_name}: {metric_name.upper()}")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the individual plot
    file_name: str = f"{experiment_name}_{metric_name.lower()}.png"
    save_path: str = os.path.join(output_folder, file_name)
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Saved: {save_path}")


def process_history(csv_path: str, output_folder: str) -> None:
    """
    Reads history CSV and triggers individual plot generation for Loss, ACC, and MAE.
    """
    try:
        df: pd.DataFrame = pd.read_csv(csv_path)
    except Exception as e:
        logging.info(f"Could not read {csv_path}: {e}")
        return

    experiment_name: str = os.path.basename(os.path.dirname(csv_path))

    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Metric configuration: (Label, Train Col, Val Col, Goal)
    metrics: list[tuple[str, str, str, str]] = [
        ("loss", "train_loss", "val_loss", "min"),
        ("acc", "train_acc", "val_acc", "max"),
        ("mae", "train_mae", "val_mae", "min")
    ]

    for label, t_col, v_col, goal in metrics:
        try:
            plot_individual_metric(df, label, t_col, v_col, goal, experiment_name, output_folder)
        except KeyError as e:
            logging.error(f"Found KeyError with csv_path '{csv_path}', t_col '{t_col}', label '{label}', t_col '{t_col}', v_col '{v_col}', goal '{goal}'")
            logging.error(f"Error message: {e}")


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Generate separate PNG training curves from history.csv files.")
    parser.add_argument("--input_dir", type=str, default="./out/", help="Directory containing subdirectories with history.csv.")
    parser.add_argument("--output_dir", type=str, default="./curves/", help="Directory where the plot images will be saved.")

    args = parser.parse_args()

    input_path: str = args.input_dir
    output_path: str = args.output_dir

    if not os.path.isdir(input_path):
        logging.error(f"Error: {input_path} is not a valid directory.")
        return

    # Look for history.csv in one-level deep subdirectories
    subdirs: list[str] = [
        os.path.join(input_path, d) for d in os.listdir(input_path)
        if os.path.isdir(os.path.join(input_path, d))
    ]

    for subdir in subdirs:
        csv_file: str = os.path.join(subdir, "history.csv")
        if os.path.exists(csv_file):
            logging.info(f"Processing experiment: {os.path.basename(subdir)}")
            process_history(csv_file, output_path)


if __name__ == "__main__":
    main()
