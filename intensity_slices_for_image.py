#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_intensity_slices(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Center row (horizontal slice)
    mid_row = gray[gray.shape[0] // 2, :]

    # Center column (vertical slice)
    mid_col = gray[:, gray.shape[1] // 2]

    # Save path
    in_dir = os.path.dirname(os.path.abspath(image_path))
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(in_dir, f"{base}_slices.png")

    # Plot
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(mid_row, linewidth=1)
    plt.title("Center Row Intensity Slice (Horizontal Profile)")
    plt.xlabel("X (pixel index)")
    plt.ylabel("Intensity")

    plt.subplot(2, 1, 2)
    plt.plot(mid_col, linewidth=1)
    plt.title("Center Column Intensity Slice (Vertical Profile)")
    plt.xlabel("Y (pixel index)")
    plt.ylabel("Intensity")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()
    plt.close()

    print(f"Saved: {out_path}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python intensity_slices.py <image_path>")
        sys.exit(1)

    plot_intensity_slices(sys.argv[1])


if __name__ == "__main__":
    main()
