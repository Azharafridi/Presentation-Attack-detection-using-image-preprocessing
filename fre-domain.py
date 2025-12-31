#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def radial_profile(mag: np.ndarray) -> np.ndarray:
    """
    Compute 1D radial mean profile of a 2D array (FFT magnitude).
    Returns profile where index = radius (in pixels from center).
    """
    H, W = mag.shape
    cy, cx = H // 2, W // 2

    y, x = np.indices((H, W))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)

    # mean for each radius bin
    tbin = np.bincount(r.ravel(), weights=mag.ravel())
    nr = np.bincount(r.ravel())
    profile = tbin / (nr + 1e-12)
    return profile


def main():
    if len(sys.argv) != 2:
        print("Usage: python fre-domain.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Could not read image: {image_path}")
        sys.exit(2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # FFT magnitude (shifted so DC is in the center)
    F = np.fft.fft2(gray)
    Fshift = np.fft.fftshift(F)
    mag = np.abs(Fshift)

    # log helps visualize peaks better
    mag_log = np.log1p(mag)

    prof = radial_profile(mag_log)

    # x-axis = radius in pixels (0..)
    x = np.arange(len(prof))

    # output path: same directory, prefix_graph.png
    in_dir = os.path.dirname(os.path.abspath(image_path))
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(in_dir, f"{base}_graph.png")

    plt.figure()
    plt.plot(x, prof)
    plt.title("1D Frequency Profile (Radial Avg of Log FFT Magnitude)")
    plt.xlabel("Frequency radius (pixels from center)")
    plt.ylabel("Mean log-magnitude")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved 1D frequency graph: {out_path}")


if __name__ == "__main__":
    main()
