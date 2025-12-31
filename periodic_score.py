#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np


def pick_peaks(mag: np.ndarray, k: int = 12, min_dist: int = 18, center_r: int = 30):
    """
    Pick top-k peaks in FFT log-magnitude, ignoring a center radius (DC) and using min-distance suppression.
    Returns list of (y,x) in shifted FFT coords.
    """
    H, W = mag.shape
    cy, cx = H // 2, W // 2

    m = mag.copy()

    Y, X = np.ogrid[:H, :W]
    center_mask = ((Y - cy) ** 2 + (X - cx) ** 2) <= (center_r ** 2)
    m[center_mask] = 0.0

    # sort pixels by magnitude descending
    idxs = np.argsort(m.ravel())[::-1]

    peaks = []
    for idx in idxs:
        y = idx // W
        x = idx % W
        if m[y, x] <= 0:
            break

        ok = True
        for py, px in peaks:
            if (y - py) ** 2 + (x - px) ** 2 < (min_dist ** 2):
                ok = False
                break
        if ok:
            peaks.append((y, x))
            if len(peaks) >= k:
                break

    return peaks


def circular_mask(shape, centers, r):
    """Union of circles at centers (y,x) with radius r."""
    H, W = shape
    Y, X = np.ogrid[:H, :W]
    mask = np.zeros((H, W), dtype=bool)
    rr = r * r
    for (cy, cx) in centers:
        mask |= ((Y - cy) ** 2 + (X - cx) ** 2) <= rr
    return mask


def periodic_noise_score(
    gray: np.ndarray,
    k_peaks: int = 12,
    peak_r: int = 9,
    dc_r: int = 30,
    min_peak_dist: int = 18,
):
    """
    Compute Periodic Noise Score (PNS):
      PNS = (power in neighborhoods of strongest periodic peaks) / (total power),
    with DC/center region excluded and symmetric peak pairs included.

    Returns:
      pns (float), peaks (list[(y,x)]), peak_mask(bool ndarray), power_spectrum(float ndarray)
    """
    H, W = gray.shape

    # normalize
    g = gray.astype(np.float32) / 255.0

    # windowing reduces border artifacts
    win = np.outer(np.hanning(H), np.hanning(W)).astype(np.float32)
    g = g * win

    # FFT
    F = np.fft.fft2(g)
    Fsh = np.fft.fftshift(F)

    # power spectrum
    P = (np.abs(Fsh) ** 2).astype(np.float64)

    # log magnitude for peak detection
    mag = np.log1p(np.abs(Fsh)).astype(np.float32)

    # pick peaks
    peaks = pick_peaks(mag, k=k_peaks, min_dist=min_peak_dist, center_r=dc_r)

    # include symmetric counterparts (periodic noise peaks occur in pairs)
    cy, cx = H // 2, W // 2
    peaks_all = []
    for (py, px) in peaks:
        peaks_all.append((py, px))
        sy, sx = 2 * cy - py, 2 * cx - px
        if 0 <= sy < H and 0 <= sx < W:
            peaks_all.append((sy, sx))

    peak_mask = circular_mask((H, W), peaks_all, r=peak_r)

    # exclude DC/center from totals
    Y, X = np.ogrid[:H, :W]
    dc_mask = ((Y - cy) ** 2 + (X - cx) ** 2) <= (dc_r ** 2)

    total_power = P[~dc_mask].sum()
    peak_power = P[peak_mask & (~dc_mask)].sum()

    pns = float(peak_power / (total_power + 1e-12))
    return pns, peaks, peak_mask, mag


def save_debug_image(mag: np.ndarray, peaks, out_path: str):
    """
    Save a debug visualization: log-magnitude spectrum with peak markers.
    """
    H, W = mag.shape
    # normalize mag to 0..255
    m = mag.copy()
    m = (m - m.min()) / (m.max() - m.min() + 1e-12)
    vis = (m * 255).astype(np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    # draw peaks
    for (y, x) in peaks:
        cv2.circle(vis, (x, y), 6, (0, 0, 255), 2)

    cv2.imwrite(out_path, vis)


def main():
    if len(sys.argv) < 2:
        print("Usage: python periodic_noise_score.py <image_path> [--debug]")
        sys.exit(1)

    image_path = sys.argv[1]
    debug = "--debug" in sys.argv[2:]

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Could not read image: {image_path}")
        sys.exit(2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # auto-tune radii a bit based on size
    H, W = gray.shape
    dc_r = max(20, int(0.03 * min(H, W)))
    peak_r = max(6, int(0.01 * min(H, W)))
    min_dist = max(12, int(0.02 * min(H, W)))

    score, peaks, _, mag = periodic_noise_score(
        gray,
        k_peaks=12,
        peak_r=peak_r,
        dc_r=dc_r,
        min_peak_dist=min_dist,
    )

    print(f"Periodic Noise Score (PNS): {score:.6f}")
    print(f"Detected peak count: {len(peaks)}")
    print(f"DC radius={dc_r}, peak radius={peak_r}, min_peak_dist={min_dist}")

    if debug:
        in_dir = os.path.dirname(os.path.abspath(image_path))
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(in_dir, f"{base}_pns_debug.png")
        save_debug_image(mag, peaks, out_path)
        print(f"Saved debug spectrum: {out_path}")


if __name__ == "__main__":
    main()
