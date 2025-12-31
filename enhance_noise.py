#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np


def pick_fft_peaks(mag: np.ndarray, k: int = 14, min_dist: int = 18, center_r: int = 30):
    """
    Pick top-k peaks in FFT log-magnitude (shifted), ignoring DC center region.
    Simple non-max suppression using min_dist.
    Returns list of (y, x) peak locations.
    """
    H, W = mag.shape
    cy, cx = H // 2, W // 2

    m = mag.copy()

    Y, X = np.ogrid[:H, :W]
    dc_mask = ((Y - cy) ** 2 + (X - cx) ** 2) <= (center_r ** 2)
    m[dc_mask] = 0.0

    flat_idx = np.argsort(m.ravel())[::-1]
    peaks = []
    for idx in flat_idx:
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


def notch_pass_mask(shape, peaks, r: int = 10):
    """
    Notch-PASS mask: keeps only small circles around peaks and their symmetric counterparts.
    """
    H, W = shape
    cy, cx = H // 2, W // 2
    mask = np.zeros((H, W), np.float32)
    Y, X = np.ogrid[:H, :W]
    rr = r * r

    for py, px in peaks:
        # peak
        mask[((Y - py) ** 2 + (X - px) ** 2) <= rr] = 1.0
        # symmetric peak (periodic noise spikes come in pairs)
        sy, sx = 2 * cy - py, 2 * cx - px
        if 0 <= sy < H and 0 <= sx < W:
            mask[((Y - sy) ** 2 + (X - sx) ** 2) <= rr] = 1.0

    return mask


def enhance_periodic_noise(img_bgr: np.ndarray, k_peaks=14, peak_r=10, center_r=30, min_dist=18) -> np.ndarray:
    """
    Returns an image that highlights periodic noise (screen/grid/scanline patterns).
    Output is a single-channel uint8 image (0..255).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    H, W = gray.shape

    # Windowing helps reduce FFT border artifacts
    win = np.outer(np.hanning(H), np.hanning(W)).astype(np.float32)
    gray_w = gray * win

    # FFT
    F = np.fft.fft2(gray_w)
    Fsh = np.fft.fftshift(F)

    # Use log magnitude to detect spikes
    mag = np.log1p(np.abs(Fsh)).astype(np.float32)

    # Pick strong periodic peaks
    peaks = pick_fft_peaks(mag, k=k_peaks, min_dist=min_dist, center_r=center_r)

    # If no peaks found, fallback to a simple high-pass (still highlights fine patterns)
    if len(peaks) == 0:
        blur = cv2.GaussianBlur((gray * 255).astype(np.uint8), (0, 0), 7)
        hp = cv2.absdiff((gray * 255).astype(np.uint8), blur)
        return cv2.normalize(hp, None, 0, 255, cv2.NORM_MINMAX)

    # Notch-pass: keep only periodic spikes
    M = notch_pass_mask((H, W), peaks, r=peak_r)
    F_filt = Fsh * M

    # Inverse FFT -> periodic component map
    f_ishift = np.fft.ifftshift(F_filt)
    noise = np.fft.ifft2(f_ishift)
    noise = np.abs(noise)

    # Normalize + contrast boost (CLAHE)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    noise_u8 = (noise * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    noise_u8 = clahe.apply(noise_u8)

    # Optional sharpen to make lines pop a bit more
    noise_u8 = cv2.GaussianBlur(noise_u8, (0, 0), 0.8)
    noise_u8 = cv2.addWeighted(noise_u8, 1.6, cv2.GaussianBlur(noise_u8, (0, 0), 2.0), -0.6, 0)

    return noise_u8


def main():
    if len(sys.argv) != 2:
        print("Usage: python enhance_noise.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: could not read image: {image_path}")
        sys.exit(2)

    H, W = img.shape[:2]
    # light auto-tuning
    center_r = max(20, int(0.03 * min(H, W)))
    peak_r = max(6, int(0.012 * min(H, W)))
    min_dist = max(12, int(0.02 * min(H, W)))

    enhanced_noise = enhance_periodic_noise(
        img,
        k_peaks=14,
        peak_r=peak_r,
        center_r=center_r,
        min_dist=min_dist,
    )

    in_dir = os.path.dirname(os.path.abspath(image_path))
    base, ext = os.path.splitext(os.path.basename(image_path))
    out_path = os.path.join(in_dir, f"{base}_enhance_noise{ext}")

    cv2.imwrite(out_path, enhanced_noise)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
