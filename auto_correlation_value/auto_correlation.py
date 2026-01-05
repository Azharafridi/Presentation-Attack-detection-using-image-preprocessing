import os
import sys
import json
import csv
import argparse

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

from skimage import morphology


def white_top_hat(img_u8, kernel_size=15):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(img_u8, cv2.MORPH_TOPHAT, kernel)

def black_top_hat(img_u8, kernel_size=15):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(img_u8, cv2.MORPH_BLACKHAT, kernel)

def morphological_reconstruction(img_u8):
    marker = cv2.erode(img_u8, np.ones((5, 5), np.uint8))
    reconstructed = morphology.reconstruction(marker, img_u8, method="dilation")
    return reconstructed.astype(np.uint8)

def granulometry_map(img_u8):
    s1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    s2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    opening1 = cv2.morphologyEx(img_u8, cv2.MORPH_OPEN, s1)
    opening2 = cv2.morphologyEx(img_u8, cv2.MORPH_OPEN, s2)
    granulo_res = cv2.absdiff(opening1, opening2)
    return cv2.normalize(granulo_res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def get_enhancements(gray_u8, kernel_size=15):
    return [
        ("white-tophat", white_top_hat(gray_u8, kernel_size)),
        ("black-tophat", black_top_hat(gray_u8, kernel_size)),
        ("reconstruction", morphological_reconstruction(gray_u8)),
        ("granulometry", granulometry_map(gray_u8)),
    ]


def get_fft_spectrum(gray_img_u8):
    f = np.fft.fft2(gray_img_u8)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return magnitude_spectrum, fshift

def get_radial_profile(magnitude_spectrum):
    h, w = magnitude_spectrum.shape
    center = (w // 2, h // 2)
    y, x = np.indices((h, w))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2).astype(int)
    tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
    nr = np.bincount(r.ravel())
    nr[nr == 0] = 1
    return tbin / nr


def autocorr_peak_score(signal_1d, tau_min=2, tau_max_frac=0.25):
    x = np.asarray(signal_1d, dtype=np.float64).ravel()
    n = x.size
    if n < 8:
        return 0.0, 0

    x = x - np.mean(x)

    m = 1 << (2 * n - 1).bit_length()
    X = np.fft.rfft(x, n=m)
    ac = np.fft.irfft(X * np.conj(X), n=m)[:n]

    R0 = ac[0] + 1e-12
    acn = ac / R0

    tau_max = int(max(tau_min + 1, tau_max_frac * n))
    tau_max = min(tau_max, n - 1)

    region = acn[tau_min:tau_max + 1]
    if region.size == 0:
        return 0.0, 0

    idx = int(np.argmax(region))
    best_tau = tau_min + idx
    S = float(region[idx])
    return S, best_tau


def annotate_value_on_graph(ax, text):
    ax.text(
        0.98, 0.98, text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.6", alpha=0.95),
    )

def save_sidecar(out_dir, tag, base_name, values_dict):
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, f"{tag}-{base_name}-values.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(values_dict, f, indent=2)

    csv_path = os.path.join(out_dir, f"{tag}-{base_name}-values.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for k, v in values_dict.items():
            w.writerow([k, v])

    return json_path, csv_path


def analyze_and_plot(enhanced_u8, tag, base_name, out_dir, tau_min, tau_max_frac):
    gray = enhanced_u8

    orig_fft, _ = get_fft_spectrum(gray)
    radial_orig = get_radial_profile(orig_fft)

    lap_raw = cv2.Laplacian(gray, cv2.CV_64F)
    lap_u8 = cv2.convertScaleAbs(lap_raw)

    lap_fft, _ = get_fft_spectrum(lap_u8)
    radial_lap = get_radial_profile(lap_fft)

    mid_row = gray[gray.shape[0] // 2, :]
    mid_col = gray[:, gray.shape[1] // 2]
    lap_mid_row = lap_u8[lap_u8.shape[0] // 2, :]

    S_A, tau_A = autocorr_peak_score(mid_row, tau_min=tau_min, tau_max_frac=tau_max_frac)
    S_C, tau_C = autocorr_peak_score(lap_mid_row, tau_min=tau_min, tau_max_frac=tau_max_frac)

    FINAL = S_C  # prefer C for thresholding

    print(f"[{tag}] S_A(IntensitySlice)  = {S_A:.6f} (tau={tau_A})")
    print(f"[{tag}] S_C(LapSlice)        = {S_C:.6f} (tau={tau_C})")
    print(f"[{tag}] FINAL (prefer C)     = {FINAL:.6f}")

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 3, 3, 1], wspace=0.3, hspace=0.3)

    def add_side_graphs(gs_cell, data_list, titles, is_polar=False, annotate_texts=None):
        inner_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_cell, hspace=0.4)
        for i, (data, title) in enumerate(zip(data_list, titles)):
            if is_polar and i == 0:
                ax = fig.add_subplot(inner_gs[i], projection="polar")
                theta = np.linspace(0, 2 * np.pi, len(data))
                ax.plot(theta, data, color="blue", linewidth=0.5)
            else:
                ax = fig.add_subplot(inner_gs[i])
                ax.plot(data, color="black", linewidth=0.8)
                ax.fill_between(range(len(data)), data, color="gray", alpha=0.2)

            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.tick_params(axis="both", which="major", labelsize=7)

            if annotate_texts and i < len(annotate_texts) and annotate_texts[i]:
                annotate_value_on_graph(ax, annotate_texts[i])

    add_side_graphs(
        gs[0, 0],
        [mid_row, mid_col],
        ["A. Intensity Slice", "B. Horiz. Intensity Slice"],
        annotate_texts=[f"S={S_A:.6f}\nτ={tau_A}", ""],
    )

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(gray, cmap="gray")
    ax1.set_title(f"1. Enhanced Image ({tag})", fontsize=14, pad=10)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(orig_fft, cmap="gray")
    ax2.set_title("2. Original Frequency Spectrum", fontsize=14, pad=10)
    ax2.axis("off")

    add_side_graphs(
        gs[0, 3],
        [radial_orig, radial_orig],
        ["B. Radial Frequency Profile", "B. Radial Frequency Profile"],
    )

    polar_data = np.interp(
        np.linspace(0, 1, 100),
        np.linspace(0, 1, len(lap_mid_row)),
        lap_mid_row
    )

    add_side_graphs(
        gs[1, 0],
        [polar_data, lap_mid_row],
        ["1D Laplacian Profile", "C. Laplacian Intensity Slice"],
        is_polar=True,
        annotate_texts=["", f"S={S_C:.6f}\nτ={tau_C}"],
    )

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.imshow(lap_u8, cmap="gray")
    ax3.set_title("3. Laplacian Image", fontsize=14, pad=10)
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.imshow(lap_fft, cmap="gray")
    ax4.set_title("4. Laplacian Frequency Spectrum", fontsize=14, pad=10)
    ax4.axis("off")

    add_side_graphs(
        gs[1, 3],
        [radial_lap, radial_lap],
        ["D. Radial Laplacian FFT Profile", "D. Radial Laplacian FFT"],
    )

    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, f"{tag}-{base_name}-result.png")
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close()

    values = {
        "S_A_intensity_slice": float(S_A),
        "tau_A": int(tau_A),
        "S_C_laplacian_slice": float(S_C),
        "tau_C": int(tau_C),
        "FINAL_prefer_C": float(FINAL),
        "tau_min": int(tau_min),
        "tau_max_frac": float(tau_max_frac),
    }
    json_path, csv_path = save_sidecar(out_dir, tag, base_name, values)
    return fig_path, json_path, csv_path, values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_path", help="Path to original image")
    ap.add_argument("--kernel", type=int, default=15, help="Kernel size for top-hat methods")
    ap.add_argument("--out", default=None,
                    help="Output directory (default: <image_dir>/autocorr-pipeline_output)")
    ap.add_argument("--resize", type=int, default=512,
                    help="Resize to NxN before processing (0 disables). Helps reduce resolution dependence.")
    ap.add_argument("--tau_min", type=int, default=2, help="tau_min for autocorr search")
    ap.add_argument("--tau_max_frac", type=float, default=0.25, help="tau_max = tau_max_frac * slice length")
    args = ap.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: file not found: {args.image_path}")
        sys.exit(1)

    gray = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print("Error: Could not decode image.")
        sys.exit(1)

    if args.resize and args.resize > 0:
        gray = cv2.resize(gray, (args.resize, args.resize), interpolation=cv2.INTER_AREA)

    base_name = os.path.splitext(os.path.basename(args.image_path))[0]

    # ✅ Default output directory name changed here:
    out_dir = args.out or os.path.join(os.path.dirname(args.image_path), "autocorr-pipeline_output")
    os.makedirs(out_dir, exist_ok=True)

    for tag, enhanced in get_enhancements(gray, kernel_size=args.kernel):
        enh_path = os.path.join(out_dir, f"{tag}-{base_name}-enhanced.png")
        cv2.imwrite(enh_path, enhanced)

        fig_path, json_path, csv_path, values = analyze_and_plot(
            enhanced, tag, base_name, out_dir,
            tau_min=args.tau_min, tau_max_frac=args.tau_max_frac
        )

        print(f"[{tag}] saved enhanced : {enh_path}")
        print(f"[{tag}] saved figure   : {fig_path}")
        print(f"[{tag}] saved values   : {json_path} , {csv_path}")
        print(f"[{tag}] values         : {values}")
        print("-" * 70)


if __name__ == "__main__":
    main()
