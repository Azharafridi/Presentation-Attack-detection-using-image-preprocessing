import cv2
import numpy as np
import os
import sys
import json
import csv
import matplotlib
matplotlib.use("Agg")  # safe for headless runs
import matplotlib.pyplot as plt
from matplotlib import gridspec

# -----------------------------
# Stage 1: "Noise enhancement" methods
# -----------------------------
def white_top_hat(img_u8, kernel_size=15):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(img_u8, cv2.MORPH_TOPHAT, kernel)

def black_top_hat(img_u8, kernel_size=15):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(img_u8, cv2.MORPH_BLACKHAT, kernel)

def morphological_reconstruction_dilation(img_u8, marker_ksize=5, max_iters=512):
    """
    Grayscale reconstruction by dilation:
      marker = erode(img)
      iterate: marker = min(dilate(marker), mask=img) until convergence
    """
    img = img_u8.copy()
    k = np.ones((marker_ksize, marker_ksize), np.uint8)
    marker = cv2.erode(img, k)

    se = np.ones((3, 3), np.uint8)
    for _ in range(max_iters):
        prev = marker
        dil = cv2.dilate(marker, se)
        marker = np.minimum(dil, img)
        if np.array_equal(marker, prev):
            break
    return marker.astype(np.uint8)

def granulometry_map(img_u8):
    s1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    s2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    opening1 = cv2.morphologyEx(img_u8, cv2.MORPH_OPEN, s1)
    opening2 = cv2.morphologyEx(img_u8, cv2.MORPH_OPEN, s2)
    gran = cv2.absdiff(opening1, opening2)
    gran = cv2.normalize(gran, None, 0, 255, cv2.NORM_MINMAX)
    return gran.astype(np.uint8)

def apply_enhancement(gray_u8, method, kernel_size=15):
    method = method.lower()
    if method == "white":
        return white_top_hat(gray_u8, kernel_size), "white-tophat"
    if method == "black":
        return black_top_hat(gray_u8, kernel_size), "black-tophat"
    if method == "recon":
        return morphological_reconstruction_dilation(gray_u8, marker_ksize=5), "reconstruction"
    if method == "gran":
        return granulometry_map(gray_u8), "granulometry"
    raise ValueError("Unknown method. Use: white, black, recon, gran, all")

# -----------------------------
# Stage 2: FFT + radial profile
# -----------------------------
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

# -----------------------------
# Value method (one scalar per graph): normalized autocorrelation peak
# -----------------------------
def autocorr_peak_score(signal_1d, lag_min=2, lag_max_frac=0.25):
    x = np.asarray(signal_1d, dtype=np.float64).ravel()
    n = x.size
    if n < 8:
        return 0.0, 0

    x = x - np.mean(x)
    m = 1 << (2 * n - 1).bit_length()
    X = np.fft.rfft(x, n=m)
    ac = np.fft.irfft(X * np.conj(X), n=m)[:n]

    ac0 = ac[0] + 1e-12
    acn = ac / ac0

    lag_max = int(max(lag_min + 1, lag_max_frac * n))
    lag_max = min(lag_max, n - 1)

    region = acn[lag_min:lag_max + 1]
    if region.size == 0:
        return 0.0, 0

    idx = int(np.argmax(region))
    best_lag = lag_min + idx
    best_score = float(region[idx])
    return best_score, best_lag

# -----------------------------
# IMPORTANT CHANGE:
# Put text INSIDE the plot (top-right) so it always appears in saved PNG
# -----------------------------
def annotate_value_on_graph(ax, text):
    ax.text(
        0.98, 0.98, text,          # inside axes
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.6", alpha=0.95)
    )

def save_values_sidecar(out_dir, tag, base_name, values_dict):
    """
    Save as JSON + CSV next to the output figure.
    """
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, f"{tag}-{base_name}-values.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(values_dict, f, indent=2)

    csv_path = os.path.join(out_dir, f"{tag}-{base_name}-values.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "value"])
        for k, v in values_dict.items():
            w.writerow([k, v])

    return json_path, csv_path

# -----------------------------
# Main analysis+plot for one enhanced image
# -----------------------------
def analyze_and_plot(enhanced_u8, base_name, tag, out_dir):
    gray = enhanced_u8

    # FFT analysis
    orig_fft, _ = get_fft_spectrum(gray)
    radial_orig = get_radial_profile(orig_fft)

    # Laplacian image
    lap_raw = cv2.Laplacian(gray, cv2.CV_64F)
    lap_u8 = cv2.convertScaleAbs(lap_raw)

    # Lap FFT analysis
    lap_fft, _ = get_fft_spectrum(lap_u8)
    radial_lap = get_radial_profile(lap_fft)

    # Slices
    mid_row = gray[gray.shape[0] // 2, :]
    mid_col = gray[:, gray.shape[1] // 2]
    lap_mid_row = lap_u8[lap_u8.shape[0] // 2, :]

    # Compute the two scalar values you asked for
    a_score, a_lag = autocorr_peak_score(mid_row)       # A. Intensity Slice (mid-row)
    c_score, c_lag = autocorr_peak_score(lap_mid_row)   # C. Laplacian Intensity Slice (mid-row)

    # Print terminal
    print(f"[{tag}] A(IntensitySlice)  score={a_score:.6f} lag={a_lag}")
    print(f"[{tag}] C(LapIntensity)   score={c_score:.6f} lag={c_lag}")

    # --- Plot layout (same structure as your script) ---
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 3, 3, 1], wspace=0.3, hspace=0.3)

    def add_side_graphs(gs_cell, data_list, titles, is_polar=False, annotate_texts=None):
        inner_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_cell, hspace=0.4)
        for i, (data, title) in enumerate(zip(data_list, titles)):
            if is_polar and i == 0:
                ax = fig.add_subplot(inner_gs[i], projection='polar')
                theta = np.linspace(0, 2 * np.pi, len(data))
                ax.plot(theta, data, color='blue', linewidth=0.5)
            else:
                ax = fig.add_subplot(inner_gs[i])
                ax.plot(data, color='black', linewidth=0.8)
                ax.fill_between(range(len(data)), data, color='gray', alpha=0.2)

            ax.set_title(title, fontsize=9, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=7)

            # Put the value INSIDE the graph so it's saved for sure
            if annotate_texts and i < len(annotate_texts) and annotate_texts[i]:
                annotate_value_on_graph(ax, annotate_texts[i])

    # PANEL 1: Enhanced image + intensity slices (A gets value)
    add_side_graphs(
        gs[0, 0],
        [mid_row, mid_col],
        ['A. Intensity Slice', 'B. Horiz. Intensity Slice'],
        annotate_texts=[
            f"A_value={a_score:.6f}\nlag={a_lag}",
            ""
        ]
    )

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(gray, cmap='gray')
    ax1.set_title(f'1. Enhanced Image ({tag})', fontsize=14, pad=10)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(orig_fft, cmap='gray')
    ax2.set_title('2. Original Frequency Spectrum', fontsize=14, pad=10)
    ax2.axis('off')

    add_side_graphs(
        gs[0, 3],
        [radial_orig, radial_orig],
        ['B. Radial Frequency Profile', 'B. Radial Frequency Profile']
    )

    # PANEL 3: Laplacian image + laplacian slice (C gets value)
    polar_data = np.interp(
        np.linspace(0, 1, 100),
        np.linspace(0, 1, len(lap_mid_row)),
        lap_mid_row
    )

    add_side_graphs(
        gs[1, 0],
        [polar_data, lap_mid_row],
        ['1D Laplacian Profile', 'C. Laplacian Intensity Slice'],
        is_polar=True,
        annotate_texts=[
            "",
            f"C_value={c_score:.6f}\nlag={c_lag}"
        ]
    )

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.imshow(lap_u8, cmap='gray')
    ax3.set_title('3. Laplacian Image', fontsize=14, pad=10)
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.imshow(lap_fft, cmap='gray')
    ax4.set_title('4. Laplacian Frequency Spectrum', fontsize=14, pad=10)
    ax4.axis('off')

    add_side_graphs(
        gs[1, 3],
        [radial_lap, radial_lap],
        ['D. Radial Laplacian FFT Profile', 'D. Radial Laplacian FFT']
    )

    # SAVE FIGURE
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"{tag}-{base_name}-result.png")
    plt.savefig(out_png, dpi=150)  # no bbox tight needed now (text is inside axes)
    plt.close()

    # SAVE VALUES to sidecar files too
    values = {
        "A_value_autocorr_peak": float(a_score),
        "A_best_lag": int(a_lag),
        "C_value_autocorr_peak": float(c_score),
        "C_best_lag": int(c_lag),
    }
    json_path, csv_path = save_values_sidecar(out_dir, tag, base_name, values)

    return values, out_png, json_path, csv_path

# -----------------------------
# Entry point
# -----------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python pipeline_all_in_one.py <image_path> [method]")
        print("method: all | white | black | recon | gran")
        sys.exit(1)

    img_path = sys.argv[1]
    method = sys.argv[2].lower() if len(sys.argv) >= 3 else "all"

    if not os.path.exists(img_path):
        print(f"Error: File '{img_path}' not found.")
        sys.exit(1)

    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print("Error: Could not decode image.")
        sys.exit(1)

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    out_dir = os.path.join(os.path.dirname(img_path), "pipeline_outputs")

    methods = ["white", "black", "recon", "gran"] if method == "all" else [method]

    for m in methods:
        enhanced, tag = apply_enhancement(gray, m, kernel_size=15)

        enh_path = os.path.join(out_dir, f"{tag}-{base_name}.png")
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(enh_path, enhanced)

        values, fig_path, json_path, csv_path = analyze_and_plot(enhanced, base_name, tag, out_dir)

        print(f"[{tag}] saved enhanced: {enh_path}")
        print(f"[{tag}] saved figure  : {fig_path}")
        print(f"[{tag}] saved json   : {json_path}")
        print(f"[{tag}] saved csv    : {csv_path}")
        print(f"[{tag}] values       : {values}")
        print("-" * 60)

if __name__ == "__main__":
    main()
