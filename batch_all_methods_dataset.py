import os
import csv
import json
import argparse
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec


# =============================
# Enhancement methods (Stage 1)
# =============================
def white_top_hat(img_u8, kernel_size=15):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(img_u8, cv2.MORPH_TOPHAT, k)

def black_top_hat(img_u8, kernel_size=15):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(img_u8, cv2.MORPH_BLACKHAT, k)

def reconstruction_dilation(img_u8, marker_ksize=5, max_iters=512):
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

def get_enhancements(gray_u8, kernel_size=15):
    return [
        ("white-tophat", white_top_hat(gray_u8, kernel_size)),
        ("black-tophat", black_top_hat(gray_u8, kernel_size)),
        ("reconstruction", reconstruction_dilation(gray_u8, marker_ksize=5)),
        ("granulometry", granulometry_map(gray_u8)),
    ]


# =============================
# Graph helpers (Stage 2)
# =============================
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

def annotate_value_on_graph(ax, text, fontsize=7):
    ax.text(
        0.98, 0.98, text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=fontsize,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.6", alpha=0.95),
    )


# =============================
# Metrics (ALL methods)
# =============================
def autocorr_peak_score_1d(signal_1d, tau_min=2, tau_max_frac=0.25):
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
    return float(region[idx]), int(best_tau)

def fft_peakiness_1d(signal_1d, eps=1e-12):
    x = np.asarray(signal_1d, dtype=np.float64).ravel()
    if x.size < 8:
        return 0.0
    x = x - np.mean(x)
    F = np.fft.rfft(x)
    P = (np.abs(F) ** 2)
    if P.size <= 2:
        return 0.0
    P = P[1:]  # remove DC
    return float(np.max(P) / (np.median(P) + eps))

def cepstrum_peak_1d(signal_1d, q_min=2, q_max_frac=0.25, eps=1e-12):
    x = np.asarray(signal_1d, dtype=np.float64).ravel()
    n = x.size
    if n < 16:
        return 0.0, 0
    x = x - np.mean(x)

    F = np.fft.rfft(x)
    mag = np.abs(F) + eps
    log_mag = np.log(mag)
    ceps = np.fft.irfft(log_mag, n=n)

    q_max = int(max(q_min + 1, q_max_frac * n))
    q_max = min(q_max, n - 1)
    region = ceps[q_min:q_max + 1]
    if region.size == 0:
        return 0.0, 0

    idx = int(np.argmax(region))
    best_q = q_min + idx
    return float(region[idx]), int(best_q)

def autocorr_peak_ratio_2d(img_u8, center_exclusion=7, eps=1e-12):
    img = np.asarray(img_u8, dtype=np.float64)
    img = img - np.mean(img)

    F = np.fft.fft2(img)
    P = np.abs(F) ** 2
    ac = np.fft.ifft2(P).real
    ac = np.fft.fftshift(ac)

    h, w = ac.shape
    cy, cx = h // 2, w // 2
    center_val = ac[cy, cx] + eps

    mask = np.ones_like(ac, dtype=bool)
    y0, y1 = max(0, cy - center_exclusion), min(h, cy + center_exclusion + 1)
    x0, x1 = max(0, cx - center_exclusion), min(w, cx + center_exclusion + 1)
    mask[y0:y1, x0:x1] = False

    peak = float(np.max(ac[mask])) if np.any(mask) else 0.0
    return float(peak / center_val)

def spectral_peakiness_2d(img_u8, r0_frac=0.05, r1_frac=0.35, topk=20, eps=1e-12):
    img = np.asarray(img_u8, dtype=np.float64)
    img = img - np.mean(img)

    h, w = img.shape
    cy, cx = h // 2, w // 2
    rmax = min(h, w) / 2.0
    r0 = r0_frac * rmax
    r1 = r1_frac * rmax

    F = np.fft.fftshift(np.fft.fft2(img))
    P = np.abs(F) ** 2

    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    band = (r >= r0) & (r <= r1)

    vals = P[band].ravel()
    if vals.size < 10:
        return 0.0

    med = np.median(vals) + eps
    k = min(topk, vals.size)
    top_vals = np.partition(vals, -k)[-k:]
    return float(np.mean(top_vals) / med)

def spectral_entropy_2d(img_u8, r0_frac=0.05, r1_frac=0.35, eps=1e-12):
    img = np.asarray(img_u8, dtype=np.float64)
    img = img - np.mean(img)

    h, w = img.shape
    cy, cx = h // 2, w // 2
    rmax = min(h, w) / 2.0
    r0 = r0_frac * rmax
    r1 = r1_frac * rmax

    F = np.fft.fftshift(np.fft.fft2(img))
    P = np.abs(F) ** 2

    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    band = (r >= r0) & (r <= r1)

    vals = P[band].ravel()
    if vals.size < 10:
        return 0.0

    s = np.sum(vals) + eps
    p = vals / s
    H = -np.sum(p * np.log(p + eps))
    return float(H / (np.log(p.size + eps)))  # normalized

def spectral_flatness_2d(img_u8, r0_frac=0.05, r1_frac=0.35, eps=1e-12):
    img = np.asarray(img_u8, dtype=np.float64)
    img = img - np.mean(img)

    h, w = img.shape
    cy, cx = h // 2, w // 2
    rmax = min(h, w) / 2.0
    r0 = r0_frac * rmax
    r1 = r1_frac * rmax

    F = np.fft.fftshift(np.fft.fft2(img))
    P = np.abs(F) ** 2

    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    band = (r >= r0) & (r <= r1)

    vals = P[band].ravel()
    if vals.size < 10:
        return 0.0

    vals = vals + eps
    gm = np.exp(np.mean(np.log(vals)))
    am = np.mean(vals)
    return float(gm / (am + eps))

def orientation_coherence(img_u8, bins=36, eps=1e-12):
    img = np.asarray(img_u8, dtype=np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx * gx + gy * gy) + eps
    ang = np.arctan2(gy, gx)

    hist, _ = np.histogram(ang, bins=bins, range=(-np.pi, np.pi), weights=mag)
    return float(np.max(hist) / (np.mean(hist) + eps))

def haar_wavelet_dir_ratio(img_u8, eps=1e-12):
    img = np.asarray(img_u8, dtype=np.float64)
    h, w = img.shape
    h2 = (h // 2) * 2
    w2 = (w // 2) * 2
    if h2 < 4 or w2 < 4:
        return 0.0
    img = img[:h2, :w2]

    a = (img[:, 0::2] + img[:, 1::2]) * 0.5
    d = (img[:, 0::2] - img[:, 1::2]) * 0.5

    LL = (a[0::2, :] + a[1::2, :]) * 0.5
    LH = (a[0::2, :] - a[1::2, :]) * 0.5
    HL = (d[0::2, :] + d[1::2, :]) * 0.5
    HH = (d[0::2, :] - d[1::2, :]) * 0.5

    E_LH = np.sum(LH * LH)
    E_HL = np.sum(HL * HL)
    E_HH = np.sum(HH * HH)
    return float(max(E_LH, E_HL) / ((E_LH + E_HL + E_HH) + eps))

def banding_psd_peak(img_u8, axis=0, eps=1e-12):
    img = np.asarray(img_u8, dtype=np.float64)
    v = np.mean(img, axis=axis)
    v = v - np.mean(v)
    if v.size < 16:
        return 0.0
    F = np.fft.rfft(v)
    P = np.abs(F) ** 2
    if P.size <= 2:
        return 0.0
    P = P[1:]  # remove DC
    return float(np.max(P) / (np.sum(P) + eps))


# =============================
# Plot + compute metrics per (image, enhancement)
# =============================
def analyze_one(enhanced_u8, tag, out_dir_img, tau_min, tau_max_frac):
    gray = enhanced_u8

    orig_fft_vis, _ = get_fft_spectrum(gray)
    radial_orig = get_radial_profile(orig_fft_vis)

    lap_raw = cv2.Laplacian(gray, cv2.CV_64F)
    lap_u8 = cv2.convertScaleAbs(lap_raw)

    lap_fft_vis, _ = get_fft_spectrum(lap_u8)
    radial_lap = get_radial_profile(lap_fft_vis)

    mid_row = gray[gray.shape[0] // 2, :]
    mid_col = gray[:, gray.shape[1] // 2]
    lap_mid_row = lap_u8[lap_u8.shape[0] // 2, :]

    # 1D metrics on A and C
    S_A, tau_A = autocorr_peak_score_1d(mid_row, tau_min=tau_min, tau_max_frac=tau_max_frac)
    S_C, tau_C = autocorr_peak_score_1d(lap_mid_row, tau_min=tau_min, tau_max_frac=tau_max_frac)

    peak1d_A = fft_peakiness_1d(mid_row)
    peak1d_C = fft_peakiness_1d(lap_mid_row)

    ceps_A, qA = cepstrum_peak_1d(mid_row)
    ceps_C, qC = cepstrum_peak_1d(lap_mid_row)

    # 2D metrics (on Laplacian image)
    ac2d = autocorr_peak_ratio_2d(lap_u8)
    spk2d = spectral_peakiness_2d(lap_u8)
    sent = spectral_entropy_2d(lap_u8)
    sflat = spectral_flatness_2d(lap_u8)
    ocoh = orientation_coherence(lap_u8)
    wdir = haar_wavelet_dir_ratio(lap_u8)
    rowband = banding_psd_peak(lap_u8, axis=1)
    colband = banding_psd_peak(lap_u8, axis=0)

    # A simple combined score (optional)
    FINAL = float(
        0.30 * S_C +
        0.20 * spk2d +
        0.15 * ac2d +
        0.15 * ocoh +
        0.10 * wdir +
        0.10 * rowband
    )

    # --- Plot (same layout) ---
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
                annotate_value_on_graph(ax, annotate_texts[i], fontsize=7)

    # Write key values near A and C
    A_text = f"A\nAC={S_A:.4f}\nFFTpk={peak1d_A:.2f}\nCeps={ceps_A:.4f}"
    C_text = (
        f"C\nAC={S_C:.4f}\nSpk2D={spk2d:.3f}\nAC2D={ac2d:.3f}\n"
        f"OrC={ocoh:.2f}\nWav={wdir:.3f}\nRowB={rowband:.3f}\n"
        f"Ent={sent:.3f}\nFlat={sflat:.3f}\nFINAL={FINAL:.3f}"
    )

    add_side_graphs(
        gs[0, 0],
        [mid_row, mid_col],
        ["A. Intensity Slice", "B. Horiz. Intensity Slice"],
        annotate_texts=[A_text, ""],
    )

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(gray, cmap="gray")
    ax1.set_title(f"1. Enhanced Image ({tag})", fontsize=14, pad=10)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(orig_fft_vis, cmap="gray")
    ax2.set_title("2. Original Frequency Spectrum", fontsize=14, pad=10)
    ax2.axis("off")

    add_side_graphs(gs[0, 3], [radial_orig, radial_orig],
                    ["B. Radial Frequency Profile", "B. Radial Frequency Profile"])

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
        annotate_texts=["", C_text],
    )

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.imshow(lap_u8, cmap="gray")
    ax3.set_title("3. Laplacian Image", fontsize=14, pad=10)
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.imshow(lap_fft_vis, cmap="gray")
    ax4.set_title("4. Laplacian Frequency Spectrum", fontsize=14, pad=10)
    ax4.axis("off")

    add_side_graphs(gs[1, 3], [radial_lap, radial_lap],
                    ["D. Radial Laplacian FFT Profile", "D. Radial Laplacian FFT"])

    out_dir_img.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir_img / f"{tag}-result.png"
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close()

    metrics = {
        "A_autocorr_peak": float(S_A), "A_autocorr_tau": int(tau_A),
        "A_fft_peakiness_1d": float(peak1d_A),
        "A_cepstrum_peak": float(ceps_A), "A_cepstrum_q": int(qA),

        "C_autocorr_peak": float(S_C), "C_autocorr_tau": int(tau_C),
        "C_fft_peakiness_1d": float(peak1d_C),
        "C_cepstrum_peak": float(ceps_C), "C_cepstrum_q": int(qC),

        "Lap_acf2d_peak_ratio": float(ac2d),
        "Lap_spectral_peakiness_topk_over_median": float(spk2d),
        "Lap_spectral_entropy_norm": float(sent),
        "Lap_spectral_flatness": float(sflat),
        "Lap_orientation_coherence": float(ocoh),
        "Lap_wavelet_dir_ratio": float(wdir),
        "Lap_row_banding_psd_peak": float(rowband),
        "Lap_col_banding_psd_peak": float(colband),

        "FINAL_combo_score": float(FINAL),

        "tau_min": int(tau_min),
        "tau_max_frac": float(tau_max_frac),
    }

    return fig_path, metrics


# =============================
# Threshold selection
# =============================
def best_threshold(values, labels):
    """
    labels: 0=real, 1=fake
    Chooses direction automatically:
      if mean(fake) > mean(real): predict fake if value > t
      else predict fake if value < t
    Returns: (threshold, direction, acc, tpr, tnr)
    """
    v = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32)
    mask = np.isfinite(v)
    v = v[mask]
    y = y[mask]
    if v.size < 4:
        return None

    mean_fake = np.mean(v[y == 1]) if np.any(y == 1) else 0.0
    mean_real = np.mean(v[y == 0]) if np.any(y == 0) else 0.0
    direction = ">" if mean_fake > mean_real else "<"

    candidates = np.unique(v)
    best = (-1.0, None, None, None, None)  # (acc, thr, tpr, tnr, dir)
    for t in candidates:
        if direction == ">":
            pred = (v > t).astype(np.int32)
        else:
            pred = (v < t).astype(np.int32)

        tp = np.sum((pred == 1) & (y == 1))
        tn = np.sum((pred == 0) & (y == 0))
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))

        tpr = tp / (tp + fn + 1e-12)
        tnr = tn / (tn + fp + 1e-12)
        acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)

        youden = tpr + tnr - 1.0
        # pick by youden first, then acc
        score = (youden, acc)

        if best[1] is None:
            best = (acc, t, tpr, tnr, direction, youden)
        else:
            if score > (best[5], best[0]):
                best = (acc, t, tpr, tnr, direction, youden)

    acc, thr, tpr, tnr, direction, youden = best
    return thr, direction, float(acc), float(tpr), float(tnr), float(youden)


# =============================
# Dataset runner
# =============================
def find_image(folder: Path, name_no_ext: str):
    exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    for e in exts:
        p = folder / (name_no_ext + e)
        if p.exists():
            return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Dataset root containing Real/ and Fake/")
    ap.add_argument("--resize", type=int, default=512, help="Resize to NxN (0 disables)")
    ap.add_argument("--kernel", type=int, default=15, help="Kernel size for top-hat")
    ap.add_argument("--tau_min", type=int, default=2)
    ap.add_argument("--tau_max_frac", type=float, default=0.25)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    real_dir = root / "Real"
    fake_dir = root / "Fake"

    out_dir = root / "autocorr-pipeline_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build file list: Real11..20 and Fake11..20
    items = []
    for cls_name, folder, label in [("Real", real_dir, 0), ("Fake", fake_dir, 1)]:
        for i in range(11, 21):
            stem = f"{cls_name}{i}"
            p = find_image(folder, stem)
            if p is None:
                print(f"[WARN] Missing: {folder}/{stem}.(jpg/png)")
                continue
            items.append((p, label, cls_name, stem))

    if not items:
        print("No images found. Check folder structure: Real/Real11.jpg ... Fake/Fake11.jpg")
        return

    all_rows = []
    metric_keys = None

    for p, label, cls_name, stem in items:
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"[WARN] Could not decode: {p}")
            continue

        if args.resize and args.resize > 0:
            gray = cv2.resize(gray, (args.resize, args.resize), interpolation=cv2.INTER_AREA)

        # Per-image output folder
        img_out = out_dir / cls_name / stem
        img_out.mkdir(parents=True, exist_ok=True)

        # Save base gray (optional debug)
        cv2.imwrite(str(img_out / "00-input-gray.png"), gray)

        for tag, enhanced in get_enhancements(gray, kernel_size=args.kernel):
            enh_path = img_out / f"{tag}-enhanced.png"
            cv2.imwrite(str(enh_path), enhanced)

            fig_path, metrics = analyze_one(
                enhanced_u8=enhanced,
                tag=tag,
                out_dir_img=img_out,
                tau_min=args.tau_min,
                tau_max_frac=args.tau_max_frac
            )

            row = {
                "label": label,
                "class": cls_name,
                "image": stem,
                "enhancement": tag,
                "input_path": str(p),
                "enhanced_path": str(enh_path),
                "figure_path": str(fig_path),
            }
            row.update(metrics)

            if metric_keys is None:
                metric_keys = [k for k in metrics.keys()]

            all_rows.append(row)

        print(f"[OK] Processed {stem} ({cls_name})")

    if not all_rows:
        print("No rows produced (all images failed).")
        return

    # Save dataset metrics CSV
    metrics_csv = out_dir / "dataset_metrics.csv"
    fieldnames = list(all_rows[0].keys())
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print(f"\nSaved dataset metrics: {metrics_csv}")

    # Compute thresholds per metric (per enhancement, to keep it fair)
    report_rows = []
    best_overall = None  # (acc, method, metric, thr, dir, youden)

    for enh in sorted(set(r["enhancement"] for r in all_rows)):
        subset = [r for r in all_rows if r["enhancement"] == enh]
        labels = [r["label"] for r in subset]

        for mk in metric_keys:
            vals = [r.get(mk, np.nan) for r in subset]
            bt = best_threshold(vals, labels)
            if bt is None:
                continue
            thr, direction, acc, tpr, tnr, youden = bt

            report_rows.append({
                "enhancement": enh,
                "metric": mk,
                "direction_fake": direction,
                "threshold": thr,
                "accuracy": acc,
                "tpr_fake": tpr,
                "tnr_real": tnr,
                "youden": youden,
            })

            if best_overall is None or (youden, acc) > (best_overall[5], best_overall[0]):
                best_overall = (acc, enh, mk, thr, direction, youden)

    report_csv = out_dir / "threshold_report.csv"
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
        w.writeheader()
        for r in report_rows:
            w.writerow(r)

    print(f"Saved threshold report: {report_csv}")

    if best_overall:
        acc, enh, mk, thr, direction, youden = best_overall
        print("\nBest overall metric (by Youden, then Acc):")
        print(f"  enhancement : {enh}")
        print(f"  metric      : {mk}")
        print(f"  rule        : predict FAKE if value {direction} {thr:.6f}")
        print(f"  youden      : {youden:.4f}")
        print(f"  accuracy    : {acc:.4f}")

    # Save best rule to a json
    best_json = out_dir / "best_threshold_rule.json"
    if best_overall:
        acc, enh, mk, thr, direction, youden = best_overall
        payload = {
            "enhancement": enh,
            "metric": mk,
            "rule": f"fake if value {direction} threshold",
            "direction_fake": direction,
            "threshold": float(thr),
            "youden": float(youden),
            "accuracy": float(acc),
            "notes": "labels: 0=Real, 1=Fake. Rule direction chosen automatically using class means."
        }
        with open(best_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved best rule: {best_json}")


if __name__ == "__main__":
    main()
