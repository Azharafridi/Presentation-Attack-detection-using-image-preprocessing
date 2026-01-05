import os
import json
import csv
import argparse

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec


# -----------------------------
# Color enhancement methods
# -----------------------------
def hsi_intensity_enhancement(img_bgr):
    # HLS: Hue, Lightness, Saturation
    hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)
    l_enhanced = cv2.equalizeHist(l)
    enhanced_hls = cv2.merge([h, l_enhanced, s])
    return cv2.cvtColor(enhanced_hls, cv2.COLOR_HLS2BGR)

def pseudocolor_processing(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

def color_balancing_tonal(img_bgr):
    # Lab color space adjustment
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    avg_a = np.average(lab[:, :, 1])
    avg_b = np.average(lab[:, :, 2])

    lab[:, :, 1] = lab[:, :, 1] - ((avg_a - 128.0) * (lab[:, :, 0] / 255.0) * 1.1)
    lab[:, :, 2] = lab[:, :, 2] - ((avg_b - 128.0) * (lab[:, :, 0] / 255.0) * 1.1)

    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# -----------------------------
# Metric 1: Normalized autocorrelation peak (NAP)
# -----------------------------
def autocorr_curve_normalized(x_1d):
    x = np.asarray(x_1d, dtype=np.float64).ravel()
    n = x.size
    if n < 8:
        return np.array([0.0], dtype=np.float64)

    x = x - np.mean(x)

    m = 1 << (2 * n - 1).bit_length()
    X = np.fft.rfft(x, n=m)
    ac = np.fft.irfft(X * np.conj(X), n=m)[:n]

    R0 = ac[0] + 1e-12
    return ac / R0  # normalized autocorrelation curve

def normalized_autocorr_peak(x_1d, tau_min=2, tau_max_frac=0.25):
    acn = autocorr_curve_normalized(x_1d)
    n = acn.size
    if n < 8:
        return 0.0, 0, acn

    tau_max = int(max(tau_min + 1, tau_max_frac * n))
    tau_max = min(tau_max, n - 1)

    region = acn[tau_min:tau_max + 1]
    if region.size == 0:
        return 0.0, 0, acn

    idx = int(np.argmax(region))
    best_tau = tau_min + idx
    S = float(region[idx])
    return S, best_tau, acn


# -----------------------------
# Metric 2: Cross Energy Ratio (CER)
# -----------------------------
def cross_energy_ratio(gray_u8, r0_frac=0.05, r1_frac=0.35, cross_halfwidth=3):
    """
    CER = E_cross / E_total

    E_total: sum |F(u,v)|^2 in a mid-frequency ring band
    E_cross: sum |F(u,v)|^2 only inside (thin horizontal + vertical bands) intersected with the same mid-band
    """
    img = gray_u8.astype(np.float64)
    img = img - np.mean(img)

    H, W = img.shape
    cy, cx = H // 2, W // 2

    F = np.fft.fftshift(np.fft.fft2(img))
    P = np.abs(F) ** 2  # |F(u,v)|^2

    # radial ring mask (mid frequencies)
    y, x = np.indices((H, W))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rmax = min(H, W) / 2.0
    r0 = r0_frac * rmax
    r1 = r1_frac * rmax
    band = (r >= r0) & (r <= r1)

    # cross mask (thin row+col bands through center)
    cross = np.zeros((H, W), dtype=bool)
    cross[max(0, cy - cross_halfwidth): min(H, cy + cross_halfwidth + 1), :] = True
    cross[:, max(0, cx - cross_halfwidth): min(W, cx + cross_halfwidth + 1)] = True

    total_mask = band
    cross_mask = band & cross

    E_total = float(np.sum(P[total_mask])) + 1e-12
    E_cross = float(np.sum(P[cross_mask]))
    return float(E_cross / E_total)


# -----------------------------
# Plot helper
# -----------------------------
def annotate(ax, text, fontsize=8):
    ax.text(
        0.98, 0.98, text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=fontsize,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.6", alpha=0.95),
    )


def make_three_method_figure(
    out_png_path,
    out_json_path,
    out_csv_path,
    base_title,
    methods_data,
):
    """
    methods_data: list of dicts with keys:
      name, bgr, gray, slice, nap_S, nap_tau, ac_curve, cer
    """
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, wspace=0.25, hspace=0.35)

    all_metrics = {}

    for col, item in enumerate(methods_data):
        name = item["name"]
        bgr = item["bgr"]
        sl = item["slice"]
        acn = item["ac_curve"]
        S = item["nap_S"]
        tau = item["nap_tau"]
        cer = item["cer"]

        # Row 0: enhanced image
        ax_img = fig.add_subplot(gs[0, col])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ax_img.imshow(rgb)
        ax_img.set_title(f"{col+1}. {name} Enhanced", fontsize=13, pad=10)
        ax_img.axis("off")
        annotate(ax_img, f"CER={cer:.6f}")

        # Row 1: intensity slice
        ax_sl = fig.add_subplot(gs[1, col])
        ax_sl.plot(sl, linewidth=0.9)
        ax_sl.fill_between(range(len(sl)), sl, alpha=0.2)
        ax_sl.set_title("Intensity Slice (mid-row)", fontsize=11)
        ax_sl.tick_params(labelsize=8)
        annotate(ax_sl, f"NAP S={S:.6f}\nτ={tau}")

        # Row 2: autocorr curve
        ax_ac = fig.add_subplot(gs[2, col])
        ax_ac.plot(acn, linewidth=0.9)
        ax_ac.axhline(0, linewidth=0.6)
        ax_ac.set_title("Normalized Autocorrelation (mid-row)", fontsize=11)
        ax_ac.set_xlim(0, len(acn) - 1)
        ax_ac.tick_params(labelsize=8)

        # store metrics
        all_metrics[f"{name}_CER"] = cer
        all_metrics[f"{name}_NAP_S"] = S
        all_metrics[f"{name}_NAP_tau"] = int(tau)

    fig.suptitle(base_title, fontsize=16, y=0.98)
    plt.savefig(out_png_path, dpi=160, bbox_inches="tight")
    plt.close()

    # Save JSON
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    # Save CSV
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in all_metrics.items():
            w.writerow([k, v])


# -----------------------------
# Main pipeline
# -----------------------------
def process_one_image(img_bgr, resize_to, tau_min, tau_max_frac, cer_r0, cer_r1, cer_cross_hw):
    if resize_to and resize_to > 0:
        img_bgr = cv2.resize(img_bgr, (resize_to, resize_to), interpolation=cv2.INTER_AREA)

    # three enhanced images
    hsi = hsi_intensity_enhancement(img_bgr)
    pseudo = pseudocolor_processing(img_bgr)
    balanced = color_balancing_tonal(img_bgr)

    results = []
    for name, bgr in [
        ("hsi", hsi),
        ("pseudo", pseudo),
        ("balanced", balanced),
    ]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mid_row = gray[gray.shape[0] // 2, :]

        S, tau, acn = normalized_autocorr_peak(
            mid_row, tau_min=tau_min, tau_max_frac=tau_max_frac
        )
        cer = cross_energy_ratio(
            gray, r0_frac=cer_r0, r1_frac=cer_r1, cross_halfwidth=cer_cross_hw
        )

        results.append({
            "name": name,
            "bgr": bgr,
            "gray": gray,
            "slice": mid_row,
            "nap_S": S,
            "nap_tau": tau,
            "ac_curve": acn,
            "cer": cer,
        })

    return hsi, pseudo, balanced, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake_dir", default="../Fake", help="Path to Fake folder (contains Fake11.jpg..Fake20.jpg)")
    ap.add_argument("--real_dir", default="../Real", help="Path to Real folder (contains Real11.jpg..Real20.jpg)")
    ap.add_argument("--out_dir", default="Enhanced_Results", help="Output directory inside current folder")
    ap.add_argument("--start", type=int, default=11)
    ap.add_argument("--end", type=int, default=20)
    ap.add_argument("--resize", type=int, default=512, help="Resize to NxN for stability (0 disables)")
    ap.add_argument("--tau_min", type=int, default=2)
    ap.add_argument("--tau_max_frac", type=float, default=0.25)

    # CER params
    ap.add_argument("--cer_r0", type=float, default=0.05)
    ap.add_argument("--cer_r1", type=float, default=0.35)
    ap.add_argument("--cer_cross_hw", type=int, default=3)

    args = ap.parse_args()

    out_base = args.out_dir
    os.makedirs(out_base, exist_ok=True)

    classes = [
        ("Fake", args.fake_dir),
        ("Real", args.real_dir),
    ]

    for prefix, folder_path in classes:
        print(f"\n--- Processing class: {prefix} | source: {folder_path} ---")

        for i in range(args.start, args.end + 1):
            filename = f"{prefix}{i}.jpg"
            filepath = os.path.join(folder_path, filename)

            if not os.path.exists(filepath):
                print(f"[WARN] File not found: {filepath}")
                continue

            img = cv2.imread(filepath)
            if img is None:
                print(f"[WARN] Could not read: {filepath}")
                continue

            # create output folder for this image
            img_out_dir = os.path.join(out_base, f"{prefix}{i}")
            os.makedirs(img_out_dir, exist_ok=True)

            # process + compute metrics
            hsi_img, pseudo_img, bal_img, methods_data = process_one_image(
                img_bgr=img,
                resize_to=args.resize,
                tau_min=args.tau_min,
                tau_max_frac=args.tau_max_frac,
                cer_r0=args.cer_r0,
                cer_r1=args.cer_r1,
                cer_cross_hw=args.cer_cross_hw,
            )

            # save enhanced images
            cv2.imwrite(os.path.join(img_out_dir, f"hsi-{filename}"), hsi_img)
            cv2.imwrite(os.path.join(img_out_dir, f"pseudo-{filename}"), pseudo_img)
            cv2.imwrite(os.path.join(img_out_dir, f"balanced-{filename}"), bal_img)

            # save combined figure + values
            out_png = os.path.join(img_out_dir, f"analysis-{prefix}{i}.png")
            out_json = os.path.join(img_out_dir, f"analysis-{prefix}{i}-values.json")
            out_csv = os.path.join(img_out_dir, f"analysis-{prefix}{i}-values.csv")

            title = f"{prefix}{i} — 3 Enhancements (No Laplacian) | Intensity Slice + NAP + CER"
            make_three_method_figure(out_png, out_json, out_csv, title, methods_data)

            # print key values to terminal
            print(f"[OK] {filename} -> {img_out_dir}")
            for md in methods_data:
                print(
                    f"   {md['name']:9s} | CER={md['cer']:.6f} | NAP_S={md['nap_S']:.6f} (tau={md['nap_tau']})"
                )

    print(f"\nDone. Outputs saved in: {os.path.abspath(out_base)}")


if __name__ == "__main__":
    main()
