import argparse
import cv2
import numpy as np


def detect_face_mask(gray_u8: np.ndarray, expand: float = 0.25) -> np.ndarray:
    """
    Returns a soft mask in [0..1] where 1 ~= face region (to suppress it later).
    Uses Haar cascade (ships with OpenCV).
    """
    H, W = gray_u8.shape
    face_mask = np.zeros((H, W), np.float32)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray_u8, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        return face_mask  # no face found -> no suppression

    # pick the largest face
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])

    # expand a bit
    ex = int(w * expand)
    ey = int(h * expand)
    x0 = max(0, x - ex)
    y0 = max(0, y - ey)
    x1 = min(W, x + w + ex)
    y1 = min(H, y + h + ey)

    # ellipse mask (soft)
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    ax = max(1, (x1 - x0) // 2)
    ay = max(1, (y1 - y0) // 2)

    cv2.ellipse(face_mask, (cx, cy), (ax, ay), 0, 0, 360, 1.0, -1)

    # soften edges
    k = int(0.03 * min(H, W))
    k = k + 1 if k % 2 == 0 else k
    if k >= 3:
        face_mask = cv2.GaussianBlur(face_mask, (k, k), 0)

    return face_mask


def pick_fft_peaks(mag: np.ndarray, num_peaks=10, min_dist=18, center_radius=25) -> list[tuple[int, int]]:
    """
    Pick strong peaks in FFT magnitude (shifted) away from center (DC),
    using a simple non-max suppression.
    """
    H, W = mag.shape
    cy, cx = H // 2, W // 2

    m = mag.copy()

    # remove DC / low-frequency center area
    Y, X = np.ogrid[:H, :W]
    center_mask = ((Y - cy) ** 2 + (X - cx) ** 2) <= (center_radius ** 2)
    m[center_mask] = 0.0

    # flatten indices by magnitude descending
    flat_idx = np.argsort(m.ravel())[::-1]

    peaks = []
    for idx in flat_idx:
        if len(peaks) >= num_peaks:
            break
        y = idx // W
        x = idx % W

        if m[y, x] <= 0:
            break

        # enforce min distance between peaks
        ok = True
        for py, px in peaks:
            if (y - py) ** 2 + (x - px) ** 2 < (min_dist ** 2):
                ok = False
                break
        if ok:
            peaks.append((y, x))

    return peaks


def notch_pass_mask(shape, peaks, r=8):
    """
    Build a notch-PASS mask: keeps only small circles around peaks and their symmetric counterparts.
    """
    H, W = shape
    cy, cx = H // 2, W // 2
    mask = np.zeros((H, W), np.float32)
    Y, X = np.ogrid[:H, :W]

    for (py, px) in peaks:
        # peak
        mask[((Y - py) ** 2 + (X - px) ** 2) <= r * r] = 1.0
        # symmetric w.r.t center
        sy, sx = 2 * cy - py, 2 * cx - px
        if 0 <= sy < H and 0 <= sx < W:
            mask[((Y - sy) ** 2 + (X - sx) ** 2) <= r * r] = 1.0

    return mask


def extract_screen_lines(
    img_bgr: np.ndarray,
    num_peaks=10,
    peak_radius=9,
    min_peak_dist=18,
    center_radius=28,
    suppress_face=True,
):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32) / 255.0
    H, W = gray.shape

    # windowing reduces border ringing
    win = np.outer(np.hanning(H), np.hanning(W)).astype(np.float32)
    gray_w = gray_f * win

    # FFT
    F = np.fft.fft2(gray_w)
    Fshift = np.fft.fftshift(F)
    mag = np.log1p(np.abs(Fshift)).astype(np.float32)

    # pick spectral peaks (periodic pattern spikes)
    peaks = pick_fft_peaks(mag, num_peaks=num_peaks, min_dist=min_peak_dist, center_radius=center_radius)

    # notch-PASS: keep only periodic spikes -> gives pattern map
    M = notch_pass_mask((H, W), peaks, r=peak_radius)
    F_filt = Fshift * M

    # inverse FFT -> line map
    f_ishift = np.fft.ifftshift(F_filt)
    line_map = np.fft.ifft2(f_ishift)
    line_map = np.abs(line_map)

    # normalize
    line_map = (line_map - line_map.min()) / (line_map.max() - line_map.min() + 1e-8)
    line_u8 = (line_map * 255).astype(np.uint8)

    # local contrast boost
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    line_u8 = clahe.apply(line_u8)

    # optionally suppress face area so you enhance mostly background screen lines
    if suppress_face:
        face_mask = detect_face_mask(gray, expand=0.25)  # 1 on face
        keep_mask = (1.0 - face_mask).astype(np.float32)  # 1 outside face
        line_u8 = (line_u8.astype(np.float32) * keep_mask).astype(np.uint8)

    # overlay for visualization
    overlay = cv2.addWeighted(gray, 1.0, line_u8, 0.85, 0)

    return line_u8, overlay, peaks


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Enhance/extract screen-line (moire/scanline) patterns using FFT notch-pass filtering."
    )

    # ✅ Positional input (what you want)
    ap.add_argument("input", nargs="?", help="Path to spoof image (positional)")

    # ✅ Optional --input still supported (backward compatible)
    ap.add_argument("--input", dest="input_opt", help="Path to spoof image (optional flag)")

    ap.add_argument("--line_out", default="line_map.png")
    ap.add_argument("--overlay_out", default="overlay.png")
    ap.add_argument("--num_peaks", type=int, default=10)
    ap.add_argument("--peak_radius", type=int, default=9)
    ap.add_argument("--no_face_suppress", action="store_true")

    args = ap.parse_args()

    input_path = args.input if args.input else args.input_opt
    if not input_path:
        ap.error("Please provide an input image path, e.g. python enhancing_screenlines.py /path/to/img.jpg")

    import cv2

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not read: {input_path}")

    line_map, overlay, peaks = extract_screen_lines(
        img,
        num_peaks=args.num_peaks,
        peak_radius=args.peak_radius,
        suppress_face=(not args.no_face_suppress),
    )

    cv2.imwrite(args.line_out, line_map)
    cv2.imwrite(args.overlay_out, overlay)

    print("Saved:", args.line_out, args.overlay_out)
    print("Detected FFT peaks (y,x):", peaks)


if __name__ == "__main__":
    main()
