import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.stats import kurtosis, skew
 
def get_robust_pad_verdict(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("Error: Image not found.")
        return
 
    # 1. IMAGE PREPARATION
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape
 
    # 2. FEATURE A: NOISE RESIDUAL ANALYSIS (Section 5.4)
    # We use a 3x3 median filter to estimate the "clean" image and subtract it.
    # The 'residual' should be random for real skin and structured for screens.
    smooth = cv2.medianBlur((gray * 255).astype(np.uint8), 3).astype(np.float32) / 255.0
    residual = gray - smooth
   
    # Calculate Residual Variance (Screens have high 'structured' variance)
    res_var = np.var(residual)
    res_kurt = kurtosis(residual.flatten())
 
    # 3. FEATURE B: MOIRÉ FREQUENCY SEARCH (Chapter 4)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    # Mask out the DC (center) and the low frequencies (natural content)
    cy, cx = h // 2, w // 2
    mag[cy-20:cy+20, cx-20:cx+20] = 0
    freq_max = np.max(mag)
    freq_mean = np.mean(mag)
    moire_index = freq_max / (freq_mean + 1e-6)
 
    # 4. FEATURE C: GRADIENT POLARITY (Chapter 10)
    # Real skin has soft, multidirectional gradients.
    # Screens have sharp, 1D/2D grid-aligned gradients.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    _, ang = cv2.cartToPolar(gx, gy)
    # Standard deviation of angles: Real = High (Random), Spoof = Low (Aligned)
    angle_std = np.std(ang)
 
    # --- ROBUST SCORING LOGIC ---
    # These thresholds are adjusted for normalized float32 images.
    scores = {
        "Residual_Noise": 1 if res_var > 0.0005 else 0,   # High noise = Spoof
        "Frequency_Peak": 1 if moire_index > 250 else 0, # High peak = Moiré
        "Gradient_Alignment": 1 if angle_std < 1.4 else 0 # Low std = Grid Alignment
    }
 
    total_score = sum(scores.values())
   
    # FINAL VERDICT
    if total_score >= 2:
        verdict = "SPOOF DETECTED"
    elif total_score == 1:
        verdict = "SUSPICIOUS / UNCERTAIN"
    else:
        verdict = "LIVE / GENUINE"
 
    # --- OUTPUT ---
    print(f"\n===== ROBUST PAD ANALYSIS: {os.path.basename(image_path)} =====")
    for k, v in scores.items():
        status = "FLAGGED" if v == 1 else "OK"
        print(f"{k:20}: {status}")
    print(f"Angle Std: {angle_std:.4f} | Moire Index: {moire_index:.2f}")
    print(f"VERDICT: {verdict}")
    print("="*50)
 
    # VISUALIZATION
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(img_rgb), plt.title("Original")
    plt.subplot(132), plt.imshow(np.abs(residual), cmap='hot'), plt.title("Noise Residual (Structured?)")
    plt.subplot(133), plt.imshow(ang, cmap='hsv'), plt.title("Gradient Directions")
    plt.tight_layout()
    plt.show()
 
if __name__ == "__main__":
    if len(sys.argv) > 1:
        get_robust_pad_verdict(sys.argv[1])
    else:
        print("Usage: python PAD.py image.jpg")