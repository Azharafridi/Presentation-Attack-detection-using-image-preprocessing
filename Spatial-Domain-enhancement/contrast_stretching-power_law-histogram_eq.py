import cv2
import numpy as np
import sys
import os

def contrast_stretching(img):
    # Formula: s = (r - min) * ( (L-1) / (max - min) )
    r_min = np.min(img)
    r_max = np.max(img)
    stretched = (img - r_min) * (255.0 / (r_max - r_min))
    return stretched.astype(np.uint8)

def power_law_gamma(img, gamma=0.4):
    # Formula: s = c * r^gamma
    # Normalize to [0, 1]
    normalized_img = img / 255.0
    gamma_corrected = np.power(normalized_img, gamma)
    # Scale back to [0, 255]
    return (gamma_corrected * 255.0).astype(np.uint8)

def histogram_equalization(img):
    # Standard Histogram Equalization
    return cv2.equalizeHist(img)

def main():
    if len(sys.argv) < 2:
        print("Usage: python test.py <image_path>")
        return

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"Error: File {img_path} not found.")
        return

    # Load image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not decode image.")
        return

    img_name = os.path.basename(img_path)
    
    # 1. Apply Contrast Stretching
    cs_img = contrast_stretching(img)
    cv2.imwrite(f"contrast-stretching-{img_name}", cs_img)
    print(f"Saved: contrast-stretching-{img_name}")

    # 2. Apply Gamma Transformation (gamma < 1 enhances dark details/lines)
    gamma_img = power_law_gamma(img, gamma=0.4)
    cv2.imwrite(f"power-law-gamma-{img_name}", gamma_img)
    print(f"Saved: power-law-gamma-{img_name}")

    # 3. Apply Histogram Equalization
    he_img = histogram_equalization(img)
    cv2.imwrite(f"histogram-equalization-{img_name}", he_img)
    print(f"Saved: histogram-equalization-{img_name}")

if __name__ == "__main__":
    main()