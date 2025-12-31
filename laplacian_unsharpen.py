import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

def get_fft_spectrum(gray_img):
    """Calculates the magnitude spectrum in the frequency domain."""
    f = np.fft.fft2(gray_img)
    fshift = np.fft.fftshift(f)
    # Use log scale to make the spectrum visible
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return magnitude_spectrum

def process_image(input_path):
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        return

    # 1. Load Original Image
    img = cv2.imread(input_path)
    if img is None: return
    original_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Original Frequency Domain
    original_fft = get_fft_spectrum(original_gray)

    # 3. Laplacian Image (Method from Chapter 3)
    laplacian_raw = cv2.Laplacian(original_gray, cv2.CV_64F)
    laplacian_8bit = cv2.convertScaleAbs(laplacian_raw)

    # 4. Laplacian Frequency Domain
    laplacian_fft = get_fft_spectrum(laplacian_8bit)

    # --- PLOTTING & SAVING ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Row 1: Original
    axes[0, 0].imshow(original_gray, cmap='gray')
    axes[0, 0].set_title('1. Original Image')
    axes[0, 1].imshow(original_fft, cmap='gray')
    axes[0, 1].set_title('2. Original Frequency Spectrum')

    # Row 2: Laplacian
    axes[1, 0].imshow(laplacian_8bit, cmap='gray')
    axes[1, 0].set_title('3. Laplacian Image')
    axes[1, 1].imshow(laplacian_fft, cmap='gray')
    axes[1, 1].set_title('4. Laplacian Frequency Spectrum')

    # Remove ticks for a cleaner look
    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()

    # Save logic
    directory = os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    output_path = os.path.join(directory, f"fake11-{filename}")
    
    plt.savefig(output_path)
    print(f"Success! 4-panel analysis saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test.py <image_name>")
    else:
        process_image(sys.argv[1])