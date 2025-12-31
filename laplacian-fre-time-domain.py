import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

def get_fft_spectrum(gray_img):
    """Calculates the 2D magnitude spectrum (The 'Frequency Domain' part)."""
    f = np.fft.fft2(gray_img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return magnitude_spectrum

def apply_gabor_filter(img):
    """
    Acts as a 'Spatial-Frequency' bridge. 
    It captures specific frequencies at specific orientations.
    """
    # Parameters for screen line detection (adjust if needed)
    ksize = 31
    sigma = 4.0
    theta = np.pi / 4  # 45 degrees
    lambd = 10.0       # wavelength
    gamma = 0.5
    phi = 0
    
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, phi, ktype=cv2.CV_32F)
    return cv2.filter2D(img, cv2.CV_8U, kernel)

def process_image(input_path):
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        return

    # 1. Load Original
    img = cv2.imread(input_path)
    if img is None: return
    original_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Laplacian (Spatial Domain Detail)
    laplacian_raw = cv2.Laplacian(original_gray, cv2.CV_64F)
    laplacian_8bit = cv2.convertScaleAbs(laplacian_raw)

    # 3. Frequency Analysis (2D FFT)
    original_fft = get_fft_spectrum(original_gray)
    
    # 4. 'Time-Frequency' Style Analysis (Gabor/Spatial-Frequency)
    # This shows frequency energy localized in the spatial domain
    spatial_freq_map = apply_gabor_filter(original_gray)

    # --- PLOTTING ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Top Left: Original
    axes[0, 0].imshow(original_gray, cmap='gray')
    axes[0, 0].set_title('1. Original Spatial Domain (Time)')

    # Top Right: Frequency Domain
    axes[0, 1].imshow(original_fft, cmap='magma')
    axes[0, 1].set_title('2. 2D Frequency Domain (FFT Magnitude)')

    # Bottom Left: Laplacian
    axes[1, 0].imshow(laplacian_8bit, cmap='gray')
    axes[1, 0].set_title('3. Laplacian (High-Pass Detail)')

    # Bottom Right: Spatial-Frequency Bridge
    axes[1, 1].imshow(spatial_freq_map, cmap='jet')
    axes[1, 1].set_title('4. Localized Spatial-Frequency Map')

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()

    # Save logic
    directory = os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    output_path = os.path.join(directory, f"fake11-{filename}")
    
    plt.savefig(output_path)
    print(f"Success! Analysis saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test.py <image_name>")
    else:
        process_image(sys.argv[1])