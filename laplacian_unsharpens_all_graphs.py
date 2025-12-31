import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import gridspec

def get_fft_spectrum(gray_img):
    """Calculates the magnitude spectrum in the frequency domain."""
    f = np.fft.fft2(gray_img)
    fshift = np.fft.fftshift(f)
    # Use log scale to make the spectrum visible
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return magnitude_spectrum, fshift

def get_radial_profile(magnitude_spectrum):
    """Calculates the 1D radial average of the 2D FFT magnitude."""
    h, w = magnitude_spectrum.shape
    center = (w // 2, h // 2)
    y, x = np.indices((h, w))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)
    tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
    nr = np.bincount(r.ravel())
    nr[nr == 0] = 1
    return tbin / nr

def process_image(input_path):
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        return

    # 1. Load Original Image
    img = cv2.imread(input_path)
    if img is None:
        print("Error: Could not decode image.")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. FFT Analysis
    orig_fft, _ = get_fft_spectrum(gray)
    radial_orig = get_radial_profile(orig_fft)

    # 3. Laplacian Image (Method from Chapter 3)
    laplacian_raw = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_8bit = cv2.convertScaleAbs(laplacian_raw)

    # 4. Laplacian FFT Analysis
    lap_fft, _ = get_fft_spectrum(laplacian_8bit)
    radial_lap = get_radial_profile(lap_fft)

    # --- SETUP COMPLEX PLOT LAYOUT ---
    fig = plt.figure(figsize=(18, 12))
    # Grid: 2 Rows, 4 Columns with specific width ratios (SidePlots, Image, Image, SidePlots)
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 3, 3, 1], wspace=0.3, hspace=0.3)

    def add_side_graphs(gs_cell, data_list, titles, is_polar=False):
        """Helper to add two stacked 1D graphs into a single grid cell."""
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

    # --- PANEL 1: ORIGINAL IMAGE (Plots on Left) ---
    mid_row = gray[gray.shape[0] // 2, :]
    mid_col = gray[:, gray.shape[1] // 2]
    add_side_graphs(gs[0, 0], [mid_row, mid_col], ['A. Intensity Slice', 'B. Horiz. Intensity Slice'])
    
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(gray, cmap='gray')
    ax1.set_title('1. Original Image', fontsize=14, pad=10)
    ax1.axis('off')

    # --- PANEL 2: ORIGINAL FFT (Plots on Right) ---
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(orig_fft, cmap='gray')
    ax2.set_title('2. Original Frequency Spectrum', fontsize=14, pad=10)
    ax2.axis('off')
    
    add_side_graphs(gs[0, 3], [radial_orig, radial_orig], ['B. Radial Frequency Profile', 'B. Radial Frequency Profile'])

    # --- PANEL 3: LAPLACIAN IMAGE (Plots on Left) ---
    lap_mid_row = laplacian_8bit[laplacian_8bit.shape[0] // 2, :]
    # Generate a dummy polar profile for the visualization style in your image
    polar_data = np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(lap_mid_row)), lap_mid_row)
    
    add_side_graphs(gs[1, 0], [polar_data, lap_mid_row], ['1D Laplacian Profile', 'C. Laplacian Intensity Slice'], is_polar=True)
    
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.imshow(laplacian_8bit, cmap='gray')
    ax3.set_title('3. Laplacian Image', fontsize=14, pad=10)
    ax3.axis('off')

    # --- PANEL 4: LAPLACIAN FFT (Plots on Right) ---
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.imshow(lap_fft, cmap='gray')
    ax4.set_title('4. Laplacian Frequency Spectrum', fontsize=14, pad=10)
    ax4.axis('off')
    
    add_side_graphs(gs[1, 3], [radial_lap, radial_lap], ['D. Radial Laplacian FFT Profile', 'D. Radial Laplacian FFT'])

    # --- SAVE LOGIC ---
    directory = os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    # Ensure the prefix and directory path are handled correctly
    output_filename = f"fake11-{os.path.splitext(filename)[0]}-result.png"
    output_path = os.path.join(directory, output_filename)

    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Success! Comprehensive analysis saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test.py <image_name>")
    else:
        process_image(sys.argv[1])