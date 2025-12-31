import cv2
import numpy as np
import sys
import os

def get_distance_matrix(shape):
    M, N = shape
    u = np.arange(M)
    v = np.arange(N)
    u, v = np.meshgrid(u, v, indexing='ij')
    # Distance from the center (M/2, N/2)
    return np.sqrt((u - M/2)**2 + (v - N/2)**2)

def apply_filter(img, H):
    # Perform FFT and shift the zero frequency component to the center
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    
    # Apply filter
    res_shift = fshift * H
    
    # Inverse FFT
    f_ishift = np.fft.ifftshift(res_shift)
    img_back = np.fft.ifft2(f_ishift)
    
    # Return magnitude (absolute value) and scale to [0, 255]
    img_back = np.abs(img_back)
    return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def ideal_highpass(img, d0=30):
    D = get_distance_matrix(img.shape)
    H = np.ones(img.shape)
    H[D <= d0] = 0
    return apply_filter(img, H)

def butterworth_highpass(img, d0=30, n=2):
    D = get_distance_matrix(img.shape)
    # Handle division by zero at center
    with np.errstate(divide='ignore'):
        H = 1 / (1 + (d0 / D)**(2 * n))
    H[np.isinf(H)] = 0 # Center point handling
    return apply_filter(img, H)

def gaussian_highpass(img, d0=30):
    D = get_distance_matrix(img.shape)
    H = 1 - np.exp(-(D**2) / (2 * (d0**2)))
    return apply_filter(img, H)

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
    # Standard cutoff frequency (D0)
    # Note: Smaller D0 allows more frequencies, Larger D0 makes the image "sharper/noisier"
    D0 = 40 

    # 1. Ideal Highpass Filter
    ihpf_img = ideal_highpass(img, d0=D0)
    cv2.imwrite(f"ihpf-{img_name}", ihpf_img)
    print(f"Saved: ihpf-{img_name}")

    # 2. Butterworth Highpass Filter
    bhpf_img = butterworth_highpass(img, d0=D0, n=2)
    cv2.imwrite(f"bhpf-{img_name}", bhpf_img)
    print(f"Saved: bhpf-{img_name}")

    # 3. Gaussian Highpass Filter
    ghpf_img = gaussian_highpass(img, d0=D0)
    cv2.imwrite(f"ghpf-{img_name}", ghpf_img)
    print(f"Saved: ghpf-{img_name}")

if __name__ == "__main__":
    main()