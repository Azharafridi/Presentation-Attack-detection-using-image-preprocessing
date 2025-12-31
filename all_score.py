import sys
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def calculate_homogeneity(image):
    """
    Ref: Gonzalez & Woods, Ch. 12 (Table 12.1)
    Formula: sum(p(i,j) / (1 + |i-j|))
    Measures the closeness of the distribution of elements in the GLCM to the diagonal.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute normalized Gray-Level Co-occurrence Matrix
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return homogeneity

def canny_edge_detection(image):
    """
    Ref: Gonzalez & Woods, Ch. 10 (Section 10.2.6)
    Algorithm: Gaussian Smoothing -> Gradient Calculation -> Non-max Suppression -> Hysteresis.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 100 and 200 are the low and high hysteresis thresholds respectively
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges > 0) # Percentage of pixels that are edges
    return edge_density

def periodic_noise_score(image):
    """
    Ref: Gonzalez & Woods, Ch. 5 (Section 5.2.3)
    Analysis: Periodic noise manifests as impulsive spikes (bursts) in the Fourier Spectrum.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Analyze spikes far from the DC (center) component
    mask = dist_from_center > 10
    stats_region = magnitude_spectrum[mask]
    
    # Statistical outlier detection in the frequency domain
    threshold = np.mean(stats_region) + 4 * np.std(stats_region)
    peaks = (magnitude_spectrum > threshold) & mask
    
    # Returns the ratio of frequency spikes to total pixels
    return np.sum(peaks) / (h * w)

def frequency_burst_score(image):
    """
    Ref: Gonzalez & Woods, Ch. 4 (Filtering in the Frequency Domain)
    Captures localized high-frequency energy 'bursts' using the Laplacian variance.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return np.var(laplacian)

def color_anomaly_score(image):
    """
    Ref: Gonzalez & Woods, Ch. 6 (Section 6.7.2)
    Algorithm: Color vector distance. Measures Mahalanobis distance in RGB space.
    """
    pixels = image.reshape(-1, 3).astype(float)
    mean_vec = np.mean(pixels, axis=0)
    cov_mat = np.cov(pixels.T)
    
    try:
        inv_cov = np.linalg.inv(cov_mat)
        diff = pixels - mean_vec
        # D(z, m) = sqrt((z - m)^T C^-1 (z - m))
        mahalanobis = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
        return np.mean(np.sqrt(mahalanobis))
    except np.linalg.LinAlgError:
        return 0.0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test.py <image_path>")
        sys.exit(0)
    
    path = sys.argv[1]
    img = cv2.imread(path)
    if img is None:
        print(f"Error: Could not read {path}")
        sys.exit(0)
        
    print(f"Results for: {path}")
    print("-" * 35)
    print(f"Texture Homogeneity:    {calculate_homogeneity(img):.6f}")
    print(f"Canny Edge Density:     {canny_edge_detection(img):.6f}")
    print(f"Periodic Noise Score:   {periodic_noise_score(img):.6f}")
    print(f"Frequency Burst Score:  {frequency_burst_score(img):.4f}")
    print(f"Color Anomaly Score:    {color_anomaly_score(img):.6f}")