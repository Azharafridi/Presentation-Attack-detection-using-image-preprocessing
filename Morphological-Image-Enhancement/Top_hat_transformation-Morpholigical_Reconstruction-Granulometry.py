import cv2
import numpy as np
import sys
import os
from skimage import morphology

def white_top_hat(img, kernel_size=15):
    # Enhances bright objects (lines) on a dark background
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

def black_top_hat(img, kernel_size=15):
    # Enhances dark objects (lines) on a bright background
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

def morphological_reconstruction(img):
    # Reconstruction by dilation: extracts marked features while preserving shapes
    # Create a marker (eroded version of the image)
    marker = cv2.erode(img, np.ones((5,5), np.uint8))
    # The mask is the original image
    reconstructed = morphology.reconstruction(marker, img, method='dilation')
    return reconstructed.astype(np.uint8)

def granulometry_map(img):
    # This applies successive openings with increasing kernel sizes 
    # and calculates the difference (surface area loss).
    # For visualization, we show a "residue" image highlighting specific structures.
    s1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    s2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    
    opening1 = cv2.morphologyEx(img, cv2.MORPH_OPEN, s1)
    opening2 = cv2.morphologyEx(img, cv2.MORPH_OPEN, s2)
    
    # The difference highlights structures that fit in s2 but not s1
    granulo_res = cv2.absdiff(opening1, opening2)
    return cv2.normalize(granulo_res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def main():
    if len(sys.argv) < 2:
        print("Usage: python morph_test.py <image_path>")
        return

    img_path = sys.argv[1]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return

    img_name = os.path.basename(img_path)

    # 1. White Top-Hat (Good if screen lines are brighter than surroundings)
    wth = white_top_hat(img)
    cv2.imwrite(f"white-tophat-{img_name}", wth)

    # 2. Black Top-Hat (Good if screen lines are darker pixel-grid gaps)
    bth = black_top_hat(img)
    cv2.imwrite(f"black-tophat-{img_name}", bth)

    # 3. Morphological Reconstruction
    recon = morphological_reconstruction(img)
    cv2.imwrite(f"reconstruction-{img_name}", recon)

    # 4. Granulometry-based contrast
    gran = granulometry_map(img)
    cv2.imwrite(f"granulometry-{img_name}", gran)

    print("Morphological processing complete.")

if __name__ == "__main__":
    main()