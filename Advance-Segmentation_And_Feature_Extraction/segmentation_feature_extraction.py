import cv2
import numpy as np
import os

def hough_line_extraction(img):
    # Edge detection is a prerequisite for Hough Transform
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    # Create a copy to draw lines on
    line_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return line_img

def sift_feature_mapping(img):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    # Find keypoints and descriptors
    kp, des = sift.detectAndCompute(img, None)
    # Draw keypoints as small circles
    sift_img = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return sift_img

def harris_corner_detection(img):
    # Harris detection requires float32 input
    gray = np.float32(img)
    # dst = cv2.cornerHarris(src, blockSize, ksize, k)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    
    # Result is dilated for marking the corners
    dst = cv2.dilate(dst, None)
    
    # Threshold for an optimal value, it may vary depending on the image.
    out_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    out_img[dst > 0.01 * dst.max()] = [0, 0, 255] # Mark corners in Red
    return out_img

def process_folders():
    base_dirs = {
        "Fake": "../Fake",
        "Real": "../Real"
    }
    
    output_base = "Feature_Extraction_Results"
    if not os.path.exists(output_base):
        os.makedirs(output_base)

    for prefix, folder_path in base_dirs.items():
        print(f"\n--- Processing Features: {folder_path} ---")
        
        for i in range(11, 21):
            filename = f"{prefix}{i}.jpg"
            filepath = os.path.join(folder_path, filename)
            
            if not os.path.exists(filepath):
                continue

            # These methods work best on grayscale
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None: continue

            img_output_dir = os.path.join(output_base, f"{prefix}{i}")
            if not os.path.exists(img_output_dir):
                os.makedirs(img_output_dir)

            # 1. Hough Transform (Line Detection)
            cv2.imwrite(os.path.join(img_output_dir, f"hough-{filename}"), hough_line_extraction(img))

            # 2. SIFT (Feature Fingerprint)
            cv2.imwrite(os.path.join(img_output_dir, f"sift-{filename}"), sift_feature_mapping(img))

            # 3. Harris Corner (Grid Intersections)
            cv2.imwrite(os.path.join(img_output_dir, f"harris-{filename}"), harris_corner_detection(img))
            
            print(f"Features extracted for: {filename}")

if __name__ == "__main__":
    process_folders()