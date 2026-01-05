import cv2
import numpy as np
import os

def hsi_intensity_enhancement(img):
    # HLS: Hue, Lightness (Intensity), Saturation
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)
    l_enhanced = cv2.equalizeHist(l)
    enhanced_hls = cv2.merge([h, l_enhanced, s])
    return cv2.cvtColor(enhanced_hls, cv2.COLOR_HLS2BGR)

def pseudocolor_processing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Mapping gray levels to the JET spectrum to highlight periodic screen lines
    pseudo_img = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return pseudo_img

def color_balancing_tonal(img):
    # Lab color space allows adjustment of luminance and color separately
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

def process_folders():
    # Relative paths to go OUTSIDE current folder
    base_dirs = {
        "Fake": "../Fake",
        "Real": "../Real"
    }
    
    # Create output directory inside the current folder
    output_base = "Enhanced_Results"
    if not os.path.exists(output_base):
        os.makedirs(output_base)

    for prefix, folder_path in base_dirs.items():
        print(f"\n--- Checking Folder: {folder_path} ---")
        
        # Only process numbers 11 through 20 as requested
        for i in range(11, 21):
            filename = f"{prefix}{i}.jpg"
            filepath = os.path.join(folder_path, filename)
            
            if not os.path.exists(filepath):
                print(f"File not found: {filename}")
                continue

            img = cv2.imread(filepath)
            if img is None:
                print(f"Could not read: {filename}")
                continue

            # Create a dedicated output subfolder for this specific image
            img_output_dir = os.path.join(output_base, f"{prefix}{i}")
            if not os.path.exists(img_output_dir):
                os.makedirs(img_output_dir)

            # Apply Methods and Save
            cv2.imwrite(os.path.join(img_output_dir, f"hsi-{filename}"), hsi_intensity_enhancement(img))
            cv2.imwrite(os.path.join(img_output_dir, f"pseudo-{filename}"), pseudocolor_processing(img))
            cv2.imwrite(os.path.join(img_output_dir, f"balanced-{filename}"), color_balancing_tonal(img))
            
            print(f"Successfully processed: {filename}")

if __name__ == "__main__":
    process_folders()
    print("\nProcessing complete. Outputs are in 'color-image-enhancement/Enhanced_Results/'")