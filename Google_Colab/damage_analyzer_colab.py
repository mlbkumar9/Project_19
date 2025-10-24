"""
Damage Analyzer (OpenCV Version)

This script analyzes images to classify damage based on penetration (white pixels)
and dents (colored pixels). It also generates masks of the penetration areas for
use in training machine learning models.

"""
import cv2
import numpy as np
import os
import csv

def analyze_dents_direct_hsv(image, structure_mask):
    if image is None: return 0
    image_on_structure = cv2.bitwise_and(image, image, mask=structure_mask)
    hsv_image = cv2.cvtColor(image_on_structure, cv2.COLOR_BGR2HSV)
    lower_dent_range = np.array([25, 30, 30])
    upper_dent_range = np.array([100, 255, 255])
    dent_mask = cv2.inRange(hsv_image, lower_dent_range, upper_dent_range)
    return cv2.countNonZero(dent_mask)

def get_largest_hole_area(penetration_mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(penetration_mask, 4, cv2.CV_32S)
    if len(stats) < 2: return 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    return np.max(areas) if areas.size > 0 else 0

def get_structure_mask_and_area(gray_image):
    _, structure_mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return structure_mask, cv2.countNonZero(structure_mask)

def create_border_mask(image_shape, border_size):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (border_size, border_size), (w - border_size, h - border_size), 255, -1)
    return mask

def analyze_damage_and_dents(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None: return None, None, None

        image_area = image.shape[0] * image.shape[1]
        if image_area == 0: return None, None, None

        # --- Create a mask to ignore a 12-pixel border ---
        border_mask = create_border_mask(image.shape, 12)

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # --- CALCULATIONS ---
        structure_mask_raw, _ = get_structure_mask_and_area(gray_image)
        # Final structure mask excludes the border
        structure_mask = cv2.bitwise_and(structure_mask_raw, border_mask)
        structure_area = cv2.countNonZero(structure_mask)
        if structure_area == 0: structure_area = image_area

        dent_area = analyze_dents_direct_hsv(image, structure_mask)
        dent_percent = (dent_area / structure_area) * 100

        _, penetration_mask_raw = cv2.threshold(gray_image, 220, 255, cv2.THRESH_BINARY)
        penetration_mask = cv2.bitwise_and(penetration_mask_raw, border_mask)

        max_hole_area = get_largest_hole_area(penetration_mask)
        total_hole_area = cv2.countNonZero(penetration_mask)
        
        hole_percent_vs_total_image = (total_hole_area / image_area) * 100
        hole_percent_display = (max_hole_area / structure_area) * 100 if structure_area > 0 else 0

        # --- CLASSIFICATION ---
        MIN_SIGNIFICANT_HOLE_SIZE = 2500
        is_penetrated = max_hole_area > MIN_SIGNIFICANT_HOLE_SIZE
        is_severe_dent = dent_percent > 10

        if is_penetrated and is_severe_dent:
            category, severity = 'Penetrated and Dented', 'Completely Damaged'
        else:
            priority_is_penetration = is_penetrated and (hole_percent_vs_total_image >= 2)

            if priority_is_penetration:
                category, severity = 'Penetrated', 'Completely Damaged'
            else: # Priority is Dent
                if dent_percent > 70:
                    category, severity = 'Dented', 'Completely Damaged'
                elif is_penetrated:
                    category, severity = 'Penetrated', 'Completely Damaged'
                elif dent_area > 0:
                    category = 'Dented'
                    if dent_percent > 10: severity = 'Severe Dent'
                    elif dent_percent > 2: severity = 'Moderate Dent'
                    else: severity = 'Minor Dent'
                else:
                    category, severity = 'Fully Intact', '0% Damage'

        analysis = {
            'hole_percent_display': hole_percent_display,
            'dent_percent_display': dent_percent,
            'category': category,
            'severity': severity,
            'hole_area_main': max_hole_area,
            'dent_area': dent_area
        }
        
        return analysis, penetration_mask, image

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None

# --- Google Colab Configuration ---
# IMPORTANT: Set this variable to the absolute path of your project directory in Google Colab.
# If you upload the 'Project_19' folder directly, it might be '/content/Project_19'.
# If you mount Google Drive, it might be '/content/drive/MyDrive/Project_19' or similar.
COLAB_BASE_DIR = '/content/drive/MyDrive/1_Project_Files/Google_Colab/19_Project_19' # <--- ADJUST THIS PATH AS NEEDED

def main():
    # --- Configuration ---
    BASE_DIR = COLAB_BASE_DIR
    input_dir = os.path.join(BASE_DIR, 'RAW_Images')
    output_dir = os.path.join(BASE_DIR, 'Google_Colab', 'Processed_Images')
    mask_dir = os.path.join(BASE_DIR, 'Google_Colab', 'Masks')
    csv_output_file = os.path.join(BASE_DIR, 'Google_Colab', 'damage_analysis_results.csv')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    results_data = []
    if not os.path.isdir(input_dir): return

    for filename in sorted(os.listdir(input_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            
            analysis, penetration_mask, original_image = analyze_damage_and_dents(image_path)

            if analysis and original_image is not None:
                annotated_image = original_image.copy()
                
                if analysis['hole_area_main'] > 0:
                    annotated_image[penetration_mask != 0] = [0, 0, 255]

                cv2.putText(annotated_image, f"Category: {analysis['category']} ({analysis['severity']})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(annotated_image, f"Hole Area: {analysis['hole_percent_display']:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_image, f"Dent Area: {analysis['dent_percent_display']:.2f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imwrite(os.path.join(output_dir, filename), annotated_image)
                cv2.imwrite(os.path.join(mask_dir, filename), penetration_mask)

                results_data.append({
                    'Filename': filename,
                    'Category': analysis['category'],
                    'Severity': analysis['severity'],
                    'Hole_Area_Percent': f"{analysis['hole_percent_display']:.2f}",
                    'Dent_Area_Percent': f"{analysis['dent_percent_display']:.2f}",
                    'Hole_Area_Pixels_Main': analysis['hole_area_main'],
                    'Dent_Area_Pixels': analysis['dent_area']
                })

    if results_data:
        fieldnames = ['Filename', 'Category', 'Severity', 'Hole_Area_Percent', 'Dent_Area_Percent', 'Hole_Area_Pixels_Main', 'Dent_Area_Pixels']
        with open(csv_output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_data)

if __name__ == '__main__':
    main()