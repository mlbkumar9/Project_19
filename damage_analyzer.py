
"""
Damage Analyzer (OpenCV Version)

This script serves two primary purposes:
1.  Direct Damage Analysis: It analyzes images in a specified directory to detect damage,
    which is defined as the presence of white pixels (pixel value > 240). It calculates
    the total area of this damage and classifies each image as 'Manageable',
    'Partially damaged', or 'Completely damaged' based on predefined area thresholds.
2.  Training Data Generation: For each analyzed image, it saves a corresponding
    black-and-white mask file. These masks show the exact location of the detected
    damage and are used as the ground truth for training a U-Net segmentation model.

The script generates two main outputs:
-   Annotated images saved in the 'Processed_Images' directory, showing the detected
    damage highlighted in red.
-   Mask files saved in the 'Masks' directory.
-   A CSV file (`damage_analysis_results.csv`) summarizing the analysis for all images.
"""
import cv2
import numpy as np
import os
import csv

def analyze_damage_area(image_path):
    """
    Analyzes a single image to calculate the area of damage based on white pixels.

    Args:
        image_path (str): The path to the image file.

    Returns:
        A tuple containing:
        - damage_area (int): The number of white pixels detected.
        - damage_mask (numpy array): A binary mask showing the location of the damage.
        - original_image (numpy array): The original BGR image that was loaded.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image at {image_path}. Skipping.")
            return None, None, None

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, damage_mask = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY)
        damage_area = cv2.countNonZero(damage_mask)

        return damage_area, damage_mask, image

    except Exception as e:
        print(f"An error occurred during image analysis for {image_path}: {e}")
        return None, None, None

def main():
    """
    Main function to run the damage analysis process on a directory of images.
    """
    # --- Configuration ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(BASE_DIR, 'RAW_Images')
    output_dir = os.path.join(BASE_DIR, 'Processed_Images')
    mask_dir = os.path.join(BASE_DIR, 'Masks')
    csv_output_file = os.path.join(BASE_DIR, 'damage_analysis_results.csv')

    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    results_data = []

    # --- Define Thresholds for Damage Classification ---
    MANAGEABLE_AREA_THRESHOLD = 5026
    PARTIALLY_DAMAGED_AREA_THRESHOLD = 17671

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found at {input_dir}")
        return

    print("Starting Damage Analysis (OpenCV Method)...")
    print("This script will also generate masks for ML model training.")
    print("-" * 40)

    # --- Process Each Image ---
    for filename in sorted(os.listdir(input_dir)):
        if filename.lower().endswith('.png'):
            image_path = os.path.join(input_dir, filename)
            
            damage_area, damage_mask, original_image = analyze_damage_area(image_path)

            if damage_area is not None:
                category = "No Damage Detected"
                if damage_area > PARTIALLY_DAMAGED_AREA_THRESHOLD:
                    category = "Completely damaged"
                elif damage_area > MANAGEABLE_AREA_THRESHOLD:
                    category = "Partially damaged"
                elif damage_area > 0:
                    category = "Manageable"

                annotated_image = original_image.copy()
                annotated_image[damage_mask != 0] = [0, 0, 255]
                cv2.putText(annotated_image, f"Category: {category}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(annotated_image, f"Area: {damage_area} px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, annotated_image)

                mask_path = os.path.join(mask_dir, filename)
                cv2.imwrite(mask_path, damage_mask)

                results_data.append({
                    'Filename': filename,
                    'Damage Category': category,
                    'Damage Area (pixels)': damage_area
                })

                print(f"File: {filename} -> Category: {category}, Area: {damage_area} pixels")

    # --- 5. Write results to CSV ---
    if results_data:
        print("-" * 40)
        print(f"Writing results to {csv_output_file}...")
        fieldnames = ['Filename', 'Damage Category', 'Damage Area (pixels)']
        try:
            with open(csv_output_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results_data)
            print("CSV file written successfully.")
        except PermissionError:
            print(f"Warning: Permission denied to write to {csv_output_file}. It may be open in another program.")

    print("Analysis complete.")

if __name__ == '__main__':
    main()
