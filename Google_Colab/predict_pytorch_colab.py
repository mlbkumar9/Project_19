"""
U-Net Damage Predictor (PyTorch Version)

This script uses a pre-trained U-Net model to predict damage in new images.
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import cv2
import numpy as np
import os
import glob

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def predict_mask(model, image_path, device, target_size=(512, 512)):
    model.eval()
    image = cv2.imread(image_path)
    if image is None: return None, None
    original_size = (image.shape[1], image.shape[0])
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, target_size)
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_normalized.transpose((2, 0, 1))).float().unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = torch.sigmoid(output).cpu().numpy()[0, 0] > 0.5
    binary_mask = (pred_mask).astype(np.uint8) * 255
    binary_mask_resized = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)
    return binary_mask_resized, image

def main():
    # --------------------------------------------------------------------------
    #                    CHOOSE YOUR U-NET BACKBONE HERE
    # --------------------------------------------------------------------------
    # Below is a combined list of available backbones.
    # For this PyTorch script, use the PyTorch-compatible options.
    #
    #  - ResNet family:
    #      'resnet18', 'resnet34'                   # (PyTorch only)
    #      'resnet50', 'resnet101', 'resnet152'      # (PyTorch & Keras)
    #      'ResNet50V2', 'ResNet101V2', 'ResNet152V2'# (Keras only, mind the caps)
    #
    #  - VGG family:
    #      'vgg11', 'vgg13', 'vgg16', 'vgg19'        # (PyTorch & Keras, with/without '_bn')
    #
    #  - DenseNet family:
    #      'densenet121', 'densenet169', 'densenet201'# (PyTorch & Keras)
    #
    #  - MobileNet family:
    #      'mobilenet_v2'                           # (PyTorch: 'mobilenet_v2', Keras: 'MobileNetV2')
    #      'MobileNet'                              # (Keras only)
    #
    #  - EfficientNet family:
    #      'efficientnet-b0' through 'efficientnet-b7' # (PyTorch only)
    #
    BACKBONE = 'resnet50'  # <--- CHANGE THIS VALUE (use a PyTorch option)
    # --------------------------------------------------------------------------

    # --- Google Colab Configuration ---
    # IMPORTANT: Set this variable to the absolute path of your project directory in Google Colab.
    # If you upload the 'Project_19' folder directly, it might be '/content/Project_19'.
    # If you mount Google Drive, it might be '/content/drive/MyDrive/Project_19' or similar.
    COLAB_BASE_DIR = '/content/drive/MyDrive/1_Project_Files/Google_Colab/19_Project_19' # <--- ADJUST THIS PATH AS NEEDED

    # --- Configuration ---
    BASE_DIR = COLAB_BASE_DIR
    INPUT_DIR = os.path.join(BASE_DIR, 'Input_Images_To_Analyze')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'Google_Colab', 'Predictions', 'Pytorch', BACKBONE)
    MODEL_PATH = os.path.join(BASE_DIR, 'Google_Colab', 'Trained_Models', 'Pytorch', f'smp_unet_{BACKBONE}.pth')

    if not os.path.exists(MODEL_PATH): return

    os.makedirs(os.path.join(OUTPUT_DIR, 'Masks'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'Overlays'), exist_ok=True)

    device = get_device()
    model = smp.Unet(encoder_name=BACKBONE, encoder_weights="imagenet", in_channels=3, classes=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files: return

    for filename in image_files:
        image_path = os.path.join(INPUT_DIR, filename)
        binary_mask, original_image = predict_mask(model, image_path, device)
        if binary_mask is None: continue

        image_area = original_image.shape[0] * original_image.shape[1]
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # --- Final Confirmed Logic ---
        border_mask = create_border_mask(original_image.shape, 12)
        structure_mask_raw, _ = get_structure_mask_and_area(gray_image)
        structure_mask = cv2.bitwise_and(structure_mask_raw, border_mask)
        structure_area = cv2.countNonZero(structure_mask)
        if structure_area == 0: structure_area = image_area

        dent_area = analyze_dents_direct_hsv(original_image, structure_mask)
        dent_percent = (dent_area / structure_area) * 100

        binary_mask = cv2.bitwise_and(binary_mask, structure_mask)

        max_hole_area = get_largest_hole_area(binary_mask)
        total_hole_area = cv2.countNonZero(binary_mask)
        hole_percent_vs_total_image = (total_hole_area / image_area) * 100
        hole_percent_display = (max_hole_area / structure_area) * 100 if structure_area > 0 else 0

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

        print(f"Result: {category} ({severity}), Predicted Hole: {hole_percent_display:.2f}%, Dent: {dent_percent:.2f}%")

        # --- Save results ---
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'Masks', filename), binary_mask)
        overlay_image = original_image.copy()
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_image, contours, -1, (0, 0, 255), 2)
        
        cv2.putText(overlay_image, f"Category: {category} ({severity})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(overlay_image, f"Hole Area (predicted): {hole_percent_display:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay_image, f"Dent Area (detected): {dent_percent:.2f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imwrite(os.path.join(OUTPUT_DIR, 'Overlays', filename), overlay_image)

if __name__ == '__main__':
    main()
