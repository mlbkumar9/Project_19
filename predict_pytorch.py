
"""
U-Net Damage Predictor (PyTorch Version)

This script uses a pre-trained U-Net model to predict damage in new images.
It uses the segmentation-models-pytorch library for robust and correct architectures.
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

# --- 1. Prediction Function ---
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
    """Main function to configure and run the prediction process."""
    # --------------------------------------------------------------------------
    #      CHOOSE THE SAME U-NET BACKBONE THAT THE MODEL WAS TRAINED WITH
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

    # --- Configuration ---
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    INPUT_DIR = os.path.join(BASE_DIR, 'Input_Images_To_Analyze')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'Predictions', 'Pytorch', BACKBONE)
    MODEL_PATH = os.path.join(BASE_DIR, 'Trained_Models', 'Pytorch', f'smp_unet_{BACKBONE}.pth')

    # --- Pre-run Checks ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}! Please ensure a model has been trained with this backbone.")
        return

    output_mask_dir = os.path.join(OUTPUT_DIR, 'Masks')
    output_overlay_dir = os.path.join(OUTPUT_DIR, 'Overlays')
    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(output_overlay_dir, exist_ok=True)

    MANAGEABLE_AREA_THRESHOLD = 5026
    PARTIALLY_DAMAGED_AREA_THRESHOLD = 17671

    print(f"Starting Damage Prediction with PyTorch U-Net model ('{BACKBONE}' backbone)...") 
    print("-" * 60)

    # --- Load Model ---
    device = get_device()
    print(f"Using device: {device}")
    
    model = smp.Unet(
        encoder_name=BACKBONE,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # --- Process Images ---
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"No images found in {INPUT_DIR}. Please add images to analyze.")
        return

    print(f"\nFound {len(image_files)} images to analyze in {INPUT_DIR}.")

    for filename in image_files:
        print(f"--- Processing {filename} ---")
        image_path = os.path.join(INPUT_DIR, filename)
        
        binary_mask, original_image = predict_mask(model, image_path, device)
        
        if binary_mask is None: continue

        damage_area = np.count_nonzero(binary_mask)
        category = "No Damage Detected"
        if damage_area > PARTIALLY_DAMAGED_AREA_THRESHOLD:
            category = "Completely damaged"
        elif damage_area > MANAGEABLE_AREA_THRESHOLD:
            category = "Partially damaged"
        elif damage_area > 0:
            category = "Manageable"

        print(f"Result: {category}, Area: {damage_area} pixels")

        mask_savename = os.path.join(output_mask_dir, filename)
        cv2.imwrite(mask_savename, binary_mask)

        overlay_image = original_image.copy()
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_image, contours, -1, (0, 0, 255), 2)
        
        cv2.putText(overlay_image, f"Category: {category}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(overlay_image, f"Area: {damage_area} px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        overlay_savename = os.path.join(output_overlay_dir, filename)
        cv2.imwrite(overlay_savename, overlay_image)

        print(f"Saved mask and labeled overlay for {filename}")

    print("-" * 60)
    print("Prediction complete.")
    print(f"Check the results in: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
