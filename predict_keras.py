import os
import cv2
import numpy as np
import tensorflow as tf
from keras_unet_collection import models

# --- Configuration ---
IMG_WIDTH = 512
IMG_HEIGHT = 512

# --------------------------------------------------------------------------
#      CHOOSE THE SAME U-NET BACKBONE THAT THE MODEL WAS TRAINED WITH
# --------------------------------------------------------------------------
# Below is a combined list of available backbones.
# For this Keras script, use the Keras-compatible options.
# Note: Capitalization matters for Keras backbones!
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
BACKBONE = 'ResNet50'  # <--- CHANGE THIS VALUE (use a Keras option)
# --------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'Input_Images_To_Analyze')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Predictions', 'Keras', f'unet-plus_{BACKBONE}')
MODEL_PATH = os.path.join(BASE_DIR, 'Trained_Models', 'Keras', f'kuc_unet-plus_{BACKBONE}.keras')

# --- Main Prediction Function ---
def main():
    """
    Loads a trained U-Net++ model (from keras-unet-collection) and predicts damage masks.
    """
    # --- 1. Load Model ---
    print(f"Attempting to load Keras U-Net++ model with '{BACKBONE}' backbone from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found! Please ensure the model has been trained with this configuration first.")
        return
    
    model = models.unet_plus_2d((IMG_HEIGHT, IMG_WIDTH, 3), 
                               filter_num=[64, 128, 256, 512],
                               n_labels=1, 
                               stack_num_down=2, 
                               stack_num_up=2,
                               activation='ReLU', 
                               output_activation='Sigmoid',
                               batch_norm=True, 
                               pool=True, 
                               unpool=True, 
                               backbone=BACKBONE, 
                               weights=None,
                               name=f'unet-plus_{BACKBONE}')
    model.load_weights(MODEL_PATH)
    print("Model loaded successfully.")

    # --- 2. Setup Output Directories ---
    output_mask_dir = os.path.join(OUTPUT_DIR, 'Masks')
    output_overlay_dir = os.path.join(OUTPUT_DIR, 'Overlays')
    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(output_overlay_dir, exist_ok=True)

    # --- 3. Define Thresholds ---
    MANAGEABLE_AREA_THRESHOLD = 5026
    PARTIALLY_DAMAGED_AREA_THRESHOLD = 17671

    # --- 4. Find and Process Images ---
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"No images found in {INPUT_DIR}. Please add images to analyze.")
        return

    print(f"\nFound {len(image_files)} images to analyze in {INPUT_DIR}.")

    for filename in image_files:
        try:
            print(f"--- Processing {filename} ---")
            image_path = os.path.join(INPUT_DIR, filename)
            
            original_image = cv2.imread(image_path)
            if original_image is None: continue
            
            original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            input_image = cv2.resize(original_image_rgb, (IMG_WIDTH, IMG_HEIGHT))
            input_image = input_image / 255.0
            input_image = np.expand_dims(input_image, axis=0)

            # --- 5. Predict ---
            predicted_mask = model.predict(input_image)[0]

            binary_mask = (predicted_mask > 0.5).astype(np.uint8) * 255
            binary_mask_resized = cv2.resize(binary_mask, (original_image.shape[1], original_image.shape[0]))

            # --- 6. Calculate Area and Classify ---
            damage_area = cv2.countNonZero(binary_mask_resized)
            
            category = "No Damage Detected"
            if damage_area > PARTIALLY_DAMAGED_AREA_THRESHOLD:
                category = "Completely damaged"
            elif damage_area > MANAGEABLE_AREA_THRESHOLD:
                category = "Partially damaged"
            elif damage_area > 0:
                category = "Manageable"

            print(f"Result: {category}, Area: {damage_area} pixels")

            # --- 7. Save Results ---
            mask_savename = os.path.join(output_mask_dir, filename)
            cv2.imwrite(mask_savename, binary_mask_resized)

            overlay_image = original_image.copy()
            contours, _ = cv2.findContours(binary_mask_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_image, contours, -1, (0, 0, 255), 2)
            
            cv2.putText(overlay_image, f"Category: {category}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(overlay_image, f"Area: {damage_area} px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            overlay_savename = os.path.join(output_overlay_dir, filename)
            cv2.imwrite(overlay_savename, overlay_image)

            print(f"Saved mask and labeled overlay for {filename}")

        except Exception as e:
            print(f"An error occurred while processing {filename}: {e}")

    print("\nPrediction complete.")
    print(f"Check the results in: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()