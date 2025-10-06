
import os
import cv2
import numpy as np
import tensorflow as tf
from keras_unet_collection import models
import sys

IMG_WIDTH = 512
IMG_HEIGHT = 512

def main(backbone):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_DIR = os.path.join(BASE_DIR, 'Input_Images_To_Analyze')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'Predictions', 'Keras', f'unet-plus_{backbone}')
    MODEL_PATH = os.path.join(BASE_DIR, 'Trained_Models', 'Keras', f'kuc_unet-plus_{backbone}.keras')

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}!")
        return

    os.makedirs(os.path.join(OUTPUT_DIR, 'Masks'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'Overlays'), exist_ok=True)

    print(f"--- Starting Keras Prediction: U-Net++ with '{backbone}' backbone ---")

    model = models.unet_plus_2d((IMG_HEIGHT, IMG_WIDTH, 3), 
                               filter_num=[64, 128, 256, 512],
                               n_labels=1, stack_num_down=2, stack_num_up=2,
                               activation='ReLU', output_activation='Sigmoid',
                               batch_norm=True, pool=True, unpool=True, 
                               backbone=backbone, weights=None,
                               name=f'unet-plus_{backbone}')
    model.load_weights(MODEL_PATH)

    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for filename in image_files:
        image_path = os.path.join(INPUT_DIR, filename)
        original_image = cv2.imread(image_path)
        if original_image is None: continue
        
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.resize(original_image_rgb, (IMG_WIDTH, IMG_HEIGHT))
        input_image = input_image / 255.0
        input_image = np.expand_dims(input_image, axis=0)

        predicted_mask = model.predict(input_image)[0]
        binary_mask = (predicted_mask > 0.5).astype(np.uint8) * 255
        binary_mask_resized = cv2.resize(binary_mask, (original_image.shape[1], original_image.shape[0]))

        damage_area = cv2.countNonZero(binary_mask_resized)

        # --- Classify Damage ---
        category = "No Damage Detected"
        if damage_area > 17671: # PARTIALLY_DAMAGED_AREA_THRESHOLD
            category = "Completely damaged"
        elif damage_area > 5026: # MANAGEABLE_AREA_THRESHOLD
            category = "Partially damaged"
        elif damage_area > 0:
            category = "Manageable"

        print(f"  Image: {filename}, Category: {category}, Predicted Damage Area: {damage_area} pixels")

        cv2.imwrite(os.path.join(OUTPUT_DIR, 'Masks', filename), binary_mask)
        overlay_image = original_image.copy()
        contours, _ = cv2.findContours(binary_mask_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_image, contours, -1, (0, 0, 255), 2)

        # --- Add Labels to Overlay ---
        cv2.putText(overlay_image, f"Category: {category}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(overlay_image, f"Area: {damage_area} px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imwrite(os.path.join(OUTPUT_DIR, 'Overlays', filename), overlay_image)

    print(f"--- Keras Prediction Complete. Results in {OUTPUT_DIR} ---")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python automated_predict_keras.py <backbone_name>")
        sys.exit(1)
    main(sys.argv[1])
