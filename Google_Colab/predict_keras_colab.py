import os
import cv2
import numpy as np
import tensorflow as tf
from keras_unet_collection import models

# Configure TensorFlow to use GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU is available and configured for TensorFlow.")
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
else:
    print("No GPU available, TensorFlow will use CPU.")

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

# --------------------------------------------------------------------------
#                    CHOOSE YOUR U-NET BACKBONE HERE
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
IMG_WIDTH = 512
IMG_HEIGHT = 512

# --- Google Colab Configuration ---
# IMPORTANT: Set this variable to the absolute path of your project directory in Google Colab.
# If you upload the 'Project_19' folder directly, it might be '/content/Project_19'.
# If you mount Google Drive, it might be '/content/drive/MyDrive/Project_19' or similar.
COLAB_BASE_DIR = '/content/drive/MyDrive/1_Project_Files/Google_Colab/19_Project_19' # <--- ADJUST THIS PATH AS NEEDED

# --- Configuration ---
BASE_DIR = COLAB_BASE_DIR
INPUT_DIR = os.path.join(BASE_DIR, 'Input_Images_To_Analyze')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Google_Colab', 'Predictions', 'Keras', f'unet-plus_{BACKBONE}')
MODEL_PATH = os.path.join(BASE_DIR, 'Google_Colab', 'Trained_Models', 'Keras', f'kuc_unet-plus_{BACKBONE}.keras')

def main():
    if not os.path.exists(MODEL_PATH): return

    model = models.unet_plus_2d((IMG_HEIGHT, IMG_WIDTH, 3), filter_num=[64, 128, 256, 512],
                               n_labels=1, stack_num_down=2, stack_num_up=2,
                               activation='ReLU', output_activation='Sigmoid',
                               batch_norm=True, pool=True, unpool=True, backbone=BACKBONE, 
                               weights=None, name=f'unet-plus_{BACKBONE}')
    model.load_weights(MODEL_PATH)

    output_mask_dir = os.path.join(OUTPUT_DIR, 'Masks')
    output_overlay_dir = os.path.join(OUTPUT_DIR, 'Overlays')
    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(output_overlay_dir, exist_ok=True)

    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files: return

    for filename in image_files:
        try:
            image_path = os.path.join(INPUT_DIR, filename)
            original_image = cv2.imread(image_path)
            if original_image is None: continue
            
            image_area = original_image.shape[0] * original_image.shape[1]
            gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

            input_image = cv2.resize(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), (IMG_WIDTH, IMG_HEIGHT)) / 255.0
            input_image = np.expand_dims(input_image, axis=0)

            predicted_mask = model.predict(input_image)[0]
            binary_mask_resized = (cv2.resize(predicted_mask, (original_image.shape[1], original_image.shape[0])) > 0.5).astype(np.uint8) * 255

            # --- Final Confirmed Logic ---
            border_mask = create_border_mask(original_image.shape, 12)
            structure_mask_raw, _ = get_structure_mask_and_area(gray_image)
            structure_mask = cv2.bitwise_and(structure_mask_raw, border_mask)
            structure_area = cv2.countNonZero(structure_mask)
            if structure_area == 0: structure_area = image_area

            dent_area = analyze_dents_direct_hsv(original_image, structure_mask)
            dent_percent = (dent_area / structure_area) * 100

            binary_mask_resized = cv2.bitwise_and(binary_mask_resized, structure_mask)

            max_hole_area = get_largest_hole_area(binary_mask_resized)
            total_hole_area = cv2.countNonZero(binary_mask_resized)
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
            cv2.imwrite(os.path.join(output_mask_dir, filename), binary_mask_resized)
            overlay_image = original_image.copy()
            contours, _ = cv2.findContours(binary_mask_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_image, contours, -1, (0, 0, 255), 2)
            
            cv2.putText(overlay_image, f"Category: {category} ({severity})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(overlay_image, f"Hole Area (predicted): {hole_percent_display:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay_image, f"Dent Area (detected): {dent_percent:.2f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imwrite(os.path.join(output_overlay_dir, filename), overlay_image)

        except Exception as e:
            print(f"An error occurred while processing {filename}: {e}")

if __name__ == '__main__':
    main()
