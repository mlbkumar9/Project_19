"""
U-Net Damage Predictor (PyTorch Version)

This script uses a pre-trained U-Net model to predict damage in new images.
"""
import cv2
import numpy as np
import os
import onnxruntime as ort
# import torch # Not needed for ONNX inference
# import torch.nn as nn # Not needed for ONNX inference
# import segmentation_models_pytorch as smp # Not needed for ONNX inference



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
    IMG_WIDTH = 512 # Added for ONNX input shape
    IMG_HEIGHT = 512 # Added for ONNX input shape

    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    INPUT_DIR = os.path.join(BASE_DIR, 'Input_Images_To_Analyze')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'Predictions', 'Pytorch', f'{BACKBONE}_onnx') # Changed output dir name
    ONNX_MODEL_PATH = os.path.join(BASE_DIR, 'Trained_Models', 'Pytorch', f'smp_unet_{BACKBONE}.onnx') # Changed to ONNX model path

    if not os.path.exists(ONNX_MODEL_PATH): 
        print(f"ONNX model not found at {ONNX_MODEL_PATH}. Please train the model first.")
        return

    os.makedirs(os.path.join(OUTPUT_DIR, 'Masks'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'Overlays'), exist_ok=True)

    # --- ONNX Runtime Session Setup ---
    print(f"Loading ONNX model from {ONNX_MODEL_PATH} with OpenVINO Execution Provider...")
    sess_options = ort.SessionOptions()
    try:
        session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options, providers=['OpenVINOExecutionProvider', 'CPUExecutionProvider'])
        print("ONNX Runtime session created with OpenVINO Execution Provider.")
    except Exception as e:
        print(f"Failed to create ONNX Runtime session with OpenVINO EP: {e}. Falling back to CPUExecutionProvider.")
        session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options, providers=['CPUExecutionProvider'])
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files: 
        print(f"No images found in {INPUT_DIR} for prediction.")
        return

    for filename in image_files:
        image_path = os.path.join(INPUT_DIR, filename)
        original_image = cv2.imread(image_path)
        if original_image is None: 
            print(f"Could not read image: {image_path}")
            continue

        # Preprocess image for ONNX model
        original_size = (original_image.shape[1], original_image.shape[0])
        image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (IMG_WIDTH, IMG_HEIGHT))
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_tensor = image_normalized.transpose((2, 0, 1)) # Channel first for PyTorch ONNX
        image_tensor = np.expand_dims(image_tensor, axis=0) # Add batch dimension

        # ONNX inference
        predicted_mask_onnx = session.run([output_name], {input_name: image_tensor})[0]
        # The output from ONNX model will be (1, 1, 512, 512)
        pred_mask = predicted_mask_onnx[0, 0] > 0.5
        binary_mask = (pred_mask).astype(np.uint8) * 255
        binary_mask_resized = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)

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
