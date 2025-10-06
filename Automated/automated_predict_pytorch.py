
import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np
import os
import sys

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def main(backbone):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_DIR = os.path.join(BASE_DIR, 'Input_Images_To_Analyze')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'Predictions', 'Pytorch', backbone)
    MODEL_PATH = os.path.join(BASE_DIR, 'Trained_Models', 'Pytorch', f'smp_unet_{backbone}.pth')

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}!")
        return

    os.makedirs(os.path.join(OUTPUT_DIR, 'Masks'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'Overlays'), exist_ok=True)

    print(f"--- Starting PyTorch Prediction: U-Net with '{backbone}' backbone ---")

    device = get_device()
    model = smp.Unet(encoder_name=backbone, encoder_weights="imagenet", in_channels=3, classes=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for filename in image_files:
        image_path = os.path.join(INPUT_DIR, filename)
        binary_mask, original_image = predict_mask(model, image_path, device)
        if binary_mask is None: continue

        damage_area = np.count_nonzero(binary_mask)

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
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_image, contours, -1, (0, 0, 255), 2)

        # --- Add Labels to Overlay ---
        cv2.putText(overlay_image, f"Category: {category}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(overlay_image, f"Area: {damage_area} px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imwrite(os.path.join(OUTPUT_DIR, 'Overlays', filename), overlay_image)

    print(f"--- PyTorch Prediction Complete. Results in {OUTPUT_DIR} ---")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python automated_predict_pytorch.py <backbone_name>")
        sys.exit(1)
    main(sys.argv[1])
