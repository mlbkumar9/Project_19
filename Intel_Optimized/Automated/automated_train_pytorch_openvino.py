
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import os
import glob
import copy
import sys
import intel_extension_for_pytorch as ipex

def get_device():
    if ipex.xpu.is_available():
        return ipex.xpu.device()
    return torch.device("cpu")

class DamageDataset(Dataset):
    def __init__(self, image_paths, mask_paths, target_size=(512, 512)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, self.target_size)
        mask = cv2.resize(mask, self.target_size)
        image = image.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=-1).astype(np.float32) / 255.0
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        mask = torch.from_numpy(mask.transpose((2, 0, 1)))
        return image, mask

def main(backbone):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGE_DIR = os.path.join(BASE_DIR, 'RAW_Images')
    MASK_DIR = os.path.join(BASE_DIR, 'Masks')
    OUTPUT_MODEL_DIR = os.path.join(BASE_DIR, 'Trained_Models', 'Pytorch')
    MODEL_SAVE_PATH = os.path.join(OUTPUT_MODEL_DIR, f'smp_unet_{backbone}.pth')
    EPOCHS = 25
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4

    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

    print(f"--- Starting PyTorch Training: U-Net with '{backbone}' backbone ---")

    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, '*.png')))
    mask_paths = sorted(glob.glob(os.path.join(MASK_DIR, '*.png')))
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    train_dataset = DamageDataset(train_images, train_masks)
    val_dataset = DamageDataset(val_images, val_masks)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = get_device()
    model = smp.Unet(
        encoder_name=backbone,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Apply IPEX optimization
    model, optimizer = ipex.optimize(model, optimizer=optimizer)

    best_val_loss = float('inf')
    best_model_wts = None

    for epoch in range(EPOCHS):
        model.train()
        running_train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * images.size(0)
        epoch_train_loss = running_train_loss / len(train_dataset)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                running_val_loss += loss.item() * images.size(0)
        epoch_val_loss = running_val_loss / len(val_dataset)

        print(f"  Epoch {epoch+1}/{EPOCHS} -> Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    if best_model_wts:
        torch.save(best_model_wts, MODEL_SAVE_PATH)
        print(f"--- PyTorch Training Complete. Best model saved to {MODEL_SAVE_PATH} ---")

        # --- Convert to ONNX format ---
        ONNX_MODEL_SAVE_PATH = os.path.join(OUTPUT_MODEL_DIR, f'smp_unet_{backbone}.onnx')
        print(f"Converting PyTorch model to ONNX format and saving to {ONNX_MODEL_SAVE_PATH}...")
        
        # Load the best model weights into the model architecture
        model.load_state_dict(best_model_wts)
        model.eval() # Set the model to evaluation mode

        # Create a dummy input for ONNX export
        dummy_input = torch.randn(1, 3, 512, 512).to(device) # Batch size 1, 3 channels, 512x512

        torch.onnx.export(model, 
                          dummy_input, 
                          ONNX_MODEL_SAVE_PATH, 
                          export_params=True, 
                          opset_version=13, 
                          do_constant_folding=True, 
                          input_names=['input'], 
                          output_names=['output'],
                          dynamic_axes={'input' : {0 : 'batch_size'},    # variable batch size
                                        'output' : {0 : 'batch_size'}})
        print(f"ONNX model saved to {ONNX_MODEL_SAVE_PATH}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python automated_train_pytorch.py <backbone_name>")
        sys.exit(1)
    main(sys.argv[1])
