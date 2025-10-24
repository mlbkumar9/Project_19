"""
U-Net Model Trainer (Multi-Backbone)

This script trains a U-Net segmentation model to detect damage in images.
It uses the segmentation-models-pytorch library for robust and correct architectures.
"""
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

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Dataset Class ---
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

# --- 2. Main Training Function ---
def main():
    """Main function to configure and run the training process."""
    # --------------------------------------------------------------------------
    #                    CHOOSE YOUR U-NET BACKBONE HERE
    #--------------------------------------------------------------------------
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
    #--------------------------------------------------------------------------

    # --- Configuration ---
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    IMAGE_DIR = os.path.join(BASE_DIR, 'RAW_Images')
    MASK_DIR = os.path.join(BASE_DIR, 'Masks')
    OUTPUT_MODEL_DIR = os.path.join(BASE_DIR, 'Trained_Models', 'Pytorch')
    MODEL_SAVE_PATH = os.path.join(OUTPUT_MODEL_DIR, f'smp_unet_{BACKBONE}.pth')
    EPOCHS = 25
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4

    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

    print(f"Starting U-Net Model Training (PyTorch) with '{BACKBONE}' backbone...")
    print("-" * 50)

    # --- Data Loading and Splitting ---
    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, '*.png')))
    mask_paths = sorted(glob.glob(os.path.join(MASK_DIR, '*.png')))
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    train_dataset = DamageDataset(train_images, train_masks)
    val_dataset = DamageDataset(val_images, val_masks)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Found {len(image_paths)} images. Training on {len(train_dataset)}, validating on {len(val_dataset)}.")
    print("-" * 50)

    # --- Model Initialization ---
    device = get_device()
    print(f"Using device: {device}")
    
    model = smp.Unet(
        encoder_name=BACKBONE,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
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

        print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    # --- Save Final Model ---
    if best_model_wts:
        torch.save(best_model_wts, MODEL_SAVE_PATH)
        print("-" * 50)
        print(f"Training complete. Best model saved to {MODEL_SAVE_PATH}")
        print(f"Best Validation Loss: {best_val_loss:.4f}")
    else:
        print("Training did not result in a best model to save.")

if __name__ == '__main__':
    main()
