"""
U-Net Model Trainer (Multi-Backbone)

This script trains a U-Net segmentation model to detect damage in images.
It uses the segmentation-models-pytorch library for robust and correct architectures.
"""
import os
import numpy as np
import cv2
import tensorflow as tf
from keras_unet_collection import models
from sklearn.model_selection import train_test_split

# --- Configuration ---
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

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

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, 'RAW_Images')
MASK_DIR = os.path.join(BASE_DIR, 'Masks')
OUTPUT_MODEL_DIR = os.path.join(BASE_DIR, 'Trained_Models', 'Keras')
MODEL_SAVE_PATH = os.path.join(OUTPUT_MODEL_DIR, f'kuc_unet-plus_{BACKBONE}.keras')

# --- 1. Data Loading and Preprocessing ---
def load_data(image_dir, mask_dir, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    """
    Loads images and masks and resizes them.
    """
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])

    X = []
    y = []

    print(f"Loading {len(image_files)} images and masks...")
    for img_path, mask_path in zip(image_files, mask_files):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        X.append(img)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, target_size)
        mask = np.expand_dims(mask, axis=-1)
        y.append(mask)

    X = np.array(X, dtype='float32') / 255.0
    y = np.array(y, dtype='float32') / 255.0

    return X, y

# --- 2. Training ---
def main():
    """
    Main function to load data, build, compile, and train the model using keras-unet-collection.
    """
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

    X, y = load_data(IMAGE_DIR, MASK_DIR)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Data loaded. Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    print(f"\nBuilding U-Net++ with '{BACKBONE}' backbone using keras-unet-collection...")
    
    model = models.unet_plus_2d((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 
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
                               weights='imagenet',
                               name=f'unet-plus_{BACKBONE}')

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    model.summary()

    print(f"\n--- Starting Model Training ---")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=4,
        epochs=30,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, mode='max'),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, min_lr=1e-6, mode='max')
        ]
    )

    print(f"\n--- Training Complete. ---")
    print(f"Best model saved to {MODEL_SAVE_PATH}")

    # --- Convert to ONNX format ---
    import onnx
    import tf2onnx

    ONNX_MODEL_SAVE_PATH = os.path.join(OUTPUT_MODEL_DIR, f'kuc_unet-plus_{BACKBONE}.onnx')
    print(f"Converting Keras model to ONNX format and saving to {ONNX_MODEL_SAVE_PATH}...")
    
    # Load the best Keras model to convert
    best_keras_model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    # Define input signature for ONNX conversion
    spec = (tf.TensorSpec([1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS], tf.float32, name="input"),)

    # Convert the Keras model to ONNX
    onnx_model, _ = tf2onnx.convert.from_keras(best_keras_model, input_signature=spec, opset=13, output_path=ONNX_MODEL_SAVE_PATH)
    
    print(f"ONNX model saved to {ONNX_MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()