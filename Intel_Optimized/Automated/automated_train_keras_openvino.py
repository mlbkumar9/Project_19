import os
import numpy as np
import cv2
import tensorflow as tf
from keras_unet_collection import models
from sklearn.model_selection import train_test_split
import sys

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

def load_data(image_dir, mask_dir, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])
    X, y = [], []
    for img_path, mask_path in zip(image_files, mask_files):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        X.append(img)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, target_size)
        mask = np.expand_dims(mask, axis=-1)
        y.append(mask)
    # Return images in [0, 255] range for preprocessing
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='float32') / 255.0
    return X, y

def preprocess_for_backbone(X, backbone_name):
    """Applies the appropriate preprocessing for a given backbone."""
    # The preprocess_input functions expect input in [0, 255] range
    if 'DenseNet' in backbone_name:
        from tensorflow.keras.applications.densenet import preprocess_input
        return preprocess_input(X)
    elif 'ResNet' in backbone_name and 'V2' not in backbone_name:
        from tensorflow.keras.applications.resnet import preprocess_input
        return preprocess_input(X)
    elif 'ResNet' in backbone_name and 'V2' in backbone_name:
        from tensorflow.keras.applications.resnet_v2 import preprocess_input
        return preprocess_input(X)
    elif 'VGG' in backbone_name:
        from tensorflow.keras.applications.vgg16 import preprocess_input
        return preprocess_input(X)
    elif 'MobileNetV2' == backbone_name:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        return preprocess_input(X)
    else:
        # Default preprocessing if no backbone is matched
        return X / 255.0

def main(backbone):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGE_DIR = os.path.join(BASE_DIR, 'RAW_Images')
    MASK_DIR = os.path.join(BASE_DIR, 'Masks')
    OUTPUT_MODEL_DIR = os.path.join(BASE_DIR, 'Trained_Models', 'Keras')
    MODEL_SAVE_PATH = os.path.join(OUTPUT_MODEL_DIR, f'kuc_unet-plus_{backbone}.keras')

    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

    X, y = load_data(IMAGE_DIR, MASK_DIR)
    
    # Apply backbone-specific preprocessing
    X_processed = preprocess_for_backbone(X.copy(), backbone)

    X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    print(f"--- Starting Keras Training: U-Net++ with '{backbone}' backbone ---")

    model = models.unet_plus_2d((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 
                               filter_num=[64, 128, 256, 512],
                               n_labels=1, stack_num_down=2, stack_num_up=2,
                               activation='ReLU', output_activation='Sigmoid',
                               batch_norm=True, pool=True, unpool=True, 
                               backbone=backbone, weights='imagenet',
                               name=f'unet-plus_{backbone}')

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=4, epochs=30, verbose=2, # Verbose=2 for cleaner logs in automation
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, mode='max'),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, min_lr=1e-6, mode='max')
        ]
    )
    print(f"--- Keras Training Complete. Best model saved to {MODEL_SAVE_PATH} ---")

    # --- Convert to ONNX format ---
    import onnx
    import tf2onnx

    ONNX_MODEL_SAVE_PATH = os.path.join(OUTPUT_MODEL_DIR, f'kuc_unet-plus_{backbone}.onnx')
    print(f"Converting Keras model to ONNX format and saving to {ONNX_MODEL_SAVE_PATH}...")
    
    # Load the best Keras model to convert
    best_keras_model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    # Define input signature for ONNX conversion
    spec = (tf.TensorSpec([1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS], tf.float32, name="input"),)

    # Convert the Keras model to ONNX
    onnx_model, _ = tf2onnx.convert.from_keras(best_keras_model, input_signature=spec, opset=13, output_path=ONNX_MODEL_SAVE_PATH)
    
    print(f"ONNX model saved to {ONNX_MODEL_SAVE_PATH}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python automated_train_keras.py <backbone_name>")
        sys.exit(1)
    main(sys.argv[1])