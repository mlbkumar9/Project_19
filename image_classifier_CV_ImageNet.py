import os
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import csv

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Loads an image, resizes it, and prepares it for the classification model.

    Args:
        image_path (str): The path to the image file.
        target_size (tuple): The target size for the image (width, height).

    Returns:
        A preprocessed image ready for the model.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img)
        # Add a batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        # Scale the image pixels to the range [0, 1]
        preprocessed_image = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        return preprocessed_image
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"An error occurred during image preprocessing: {e}")
        return None

def classify_image(image_path):
    """
    Preprocesses an image and classifies it using a pre-trained MobileNetV2 model.

    Args:
        image_path (str): The path to the image file.
        
    Returns:
        A list of the top 3 decoded predictions.
    """
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)

    if preprocessed_image is not None:
        # Load the pre-trained MobileNetV2 model
        model = tf.keras.applications.MobileNetV2(weights='imagenet')

        # Make a prediction
        predictions = model.predict(preprocessed_image)

        # Decode the predictions to get human-readable labels
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
        
        return decoded_predictions
    return None

if __name__ == '__main__':
    # --- Configuration ---
    
    base_dir = r'C:\Users\Maahi\Projects\Project_19'
    image_directory = os.path.join(base_dir, 'RAW_Images')
    output_dir = os.path.join(base_dir, 'ImageNet_Classified_Images')
    csv_output_file = os.path.join(base_dir, 'imagenet_classification_results.csv')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    results_data = []

    if not os.path.isdir(image_directory):
        print(f"Error: Directory not found at {image_directory}")
    else:
        print(f"Classifying images in: {image_directory}")
        
        # --- Process Each Image ---
        for filename in sorted(os.listdir(image_directory)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_path = os.path.join(image_directory, filename)
                print(f"--- Classifying {filename} ---")
                
                predictions = classify_image(image_path)

                if predictions:
                    # --- 1. Store Results for CSV ---
                    top_1_pred = predictions[0]
                    results_data.append({
                        'Filename': filename,
                        'Top_Prediction': top_1_pred[1],
                        'Top_Prediction_Score': f"{top_1_pred[2]:.2f}",
                        'All_Predictions': f"1: {predictions[0][1]} ({predictions[0][2]:.2f}), 2: {predictions[1][1]} ({predictions[1][2]:.2f}), 3: {predictions[2][1]} ({predictions[2][2]:.2f})"
                    })
                    print(f"Top prediction: {top_1_pred[1]} ({top_1_pred[2]:.2f})")

                    # --- 2. Visualize and Save Annotated Image ---
                    annotated_image = cv2.imread(image_path)
                    
                    # Draw top 3 predictions on the image
                    for i, (imagenet_id, label, score) in enumerate(predictions):
                        text = f"{i+1}: {label} ({score:.2f})"
                        y_pos = 30 + i * 30
                        cv2.putText(annotated_image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    output_path = os.path.join(output_dir, filename)
                    cv2.imwrite(output_path, annotated_image)
                
                print("-" * (len(filename) + 24))

        # --- 3. Write results to CSV ---
        if results_data:
            print("-" * 40)
            print(f"Writing results to {csv_output_file}...")
            fieldnames = ['Filename', 'Top_Prediction', 'Top_Prediction_Score', 'All_Predictions']
            try:
                with open(csv_output_file, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(results_data)
                print("CSV file written successfully.")
            except PermissionError:
                print(f"Warning: Permission denied to write to {csv_output_file}. It may be open in another program.")

    print("Classification complete.")