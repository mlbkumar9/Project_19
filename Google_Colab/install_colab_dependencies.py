import subprocess
import sys
import os
from google.colab import drive

# This script installs all necessary dependencies for the Project_19 scripts
# when running in a Google Colab environment.
# It should be run in a Colab cell using '!python install_colab_dependencies.py'
# or by copying and pasting the individual !pip install commands into Colab cells.

def verify_colab_base_dir(expected_base_dir):
    """
    Verifies if the expected_base_dir is consistent with the mounted Google Drive.
    """
    print("\n--- Verifying COLAB_BASE_DIR ---")
    try:
        # Attempt to mount Google Drive if not already mounted
        if not os.path.exists('/content/drive'):
            print("Google Drive not mounted. Attempting to mount...")
            drive.mount('/content/drive')
            print("Google Drive mounted.")
        else:
            print("Google Drive already mounted.")

        # Check if the expected_base_dir exists
        if not os.path.exists(expected_base_dir):
            print(f"WARNING: The configured COLAB_BASE_DIR '{expected_base_dir}' does NOT exist.")
            print("Please ensure your project folder is correctly placed in Google Drive")
            print("and that COLAB_BASE_DIR in your scripts matches its absolute path.")
            print("Example: '/content/drive/MyDrive/YourProjectFolder'")
            return False
        else:
            print(f"COLAB_BASE_DIR '{expected_base_dir}' exists.")
            return True

    except Exception as e:
        print(f"An error occurred during COLAB_BASE_DIR verification: {e}")
        print("Please manually verify your Google Drive mount and COLAB_BASE_DIR setting.")
        return False
def install_package(package_list):
    """Installs a list of packages using pip."""
    for package in package_list:
        print(f"Installing {package}...")
        try:
            # Use sys.executable to ensure pip from the current environment is used
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], check=True)
            print(f"{package} installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while installing {package}: {e}")

if __name__ == '__main__':
    print("Starting dependency installation...")

    # Define the expected COLAB_BASE_DIR from one of your scripts for verification
    # This should match the path you set in your other _colab.py files
    EXPECTED_COLAB_BASE_DIR = '/content/drive/MyDrive/1_Project_Files/Google_Colab/19_Project_19'
    verify_colab_base_dir(EXPECTED_COLAB_BASE_DIR)

    common_deps = ['numpy', 'opencv-python', 'pillow', 'scikit-learn']
    tf_keras_deps = ['tensorflow', 'keras-unet-collection']
    pytorch_deps = ['torch', 'torchvision', 'torchaudio', 'segmentation-models-pytorch']

    install_package(common_deps)
    install_package(tf_keras_deps)
    install_package(pytorch_deps)

    print("\nAll dependency installation attempts completed.")
    print("Please ensure you restart your runtime if prompted by Colab after installation.")
