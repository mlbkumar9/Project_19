
import os
import subprocess
import time
import sys


# Get the absolute path of the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------
#                  COMMON BACKBONES FOR AUTOMATION
# --------------------------------------------------------------------------
BACKBONE_MAP = {
    'resnet50': 'ResNet50',
    'resnet101': 'ResNet101',
    'vgg16': 'VGG16',
    'vgg19': 'VGG19',
    'densenet121': 'DenseNet121',
    'mobilenet_v2': 'MobileNetV2',
}
# --------------------------------------------------------------------------



def run_command(command):
    """Runs a shell command and prints its output in real-time."""
    print(f"\n>>>>> Running command: {' '.join(command)} <<<<<")
    # Use the full path to the python executable that is running this script
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip(), flush=True)
    rc = process.poll()
    if rc != 0:
        print(f"!!!!!! Command failed with exit code {rc} !!!!!!")
    return rc



def main():
    """Main function to iterate through backbones and run all scripts."""
    start_time = time.time()
    print("========= STARTING FULL AUTOMATED PIPELINE =========")

    # Get the python executable path to ensure we use the same environment
    python_executable = sys.executable

    for i, (pt_backbone, keras_backbone) in enumerate(BACKBONE_MAP.items()):
        print(f"\n======================================================================")
        print(f"  BACKBONE {i+1}/{len(BACKBONE_MAP)}: PyTorch='{pt_backbone}' / Keras='{keras_backbone}'")
        print(f"======================================================================")

        # --- PyTorch Pipeline ---
        train_script_pt = os.path.join(SCRIPT_DIR, 'automated_train_pytorch_openvino.py')
        predict_script_pt = os.path.join(SCRIPT_DIR, 'automated_predict_pytorch_openvino.py')

        if run_command([python_executable, train_script_pt, pt_backbone]) == 0:
            run_command([python_executable, predict_script_pt, pt_backbone])
        else:
            print(f"Skipping PyTorch prediction for '{pt_backbone}' due to training failure.")

        print("\n------------------------- ( switching to Keras ) -------------------------")

        # --- Keras Pipeline ---
        train_script_keras = os.path.join(SCRIPT_DIR, 'automated_train_keras_openvino.py')
        predict_script_keras = os.path.join(SCRIPT_DIR, 'automated_predict_keras_openvino.py')

        if run_command([python_executable, train_script_keras, keras_backbone]) == 0:
            run_command([python_executable, predict_script_keras, keras_backbone])
        else:
            print(f"Skipping Keras prediction for '{keras_backbone}' due to training failure.")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n========= AUTOMATED PIPELINE COMPLETE =========")
    print(f"Total execution time: {total_time / 60:.2f} minutes")


if __name__ == '__main__':
    main()
