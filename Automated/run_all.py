
import os
import subprocess
import time

# This master script runs the entire training and prediction pipeline for
# both PyTorch and Keras across a set of common backbones.

# --------------------------------------------------------------------------
#                  COMMON BACKBONES FOR AUTOMATION
# --------------------------------------------------------------------------
# This map defines the backbones that are common to both frameworks.
# The key is the name for the PyTorch library, and the value is for Keras.
# Note: Capitalization matters for Keras!
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
    print("========= STARTING FULL AUTOMATED PIPELINE =========