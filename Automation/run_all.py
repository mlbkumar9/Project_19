  import subprocess
  import os

  def run_command(command):
      """Runs a command and checks for errors."""
      print(f"\n{'='*20}\nRunning command: {' '.join(command)}\n{'='*20}")
      try:
          # Using shell=True on Windows to ensure commands are found
          subprocess.run(command, check=True, shell=True)
      except subprocess.CalledProcessError as e:
          print(f"ERROR: Command failed with return code {e.returncode}")
          print(f"Failed command: {' '.join(command)}")
          # raise e # Uncomment to stop the entire script on the first error

  def main():
      """Main function to run the entire automated workflow for PyTorch models."""
      # --- Configuration ---
      # This list contains the backbones to be trained and tested in sequence.
      PYTORCH_BACKBONES = [
          'resnet50',
          'vgg16',
          'vgg19',
          'densenet121',
          'simple'
      ]

      AUTOMATION_DIR = os.path.dirname(os.path.abspath(__file__))

      for backbone in PYTORCH_BACKBONES:
          print(f"\n\n{'#'*60}")
          print(f"### STARTING PYTORCH WORKFLOW FOR BACKBONE: {backbone} ###")
          print(f"{'#'*60}\n")

          # --- PyTorch Workflow ---
          print(f"\n--- Running PyTorch Training for {backbone} ---")
          train_pytorch_cmd = ['python', os.path.join(AUTOMATION_DIR, 'automated_train_pytorch.py'), '--backbone', backbone]
          run_command(train_pytorch_cmd)

          print(f"\n--- Running PyTorch Prediction for {backbone} ---")
          predict_pytorch_cmd = ['python', os.path.join(AUTOMATION_DIR, 'automated_predict_pytorch.py'), '--backbone', backbone]
          run_command(predict_pytorch_cmd)

      print(f"\n\n{'#'*60}")
      print("### ALL PYTORCH WORKFLOWS COMPLETE ###")
      print(f"{'#'*60}\n")

  if __name__ == '__main__':
      main()