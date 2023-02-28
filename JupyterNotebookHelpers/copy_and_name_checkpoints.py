from IPython.display import clear_output
import os
import re
import shutil
import glob
from  JupyterNotebookHelpers.joe_penna_dreambooth_config import parse_config_file

class CopyAndNameCheckpoints:
    def __init__(self):
        pass

    def execute(self, active_config_path, output_folder="./trained_models"):
        clear_output()
        config = parse_config_file(active_config_path)

        if config is not None:

            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

            # copy the config file to the model directory as well
            shutil.copy(active_config_path, f"{output_folder}/{config.config_file_name}")

            logs_directory = "./logs"
            if os.path.exists(logs_directory):
                latest_training_directory = glob.glob(f"{logs_directory}/*")[-1]
            else:
                print(f"No checkpoints found in {logs_directory}")
                return

            # Gather, name, and move the checkpoint files.
            checkpoint_paths = []
            steps = []
            if config.save_every_x_steps == 0:
                checkpoint_paths.append(f"{latest_training_directory}/checkpoints/last.ckpt")
                steps.append(str(config.max_training_steps))
            else:
                checkpoints_directory = f"{latest_training_directory}/checkpoints/trainstep_checkpoints"
                file_paths = glob.glob(f"{checkpoints_directory}/*")

                for i, original_file_name in enumerate(file_paths):
                    # Remove the "epoch=000000-step=0000" text
                    checkpoint_steps = re.sub(checkpoints_directory + "/epoch=\d{6}-step=0*", "", original_file_name)

                    # Remove the .ckpt
                    checkpoint_steps = checkpoint_steps.replace(".ckpt", "")

                    checkpoint_paths.append(f"{original_file_name}")
                    steps.append(checkpoint_steps)

            checkpoints_found = False
            for i, training_steps in enumerate(steps):
                # Setup the filenames
                original_file_name = checkpoint_paths[i]
                file_name = config.createCheckpointFileName(training_steps)
                output_file_name = os.path.join(output_folder, file_name)

                if os.path.exists(original_file_name):
                    print(f"Moving {original_file_name} to {output_file_name}")

                    # Move the checkpoint
                    shutil.move(original_file_name, output_file_name)

                    checkpoints_found = True

            if checkpoints_found:
                print(f"✅ Download your trained model(s) from the '{output_folder}' folder and use in your favorite Stable Diffusion repo!")
            else:
                print("No checkpoints found.")

