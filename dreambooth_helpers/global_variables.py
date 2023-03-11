import datetime
import os


class DreamboothGlobalVariables():
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    log_directory_name = "logs"
    checkpoint_output_directory_name = "ckpts"
    checkpoint_intermediate_steps_directory_name = "trainstep_ckpts"
    config_directory_name = "configs"
    trained_models_directory_name = "trained_models"

    # Set in the setup function
    training_folder_name = ""
    debug = False

    def setup(self, project_name: str, debug: bool):
        self.training_folder_name = f"{self.now}_{project_name}"
        self.debug = debug
        self._create_log_folders()

    def log_directory(self):
        return os.path.join(self.log_directory_name, self.training_folder_name)

    def log_checkpoint_directory(self):
        return os.path.join(self.log_directory(), self.checkpoint_output_directory_name)

    def log_intermediate_checkpoints_directory(self):
        return os.path.join(self.log_checkpoint_directory(), self.checkpoint_intermediate_steps_directory_name)

    def log_config_directory(self):
        return os.path.join(self.log_directory(), self.config_directory_name)

    def trained_models_directory(self):
        return self.trained_models_directory_name

    def _create_log_folders(self):
        os.makedirs(self.log_directory(), exist_ok=True)
        os.makedirs(self.log_checkpoint_directory(), exist_ok=True)
        os.makedirs(self.log_config_directory(), exist_ok=True)


dreambooth_global_variables = DreamboothGlobalVariables()
