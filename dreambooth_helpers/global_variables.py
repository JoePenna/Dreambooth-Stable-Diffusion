import datetime
import os
from dreambooth_helpers.arguments import dreambooth_arguments as args

class DreamboothGlobalVariables():
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    log_directory_name = "logs"
    checkpoint_output_directory_name = "ckpts"
    checkpoint_intermediate_steps_directory_name = "trainstep_ckpts"
    config_directory_name = "configs"

    # Set in the setup function
    training_folder_name = ""
    is_debug = False

    def setup(self):
        self.training_folder_name = f"{self.now}_{args.project_name}"
        self.is_debug = args.debug
        self.CreateLogFolders()
    def LogDirectory(self):
        return os.path.join(self.log_directory_name, self.training_folder_name)
    def LogCheckpointDirectory(self):
        return os.path.join(self.LogDirectory(), self.checkpoint_output_directory_name)

    def LogIntermediateCheckpointsDirectory(self):
        return os.path.join(self.LogCheckpointDirectory(), self.checkpoint_intermediate_steps_directory_name)

    def LogConfigDirectory(self):
        return os.path.join(self.LogDirectory(), self.config_directory_name)

    def CreateLogFolders(self):
        os.makedirs(self.LogDirectory(), exist_ok=True)
        os.makedirs(self.LogCheckpointDirectory(), exist_ok=True)
        os.makedirs(self.LogIntermediateCheckpointsDirectory(), exist_ok=True)
        os.makedirs(self.LogConfigDirectory(), exist_ok=True)


dreambooth_global_variables = DreamboothGlobalVariables()