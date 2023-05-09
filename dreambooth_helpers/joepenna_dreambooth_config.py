import json
import math
import os
import glob
import shutil
import sys
from datetime import datetime, timezone

import torch
from pytorch_lightning import seed_everything


class JoePennaDreamboothConfigSchemaV1:
    def __init__(self):
        self.schema: int = 1
        self.config_date_time: str = ''
        self.project_config_filename: str = ''

        # Project
        self.project_name: str = ''
        self.seed: int = 23
        self.debug: bool = False
        self.gpu: int = 0

        # Training Steps
        self.max_training_steps: int = 2000
        self.save_every_x_steps: int = 0

        # Training & Regularization Images
        self.training_images_folder_path: str = ''
        self.training_images_count: int = 0
        self.training_images: list[str] = []
        self.regularization_images_folder_path: str = None

        # Token and Class
        self.token: str = ''
        self.token_only: bool = False
        self.class_word: str = ''

        # Training Params
        self.flip_percent: float = 0.5
        self.learning_rate: float = 1.0e-06

        # Model Info
        self.model_repo_id: str = ''
        self.model_path: str = ''

    def saturate(
            self,
            project_name: str,
            max_training_steps: int,
            save_every_x_steps: int,
            training_images_folder_path: str,
            regularization_images_folder_path: str,
            token: str,
            class_word: str,
            flip_percent: float,
            learning_rate: float,
            model_path: str,
            config_date_time: str = None,
            seed: int = 23,
            debug: bool = False,
            gpu: int = 0,
            model_repo_id: str = '',
            token_only: bool = False,
            run_seed_everything: bool = True,
    ):

        # Map the values
        self.project_name = project_name
        if self.project_name is None or self.project_name == '':
            raise Exception("'--project_name': Required.")

        if config_date_time is None:
            self.config_date_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        else:
            self.config_date_time = config_date_time

        # parameter values
        self.project_config_filename = f"{self.config_date_time}-{self.project_name}-config.json"

        self.seed = seed

        # Global seed
        if run_seed_everything:
            seed_everything(self.seed)

        self.debug = debug
        self.gpu = gpu

        self.max_training_steps = max_training_steps
        if self.max_training_steps <= 0:
            raise Exception("'--max_training_steps': Required. Must be > 0.")

        self.save_every_x_steps = save_every_x_steps
        if self.save_every_x_steps < 0:
            raise Exception("--save_every_x_steps: must be greater than or equal to 0")

        self.training_images_folder_path = training_images_folder_path

        if not os.path.exists(self.training_images_folder_path):
            raise Exception(f"Training Images Path Not Found: '{self.training_images_folder_path}'.")

        _training_image_paths = [f for f in
                                 glob.glob(os.path.join(self.training_images_folder_path, '**', '*.jpg'), recursive=True) +
                                 glob.glob(os.path.join(self.training_images_folder_path, '**', '*.jpeg'), recursive=True) +
                                 glob.glob(os.path.join(self.training_images_folder_path, '**', '*.png'), recursive=True)
                             ]

        _training_image_paths = [os.path.relpath(f, self.training_images_folder_path) for f in _training_image_paths]

        if len(_training_image_paths) <= 0:
            raise Exception(f"No Training Images (*.png, *.jpg, *.jpeg) found in '{self.training_images_folder_path}'.")

        self.training_images_count = len(_training_image_paths)
        self.training_images = _training_image_paths

        if token_only is False and regularization_images_folder_path is not None and regularization_images_folder_path != '':
            self.regularization_images_folder_path = regularization_images_folder_path

            if not os.path.exists(self.regularization_images_folder_path):
                raise Exception(f"Regularization Images Path Not Found: '{self.regularization_images_folder_path}'.")

        self.token = token
        if self.token is None or self.token == '':
            raise Exception(f"Token not provided.")

        self.token_only = token_only
        if token_only is False:
            self.class_word = class_word

        self.flip_percent = flip_percent
        if self.flip_percent < 0 or self.flip_percent > 1:
            raise Exception("--flip_p: must be between 0 and 1")

        self.learning_rate = learning_rate
        self.model_repo_id = model_repo_id

        self.model_path = model_path
        if not os.path.exists(self.model_path):
            raise Exception(f"Model Path Not Found: '{self.model_path}'.")

        self.validate_gpu_vram()

        self._create_log_folders()

    def validate_gpu_vram(self):
        def convert_size(size_bytes):
            if size_bytes == 0:
                return "0B"
            size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
            i = int(math.floor(math.log(size_bytes, 1024)))
            p = math.pow(1024, i)
            s = round(size_bytes / p, 2)
            return "%s %s" % (s, size_name[i])

            # Check total available GPU memory

        gpu_vram = int(torch.cuda.get_device_properties(self.gpu).total_memory)
        print(f"gpu_vram: {convert_size(gpu_vram)}")
        twenty_one_gigabytes = 22548578304
        if gpu_vram < twenty_one_gigabytes:
            raise Exception(f"VRAM: Currently unable to run on less than {convert_size(twenty_one_gigabytes)} of VRAM.")

    def saturate_from_file(
            self,
            config_file_path: str,
    ):
        if not os.path.exists(config_file_path):
            print(f"{config_file_path} not found.", file=sys.stderr)
            return None
        else:
            config_file = open(config_file_path)
            config_parsed = json.load(config_file)

            if config_parsed['schema'] == 1:
                self.saturate(
                    project_name=config_parsed['project_name'],
                    max_training_steps=config_parsed['max_training_steps'],
                    save_every_x_steps=config_parsed['save_every_x_steps'],
                    training_images_folder_path=config_parsed['training_images_folder_path'],
                    regularization_images_folder_path=config_parsed['regularization_images_folder_path'],
                    token=config_parsed['token'],
                    class_word=config_parsed['class_word'],
                    flip_percent=config_parsed['flip_percent'],
                    learning_rate=config_parsed['learning_rate'],
                    model_path=config_parsed['model_path'],
                    config_date_time=config_parsed['config_date_time'],
                    seed=config_parsed['seed'],
                    debug=config_parsed['debug'],
                    gpu=config_parsed['gpu'],
                    model_repo_id=config_parsed['model_repo_id'],
                    token_only=config_parsed['token_only'],
                )
            else:
                print(f"Unrecognized schema: {config_parsed['schema']}", file=sys.stderr)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def create_checkpoint_file_name(self, steps: str):
        date_string = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        return f"{date_string}_{self.project_name}_" \
               f"{int(steps):05d}_steps_" \
               f"{self.training_images_count}_training_images_" \
               f"{self.token}_token_" \
               f"{self.class_word}_class_word.ckpt".replace(" ", "_")

    def save_config_to_file(
            self,
            save_path: str,
            create_active_config: bool = False,
    ):
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        project_config_json = self.toJSON()
        config_save_path = os.path.join(save_path, self.project_config_filename)
        with open(config_save_path, "w") as config_file:
            config_file.write(project_config_json)

        if create_active_config:
            shutil.copy(config_save_path, os.path.join(save_path, "active-config.json"))
            print(project_config_json)
            print(f"✅ {self.project_config_filename} successfully generated.  Proceed to training.")

    def get_training_folder_name(self) -> str:
        return f"{self.config_date_time}_{self.project_name}"

    def log_directory(self) -> str:
        return os.path.join("logs", self.get_training_folder_name())

    def log_checkpoint_directory(self) -> str:
        return os.path.join(self.log_directory(), "ckpts")

    def log_intermediate_checkpoints_directory(self) -> str:
        return os.path.join(self.log_checkpoint_directory(), "trainstep_ckpts")

    def log_config_directory(self) -> str:
        return os.path.join(self.log_directory(), "configs")

    def trained_models_directory(self) -> str:
        return "trained_models"

    def _create_log_folders(self):
        os.makedirs(self.log_directory(), exist_ok=True)
        os.makedirs(self.log_checkpoint_directory(), exist_ok=True)
        os.makedirs(self.log_config_directory(), exist_ok=True)
        os.makedirs(self.trained_models_directory(), exist_ok=True)
