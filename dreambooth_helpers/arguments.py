import argparse
import math
import os

import torch
from pytorch_lightning import seed_everything


class DreamboothArguments():
    project_name: str
    debug: bool = False
    seed: int = 23
    max_training_steps: int
    token: str
    token_only: bool = False

    # used to be --actual_resume
    training_model: str

    # used to be --data_root
    training_images: str

    # used to be --reg_data_root
    regularization_images: str

    class_word: str

    flip_p: float = 0.5
    learning_rate: float = None
    save_every_x_steps: int = 0
    gpu: int = 0

    def parse_arguments(self):
        parser = self.get_parser()
        # We may need this here
        # parser = Trainer.add_argparse_args(parser)
        opt, unknown = parser.parse_known_args()

        # Assign the opt.* arguments to our local references

        self.project_name = opt.project_name
        self.debug = opt.debug

        self.seed = opt.seed
        seed_everything(self.seed)

        self.max_training_steps = opt.max_training_steps
        if self.max_training_steps <= 0:
            raise Exception("'--max_training_steps': Required. Must be > 0.")

        self.token = opt.token
        self.token_only = opt.token_only

        # used to be --actual_resume
        self.training_model = opt.training_model
        if not os.path.exists(self.training_model):
            raise Exception(f"'--training_model': Required. Could not find '{self.training_model}'.")

        # used to be --data_root
        self.training_images = opt.training_images
        if not os.path.exists(self.training_images):
            raise Exception(f"'--training_images': Required. Could not find '{self.training_images}'.")

        # used to be --reg_data_root
        self.regularization_images = opt.regularization_images
        if self.regularization_images is not None and not os.path.exists(self.regularization_images):
            raise Exception(f"'--regularization_images': Could not find '{self.regularization_images}'.")

        self.class_word = opt.class_word

        self.flip_p = opt.flip_p
        if self.flip_p < 0 or self.flip_p > 1:
            raise Exception("--flip_p: must be between 0 and 1")

        self.learning_rate = opt.learning_rate
        self.save_every_x_steps = opt.save_every_x_steps
        if self.save_every_x_steps < 0:
            raise Exception("--save_every_x_steps: must be greater than or equal to 0")

        self.gpu = opt.gpu

        #self.validate_gpu_vram()

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

    def get_parser(self, **parser_kwargs):
        def str2bool(v):
            if isinstance(v, bool):
                return v
            if v.lower() in ("yes", "true", "t", "y", "1"):
                return True
            elif v.lower() in ("no", "false", "f", "n", "0"):
                return False
            else:
                raise argparse.ArgumentTypeError("Boolean value expected.")

        parser = argparse.ArgumentParser(**parser_kwargs)

        parser.add_argument(
            "--project_name",
            type=str,
            required=True,
            default=None,
            help="Name of the project"
        )
        parser.add_argument(
            "--debug",
            type=str2bool,
            nargs="?",
            const=True,
            default=self.debug,
            help="Enable debug logging",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=self.seed,
            help="seed for seed_everything",
        )

        parser.add_argument(
            "--max_training_steps",
            type=int,
            required=True,
            help="Number of training steps to run"
        )

        parser.add_argument(
            "--token",
            type=str,
            required=True,
            help="Unique token you want to represent your trained model. Ex: firstNameLastName."
        )

        parser.add_argument(
            "--token_only",
            type=str2bool,
            const=True,
            default=self.token_only,
            nargs="?",
            help="Train only using the token and no class."
        )

        parser.add_argument(
            "--training_model",
            type=str,
            required=True,
            help="Path to model to train (model.ckpt)"
        )

        parser.add_argument(
            "--training_images",
            type=str,
            required=True,
            help="Path to training images directory"
        )

        parser.add_argument(
            "--regularization_images",
            type=str,
            required=False,
            help="Path to directory with regularization images"
        )
        parser.add_argument(
            "--class_word",
            type=str,
            required=False,
            help="Match class_word to the category of images you want to train. Example: 'man', 'woman', 'dog', or 'artstyle'."
        )

        parser.add_argument(
            "--flip_p",
            type=float,
            required=False,
            default=self.flip_p,
            help="Flip Percentage "
                 "Example: if set to 0.5, will flip (mirror) your training images 50% of the time."
                 "This helps expand your dataset without needing to include more training images."
                 "This can lead to worse results for face training since most people's faces are not perfectly symmetrical."
        )

        parser.add_argument(
            "--learning_rate",
            type=float,
            required=False,
            default=self.learning_rate,
            help="Set the learning rate. Defaults to 1.0e-06 (0.000001).  Accepts scientific notation."
        )

        parser.add_argument(
            "--save_every_x_steps",
            type=int,
            required=False,
            default=self.save_every_x_steps,
            help="Saves a checkpoint every x steps"
        )

        parser.add_argument(
            "--gpu",
            type=int,
            default=self.gpu,
            required=False,
            help="Specify a GPU other than 0 to use for training.  Multi-GPU support is not currently implemented."
        )

        return parser


dreambooth_arguments = DreamboothArguments()
