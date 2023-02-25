import os
import sys
import json
from datetime import datetime, timezone
import shutil

class JoePennaDreamboothConfigSchemaV1:
    def __init__(
            self,
            config_file_name,
            date_utc,
            dataset,
            project_name,
            max_training_steps,
            training_images_count,
            training_images,
            class_word,
            flip_percent,
            token,
            learning_rate,
            save_every_x_steps,
            model_repo_id,
            model_filename,
    ):
        self.schema = 1
        self.config_file_name = config_file_name
        self.date_utc = date_utc
        self.dataset = dataset
        self.project_name = project_name
        self.max_training_steps = max_training_steps
        self.training_images_count = training_images_count
        self.training_images = training_images
        self.class_word = class_word
        self.flip_percent = flip_percent
        self.token = token
        self.learning_rate = learning_rate
        self.save_every_x_steps = save_every_x_steps
        self.model_repo_id = model_repo_id
        self.model_filename = model_filename

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def createCheckpointFileName(self, steps):
        date_string = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        return f"{date_string}_{self.project_name}_" \
               f"{steps}_steps_" \
               f"{self.training_images_count}_training_images_" \
               f"{self.token}_token_" \
               f"{self.class_word}_class_word.ckpt".replace(" ", "_")

def parse_config_file(config_file_path) -> JoePennaDreamboothConfigSchemaV1:
    # parse the config file "joepenna-dreambooth-configs/active-config.json"
    if not os.path.exists(config_file_path):
        print(f"{config_file_path} not found.", file=sys.stderr)
        return None
    else:
        config_file = open(config_file_path)
        config_parsed = json.load(config_file)

        if config_parsed['schema'] == 1:
            return JoePennaDreamboothConfigSchemaV1(
                config_parsed['config_file_name'],
                config_parsed['date_utc'],
                config_parsed['dataset'],
                config_parsed['project_name'],
                config_parsed['max_training_steps'],
                config_parsed['training_images_count'],
                config_parsed['training_images'],
                config_parsed['class_word'],
                config_parsed['flip_percent'],
                config_parsed['token'],
                config_parsed['learning_rate'],
                config_parsed['save_every_x_steps'],
                config_parsed['model_repo_id'],
                config_parsed['model_filename'],
            )
        else:
            print(f"Unrecognized schema: {config_parsed['schema']}", file=sys.stderr)

def save_config_file_v1(
        dataset,
        project_name,
        max_training_steps,
        training_images_count,
        training_images,
        class_word,
        flip_percent,
        token,
        learning_rate,
        save_every_x_steps,
        model_repo_id,
        model_filename,
):
    # setup our values for training
    config_date_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    project_config_filename = f"{config_date_time}-{project_name}-joepenna-dreambooth-config.json"


    project_config = JoePennaDreamboothConfigSchemaV1(
        project_config_filename,
        config_date_time,
        dataset,
        project_name,
        max_training_steps,
        training_images_count,
        training_images,
        class_word,
        flip_percent,
        token,
        learning_rate,
        save_every_x_steps,
        model_repo_id,
        model_filename,
    )

    project_config_json = project_config.toJSON()
    with open(project_config_filename, "w") as config_file:
        config_file.write(project_config_json)

    config_save_path = "./joepenna-dreambooth-configs"
    if not os.path.exists(config_save_path):
        os.mkdir(config_save_path)

    shutil.copy(project_config_filename, f"{config_save_path}/{project_config_filename}")
    shutil.move(project_config_filename, f"{config_save_path}/active-config.json")

    print(project_config_json)
    print(f"✅ {project_config_filename} successfully generated.  Proceed to training.")
