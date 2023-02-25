from IPython.core.display_functions import display
from IPython.display import clear_output
import os
import sys
import shutil
from JupyterNotebookHelpers.joe_penna_dreambooth_config import save_config_file_v1
from ipywidgets import widgets, Layout, HBox
from git import Repo
from JupyterNotebookHelpers.download_model import SDModelOption

class SetupTraining:
    form_widgets = []
    training_images_save_path = "./training_images"
    selected_model: SDModelOption = None

    def __init__(
            self,
            style={'description_width': '150px'},
            label_style={'font_size': '10px', 'text_color': '#777'},
            layout=Layout(width="400px"),
            input_and_description_layout=Layout(width="812px"),
    ):
        self.form_widgets = []
        self.style = style
        self.label_style = label_style
        self.layout = layout
        self.input_and_description_layout = input_and_description_layout

        # Training Images
        self.training_images_uploader = widgets.FileUpload(
            accept='image/*',
            multiple=True,
            description='Training Images',
            tooltip='Training Images',
            button_style='warning',
            style=self.style,
            layout=self.layout,
        )
        self.form_widgets.append(self.training_images_uploader)

        # Regularization Images
        self.reg_images_select = widgets.Dropdown(
            options=["man_euler", "man_unsplash", "person_ddim", "woman_ddim", "artstyle"],
            value="person_ddim",
            description="Regularization Images: ",
            style=self.style,
            layout=self.layout,
        )
        self.build_input_and_label(self.reg_images_select, "person_ddim recommended")

        # Project Name
        self.project_name_input = widgets.Text(
            placeholder='Project Name',
            description='Project Name: ',
            value='ProjectName',
            style=self.style,
            layout=self.layout,
        )
        self.build_input_and_label(self.project_name_input,
                                   "This isn't used for training, just to help you remember what your trained into the model.")

        # Max Training steps
        self.max_training_steps_input = widgets.BoundedIntText(
            value=2000,  # default value
            min=0,  # min value
            max=100000,  # max value
            step=100,  # incriment size
            description='Max Training Steps: ',  # slider label
            tooltip='Max Training Steps',
            style=self.style,
            layout=self.layout,
        )
        self.build_input_and_label(self.max_training_steps_input, "How many steps do you want to train for?")

        # Learning Rate
        self.learning_rate_select = widgets.Dropdown(
            options=[2.0e-06, 1.5e-06, 1.0e-06, 8.0e-07, 6.0e-07, 5.0e-07, 4.0e-07],
            value=1.0e-06,
            description="Learning Rate: ",
            style=self.style,
            layout=self.layout,
        )
        self.build_input_and_label(self.learning_rate_select,
                                   "How fast do you want to train? 1.0e-06 is highly recommended.")

        # Class
        self.class_word_input = widgets.Text(
            value='person',
            placeholder='man / person / woman / artstyle / etc',
            description='Class Word: ',
            style=self.style,
            layout=self.layout,
        )
        self.build_input_and_label(self.class_word_input, "Typical uses are 'man', 'person', 'woman', or 'artstyle'")

        # Flip slider
        self.flip_slider = widgets.FloatSlider(
            value=0.5,
            min=0,
            max=1,
            step=0.05,
            description="Flip Images %: ",
            style=self.style,
            layout=self.layout,
        )
        self.build_input_and_label(self.flip_slider,
                                   "Set to 0.0 or 0.1 if you are training a person's face.  0.75 is the same as 0.25")

        # Token
        self.token_input = widgets.Text(
            value='firstNameLastName',
            placeholder='firstNameLastName',
            description='Token: ',
            style=self.style,
            layout=self.layout,
        )
        self.build_input_and_label(self.token_input,
                                   "Chose your unique token you want to train into stable diffusion (don't use 'sks')")

        # Save every x steps
        self.save_every_x_steps_input = widgets.BoundedIntText(
            value=0,  # default value
            min=0,  # min value
            max=100000,  # max value
            step=50,  # increment size
            description='Save every (x) steps: ',  # slider label
            tooltip='Save every (x) steps.  Leave at 0 to only save the final checkpoint',
            style=self.style,
            layout=self.layout,
        )
        self.build_input_and_label(self.save_every_x_steps_input, "Change to save intermediate checkpoints")

        self.save_form_button = widgets.Button(
            description="Save",
            disabled=False,
            button_style='success',
            tooltip='Save',
            icon='save',
            style=self.style,
            layout=self.layout,
        )
        self.form_widgets.append(self.save_form_button);

        # bind the save_form_button to the submit_form_click event
        self.save_form_button.on_click(self.submit_form_click)

        self.output = widgets.Output()

    def build_label(self, text):
        return widgets.Label(
            value=text,
            style=self.label_style,
            layout=self.layout,
        )

    def build_input_and_label(self, control, label_text):
        form_box = widgets.HBox(
            [control, self.build_label(label_text)],
            layout=self.input_and_description_layout
        )
        self.form_widgets.append(form_box)

    def show_form(self, selected_model: SDModelOption):
        clear_output()

        self.selected_model = selected_model

        # display the form
        for i, widget in enumerate(self.form_widgets):
            if widget != self.save_form_button:
                display(widget)
            else:
                display(widget, self.output)

    def submit_form_click(self, b):
        with self.output:
            self.output.clear_output()

            dataset = self.reg_images_select.value
            uploaded_images = self.training_images_uploader.value
            project_name = self.project_name_input.value
            max_training_steps = int(self.max_training_steps_input.value)
            class_word = self.class_word_input.value
            flip_percent = float(self.flip_slider.value)
            token = self.token_input.value
            learning_rate = self.learning_rate_select.value
            save_every_x_steps = int(self.save_every_x_steps_input.value)

            if len(uploaded_images) == 0:
                print("No training images provided, please click the 'Training Images' upload button.", file=sys.stderr)
                return

            # Regularization Images
            self.download_regularization_images(dataset)

            # Training images
            images = self.handle_training_images(uploaded_images)

            save_config_file_v1(
                dataset=dataset,
                project_name=project_name,
                max_training_steps=max_training_steps,
                training_images_count=len(images),
                training_images=images,
                class_word=class_word,
                flip_percent=flip_percent,
                token=token,
                learning_rate=learning_rate,
                save_every_x_steps=save_every_x_steps,
                model_repo_id=self.selected_model.repo_id,
                model_filename=self.selected_model.filename,
            )

    def download_regularization_images(self, dataset):
        # Download Regularization Images
        repo_name = f"Stable-Diffusion-Regularization-Images-{dataset}"
        regularization_images_git_folder = f"./{repo_name}"
        if not os.path.exists(regularization_images_git_folder):
            print(f"Downloading regularization images for {dataset}")
            Repo.clone_from(f"https://github.com/djbielejeski/{repo_name}.git", repo_name)

            print(f"✅ Regularization images for {dataset} downloaded successfully.")
            regularization_images_root_folder = "regularization_images"
            if not os.path.exists(regularization_images_root_folder):
                os.mkdir(regularization_images_root_folder)

            regularization_images_dataset_folder = f"{regularization_images_root_folder}/{dataset}"
            if not os.path.exists(regularization_images_dataset_folder):
                os.mkdir(regularization_images_dataset_folder)

            regularization_images = os.listdir(f"{regularization_images_git_folder}/{dataset}")
            for file_name in regularization_images:
                shutil.move(os.path.join(f"{regularization_images_git_folder}/{dataset}", file_name),
                            regularization_images_dataset_folder)

        else:
            print(f"✅ Regularization images for {dataset} already exist. Skipping download...")
    def handle_training_images(self, uploaded_images):
        if os.path.exists(self.training_images_save_path):
            # remove existing images
            shutil.rmtree(self.training_images_save_path)

        # Create the training images directory
        os.mkdir(self.training_images_save_path)

        images = []
        image_widgets = []
        for i, img in enumerate(uploaded_images):
            images.append(img.name)
            image_widgets.append(widgets.Image(
                value=img.content
            ))
            with open(f"{self.training_images_save_path}/{img.name}", "w+b") as image_file:
                image_file.write(img.content)

        display(HBox(image_widgets))

        print(f"✅ Training images uploaded successfully.")

        return images