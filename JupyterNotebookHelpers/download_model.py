from IPython.display import clear_output
import shutil
from huggingface_hub import hf_hub_download
from ipywidgets import widgets, Layout, HBox


class SDModelOption:
    def __init__(self, repo_id, filename, manual=False):
        self.repo_id = repo_id
        self.filename = filename
        self.manual = manual

    def download(self):
        if self.is_valid():
            print(f"Downloading '{self.repo_id}/{self.filename}'")
            return hf_hub_download(
                repo_id=self.repo_id,
                filename=self.filename
            )
        else:
            raise Exception(
                f"Model not valid. repo_id: {self.repo_id} or filename: {self.filename} are missing or invalid.")

    def is_valid(self):
        return (self.repo_id is not None and self.repo_id != '') and \
            (self.filename is not None and self.filename != '' and '.ckpt' in self.filename)


class DownloadModel:

    def __init__(
            self,
            style={'description_width': '150px'},
            layout=Layout(width="400px")
    ):
        self.style = style
        self.layout = layout

        self.model_definitions = [
            SDModelOption(repo_id="panopstor/EveryDream", filename="sd_v1-5_vae.ckpt"),
            SDModelOption(repo_id="runwayml/stable-diffusion-v1-5", filename="v1-5-pruned-emaonly.ckpt"),
            SDModelOption(repo_id="runwayml/stable-diffusion-v1-5", filename="v1-5-pruned.ckpt"),
            SDModelOption(repo_id="CompVis/stable-diffusion-v-1-4-original", filename="sd-v1-4.ckpt"),
            SDModelOption(repo_id="CompVis/stable-diffusion-v-1-4-original", filename="sd-v1-4-full-ema.ckpt"),
            SDModelOption(repo_id=None, filename=None, manual=True),
        ]
        self.available_models = [
            ("sd_v1-5_vae.ckpt - 4.27gb - EveryDream (incl. vae) - Recommended", 0),
            ("v1-5-pruned-emaonly.ckpt - 4.27gb - runwayml", 1),
            ("v1-5-pruned.ckpt - 7.7gb - runwayml", 2),
            ("sd-v1-4.ckpt - 4.27gb - CompVis", 3),
            ("sd-v1-4-full-ema.ckpt - 7.7gb - CompVis", 4),
            ("Manual", 5),
        ]

        self.last_selected_index = 0

        self.model_options = widgets.Dropdown(
            options=self.available_models,
            value=0,
            description="Model: ",
            style=style,
            layout=layout,
        )
        self.model_options.observe(self.model_options_changed)

        self.model_repo_id_input = widgets.Text(
            placeholder=self.model_definitions[0].repo_id,
            description='Selected Repo Id: ',
            value='',
            disabled=True,
            style=style,
            layout=layout,
        )

        self.model_filename_input = widgets.Text(
            placeholder=self.model_definitions[0].filename,
            description='Selected Filename: ',
            value='',
            disabled=True,
            style=style,
            layout=layout,
        )

        self.download_model_button = widgets.Button(
            description="Download Model",
            disabled=False,
            button_style='success',
            tooltip='Download Model',
            icon='download',
            style=self.style,
            layout=self.layout,
        )
        self.download_model_button.on_click(self.download_model)
        self.output = widgets.Output()

    def show_form(self):
        display(self.model_options)
        display(self.model_repo_id_input)
        display(self.model_filename_input)
        display(self.download_model_button, self.output)

    def download_model(self, b):
        with self.output:
            self.output.clear_output()
            selected_model = self.get_selected_model()

            if selected_model.manual:
                selected_model.repo_id = self.model_repo_id_input.value
                selected_model.filename = self.model_filename_input.value

            if selected_model.is_valid():
                self.download_model_button.disabled = True
                try:
                    downloaded_model_path = selected_model.download()

                    # copy the downloaded model to the root
                    shutil.copy(downloaded_model_path, f"{selected_model.filename}")

                    print(f"✅ '{selected_model.repo_id}/{selected_model.filename}' successfully downloaded")
                except:
                    print("❌ Error downloading the model.")

                # Cleanup
                self.download_model_button.disabled = False
            else:
                print("❌ Specified model is invalid.")

    def model_options_changed(self, *args):
        if self.last_selected_index is not self.model_options.value:
            self.last_selected_index = self.model_options.value
            selected_model = self.get_selected_model()

            if selected_model.manual:
                self.model_repo_id_input.disabled = False
                self.model_repo_id_input.placeholder = 'runwayml/stable-diffusion-v1-5'
                self.model_repo_id_input.description = "Enter Repo Id: "
                self.model_filename_input.disabled = False
                self.model_filename_input.placeholder = 'v1-5-pruned-emaonly.ckpt'
                self.model_filename_input.description = "Enter Filename: "
            else:
                self.model_repo_id_input.disabled = True
                self.model_repo_id_input.placeholder = selected_model.repo_id
                self.model_repo_id_input.description = "Selected Repo Id: "
                self.model_filename_input.disabled = True
                self.model_filename_input.placeholder = selected_model.filename
                self.model_filename_input.description = "Selected Filename: "

    def get_selected_model(self) -> SDModelOption:
        return self.model_definitions[self.model_options.value]
