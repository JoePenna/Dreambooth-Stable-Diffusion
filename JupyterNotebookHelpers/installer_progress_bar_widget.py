from IPython.display import clear_output
from ipywidgets import widgets, Layout


class InstallerProgressBar:

    def __init__(
            self,
            style={'description_width': '150px'},
            layout=Layout(width="400px"),
    ):
        self.style = style
        self.layout = layout
        self.show_detailed_output = False

        self.installer_progress_bar_widget = widgets.IntProgress(
            value=0,
            min=0,
            max=0,
            description='Installing: ',
            bar_style='info',
            orientation='horizontal'
        )

        self.output = widgets.Output()

    def show(self, install_commands):
        clear_output()
        self.installer_progress_bar_widget.max = len(install_commands)
        display(self.installer_progress_bar_widget, self.output)

    def increment(self, step: int):
        self.installer_progress_bar_widget.value = step + 1

    def close(self):
        self.installer_progress_bar_widget.close()
