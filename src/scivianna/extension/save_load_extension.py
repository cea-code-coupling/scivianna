import os
from pathlib import Path
from typing import TYPE_CHECKING

import panel as pn
import panel_material_ui as pmui

import scivianna
from scivianna.component.server_file_browser import ServerFileBrowser
from scivianna.extension.extension import Extension
from scivianna.icon import get_icon
from scivianna.plotter_2d.generic_plotter import Plotter2D
from scivianna.slave import ComputeSlave

if TYPE_CHECKING:
    from scivianna.panel.visualisation_panel import VisualizationPanel


class SaveLoadExtension(Extension):
    def __init__(
        self, slave: ComputeSlave, plotter: Plotter2D, panel: "VisualizationPanel"
    ):
        """Constructor of the extension, saves the slave and the panel

        Parameters
        ----------
        slave : ComputeSlave
            Slave computing the displayed data
        plotter : Plotter1D
            Figure plotter
        panel : VisualizationPanel
            Panel to which the extension is attached
        """
        super().__init__("Save/Load slave", get_icon("save"), slave, plotter, panel)

        self.description = """
The save/load extension allows you to save a frame state for later fast loading.

If include_files, all loaded data are saved.

Else, only the view is saved.
"""

        self.folder_path = None
        loaded_files = self.slave.file_read
        for path, _ in loaded_files:
            if os.path.isfile(str(path)):
                self.folder_path = Path(path).parent

        self.include_file_box = pmui.Checkbox(
            label="Include files ", value=False, width=280
        )
        self.save_path = pmui.TextInput(
            label="Absolute file path",
            value=str(self.folder_path / "save_file.pkl"),
            width=280,
        )
        self.save_button = pmui.Button(label="Save", width=280, icon=(Path(scivianna.__file__).parent / "icon" / "save_alt.svg").read_text().strip())
        self.save_button.on_click(self.save_file)

        self.load_path = ServerFileBrowser(folder_path=self.folder_path, width=280)
        self.load_button = pmui.Button(label="Load", width=280, icon=(Path(scivianna.__file__).parent / "icon" / "file_upload.svg").read_text().strip())
        self.load_button.on_click(self.load_file)

    def save_file(self, *args, **kwargs):
        """Save frame in file
        """
        self.slave.save(self.save_path.value, self.include_file_box.value)

    def load_file(self, *args, **kwargs):
        """Load frame from file
        """
        if not os.path.isfile(self.load_path.selected_file):
            pn.state.notifications.error(
                f"File {self.load_path.selected_file} does not exits."
            )
            return
        self.slave.load(self.load_path.selected_file, self.include_file_box.value)
        self.panel.trigger_on_file_load(self.load_path.selected_file, None)
        self.panel.recompute()

    def make_gui(
        self,
    ) -> pn.viewable.Viewable:
        """Returns a panel viewable to display in the extension tab.

        Returns
        -------
        pn.viewable.Viewable
            Viewable to display in the extension tab
        """
        return pn.Column(
            self.include_file_box,
            pmui.Typography("Save to file"),
            self.save_path,
            self.save_button,
            pmui.Typography("Load from file"),
            self.load_path,
            self.load_button,
            width=280
        )
