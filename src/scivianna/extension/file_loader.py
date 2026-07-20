import functools
import os
from pathlib import Path
from typing import Dict, TYPE_CHECKING
import panel as pn
from scivianna.component.server_file_browser import ServerFileBrowser
from scivianna.constants import GEOMETRY
from scivianna.extension.extension import Extension
from scivianna.icon import get_icon
from scivianna.plotter_2d.generic_plotter import Plotter2D

if TYPE_CHECKING:
    from scivianna.panel.visualisation_panel import VisualizationPanel
    from scivianna.slave import ComputeSlave


class FileLoader(Extension):
    """Extension to load files and send them to the slave."""

    def __init__(
        self,
        slave: "ComputeSlave",
        plotter: Plotter2D,
        panel: "VisualizationPanel"
    ):
        """Constructor of the extension, saves the slave and the panel

        Parameters
        ----------
        slave : ComputeSlave
            Slave computing the displayed data
        plotter : Plotter2D
            Figure plotter
        panel : VisualizationPanel
            Panel to which the extension is attached
        """
        super().__init__(
            "Load new files",
            get_icon("file_open"),
            slave,
            plotter,
            panel,
        )

        self.description = """
The file loader extension lets you browse files on the server file system to provide it to the code interface.
"""
        """
            Widget To send the input file
        """

        self.file_browsers: Dict[str, ServerFileBrowser] = {}

        def load_file(event, browser_name: str):
            """Request the slave to load an input file. If the file is a geometry file, the slave is reseted

            Parameters
            ----------
            data : Any
                File input data property.
            """
            file_path = self.file_browsers[browser_name].selected_file

            if file_path is not None:
                if browser_name == GEOMETRY:
                    self.slave.reset()

                self.slave.read_file(file_path, browser_name)

                self.panel.trigger_on_file_load(file_path, browser_name)

        file_input_list = self.slave.get_file_input_list()

        folder_path = None
        loaded_files = self.slave.file_read
        for path, _ in loaded_files:
            if os.path.isfile(str(path)):
                folder_path = Path(path).parent

        for name, _ in file_input_list:
            self.file_browsers[name] = ServerFileBrowser(
                folder_path=folder_path,
                name=str(name),
                width=280
            )
            self.file_browsers[name].param.watch(
                functools.partial(load_file, browser_name=name), "selected_file"
            )

        self.file_loader_list = []
        for fi in self.file_browsers:
            self.file_loader_list.append(
                pn.pane.Markdown(f"{fi} file browser", margin=(0, 0, 0, 0))
            )
            self.file_loader_list.append(self.file_browsers[fi])

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
            *self.file_loader_list,
            margin=(0, 0, 10, 10),
        )

    def to_json(self) -> dict:
        """Returns a dictionary with the information required to rebuild the extension.

        Returns
        -------
        dict
            Information dictionary
        """
        loaded_files = {}
        for name, browser in self.file_browsers.items():
            if browser.selected_file:
                loaded_files[name] = browser.selected_file
        return {
            "loaded_files": loaded_files
        }

    @classmethod
    def from_json(cls, extension: "FileLoader", info_dict: dict) -> "FileLoader":
        """Restores the extension from its information dict.

        Parameters
        ----------
        extension : FileLoader
            Extension instance to restore
        info_dict : dict
            Dictionary containing extension state information

        Returns
        -------
        FileLoader
            Restored extension
        """
        extension._restoring = True

        # Note: Files are not reloaded here as they may not exist at restore time
        # The loaded_files info is stored for reference but files need to be loaded manually
        loaded_files = info_dict.get("loaded_files", {})
        for name, file_path in loaded_files.items():
            if name in extension.file_browsers:
                browser = extension.file_browsers[name]
                # Set the selected file path in the browser if it exists
                if os.path.exists(file_path):
                    browser.selected_file = file_path

        extension._restoring = False
        return extension
