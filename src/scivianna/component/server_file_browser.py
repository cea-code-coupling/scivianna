import os
from pathlib import Path
from typing import List, Tuple, Union

import panel_material_ui as pmui
import param
from panel.viewable import Viewer


class ServerFileBrowser(Viewer):
    """The ServerFileBrowser is a file explorer based on a Select widget that lets browse on the server side."""

    current_folder: Path
    """Crrently browsing folder"""
    selected_file: param.String = param.String()
    """Full path of currently selected file"""

    def __init__(self, folder_path: Union[str, Path] = None, width: int = 300, *args, **kwargs):
        """File browser on the server side based on a pmui.Select widget

        Parameters
        ----------
        folder_path : Union[str, Path], optional
            Base folder path, by default None

        Raises
        ----------
        ValueError
            Not None provided folder path does not exist
        """
        super().__init__(*args, **kwargs)

        if folder_path is None:
            self.current_folder = Path(__file__).parent
        else:
            if not os.path.isdir(folder_path):
                raise ValueError(f"Provided folder path does not exist {folder_path}")
            self.current_folder = Path(folder_path)

        self.selected_file = ""

        self.select = pmui.Select(
            groups={
                "Directories": ["-- Cancel --", ".", ".."]
                + [e for e in os.listdir(self.current_folder) if os.path.isdir(e)],
                "Files": [e for e in os.listdir(self.current_folder) if os.path.isfile(e)],
            },
            value=".",
            size="small",
            margin=(0, 0, 0, 0),
            width=width,
        )
        self.select.param.watch(self.on_value_change, "dropdown_open")

        folder_list, file_list = self._get_folder_sorted_content(self.current_folder)
        self.select.disabled_options = []
        self.select.groups = {
            "Directories": ["-- Cancel --", ".", ".."] + folder_list,
            "Files": file_list,
        }

    def on_value_change(self, event: param.Event):
        """Event triggered by a dropdown open change, it updates the content of the select if a value was selected:
                -   If a folder is selected, the select updates its content to the folder content, and the folder path is saved;
                -   If a file is selected, its path is saved in selected_file.

        Parameters
        ----------
        event : param.Event
            Change event
        """
        # Checking if the dropdown is closing:
        #   -   option was selected
        #       -   either the value changed, or the folder changed (if .. is selected)
        #   -   the panel was closed
        #       -   neither the value not the folder was changed
        if self.select.value == "-- Cancel --":
            self.selected_file = ""
            return

        if not self.select.dropdown_open:
            if self.select.value is None:
                pass

            elif os.path.isfile(self.current_folder / self.select.value):
                print("Selected file : ", self.current_folder / self.select.value)
                self.selected_file = str(self.current_folder / self.select.value)

            elif os.path.isdir(self.current_folder / self.select.value):
                self.current_folder = (self.current_folder / self.select.value).resolve()
                try:
                    folder_list, file_list = self._get_folder_sorted_content(self.current_folder)
                    self.select.disabled_options = []
                    self.select.groups = {
                        "Directories": ["-- Cancel --", ".", ".."] + folder_list,
                        "Files": file_list,
                    }
                except PermissionError:
                    # We don't have access to the folder, so we go back to the former
                    self.select.disabled_options += [self.select.value]
                    self.current_folder = (self.current_folder / "..").resolve()

                self.selected_file = ""
                self.select.dropdown_open = True

    def _get_folder_sorted_content(self, path: Path) -> Tuple[List[str], List[str]]:
        """Returns the alphabetically sorded list of folders and files in the provided path, if access is allowed

        Parameters
        ----------
        path : Path
            Folder path to display

        Returns
        -------
        Tuple[List[str], List[str]]
            List of folder, list of files

        Raises
        ------
        PermissionError
            Access to the requested folder is denied
        """
        try:
            folder_list = [e for e in os.listdir(path) if os.path.isdir(path / e)]
            file_list = [e for e in os.listdir(path) if os.path.isfile(path / e)]

            folder_list.sort()
            file_list.sort()

            return folder_list, file_list

        except PermissionError:
            raise PermissionError

    def __panel__(self):
        """Redered panel

        Returns
        -------
        pn.viewable.Viewable
            pmui.Select widget displayed
        """
        return self.select


if __name__ == "__main__":
    ServerFileBrowser().show()
