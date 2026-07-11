from typing import Any, Dict, TYPE_CHECKING
import panel as pn
import panel_material_ui as pmui

from scivianna.extension.extension import Extension

if TYPE_CHECKING:
    from scivianna.panel.visualisation_panel import VisualizationPanel
    from scivianna.slave import ComputeSlave
    from scivianna.plotter_3d.vtk_3d_plotter import Plotter3D
    from scivianna.panel.panel_3d import Panel3D


class Slice3D(Extension):
    """Extension to control the 3D clip plane for slicing geometry.
    
    This extension provides interactive controls to enable/disable clipping,
    select the clipping axis (X, Y, Z), and adjust the clip plane position.
    """

    def __init__(
        self,
        slave: "ComputeSlave",
        plotter: "Plotter3D",
        panel: "Panel3D"
    ):
        """Constructor of the slice 3D extension.

        Parameters
        ----------
        slave : ComputeSlave
            Slave computing the displayed data
        plotter : Plotter3D
            3D plotter with VTK visualization
        panel : VisualizationPanel
            Panel to which the extension is attached
        """
        super().__init__(
            "Slice Plane",
            "view_in_ar",
            slave,
            plotter,
            panel,
        )

        self.description = """
The slice plane extension lets you clip the 3D geometry to see inside.

Controls:
- Enable/disable clipping with the checkbox
- Select axis (X, Y, Z) for the clip plane normal
- Adjust position with the slider
- Keyboard shortcuts in viewer: C (toggle), X/Y/Z (axis), ↑/↓ (move)
"""

        # Clip enabled checkbox
        self.plane_enabled_checkbox = pmui.Checkbox(
            name="Enable slice plane",
            value=False,
            width=280
        )
        self.plane_enabled_checkbox.param.watch(self._on_plane_enabled_change, "value")

        # Clip enabled checkbox
        self.clip_enabled_checkbox = pmui.Checkbox(
            name="Enable clipping",
            value=False,
            width=280
        )
        self.clip_enabled_checkbox.param.watch(self._on_clip_enabled_change, "value")

        # Axis selector
        self.clip_axis_select = pmui.Select(
            name="Clip axis",
            options=["x", "y", "z"],
            value="z",
            width=280
        )
        self.clip_axis_select.param.watch(self._on_clip_axis_change, "value")

        self.plotter.plotter.param.watch(self._on_plane_change, "clip_origin")
        self.plotter.plotter.param.watch(self._on_plane_change, "clip_normal")

    def _on_plane_enabled_change(self, event):
        """Handle clip enabled checkbox change."""
        if hasattr(self.plotter, 'plotter'):
            vtk_plotter = self.plotter.plotter
            vtk_plotter.set_plane_enabled(event.new)

    def _on_clip_enabled_change(self, event):
        """Handle clip enabled checkbox change."""
        if hasattr(self.plotter, 'plotter'):
            vtk_plotter = self.plotter.plotter
            vtk_plotter.clip_enabled = event.new

    def _on_clip_axis_change(self, event):
        """Handle clip axis selector change."""
        if hasattr(self.plotter, 'plotter'):
            vtk_plotter = self.plotter.plotter
            vtk_plotter.set_clip_axis(event.new)

    def _on_plane_change(self, event):
        print(f"Plane origin {self.plotter.plotter.clip_origin}")
        print(f"Plane normal {self.plotter.plotter.clip_normal}")

    def make_gui(self) -> pn.viewable.Viewable:
        """Returns a panel viewable to display in the extension tab.

        Returns
        -------
        pn.viewable.Viewable
            Viewable to display in the extension tab
        """
        return pn.Column(
            self.plane_enabled_checkbox,
            self.clip_enabled_checkbox,
            self.clip_axis_select
        )

    def on_file_load(self, file_path: str, file_key: str):
        """Function called when the user loads a new file.

        Parameters
        ----------
        file_path : str
            Path of the loaded file
        file_key : str
            Key associated to the loaded file
        """
        # Reset to default state
        # self.clip_enabled_checkbox.value = True
        # self.clip_axis_select.value = "z"
        pass

    def to_json(self) -> dict:
        """Returns a dictionary with the information required to rebuild the extension.

        Returns
        -------
        dict
            Information dictionary
        """
        return {
            "plane_enabled": self.plane_enabled_checkbox.value,
            "clip_enabled": self.clip_enabled_checkbox.value,
            "axis": self.clip_axis_select.value,
        }

    @classmethod
    def from_json(cls, extension: "Slice3D", info_dict: dict) -> "Slice3D":
        """Restores the extension from its information dict.

        Parameters
        ----------
        extension : Slice3D
            Extension instance to restore
        info_dict : dict
            Dictionary containing extension state information

        Returns
        -------
        Slice3D
            Restored extension
        """
        extension._restoring = True

        if "plane_enabled" in info_dict:
            extension.plane_enabled_checkbox.value = info_dict["plane_enabled"]

        if "clip_enabled" in info_dict:
            extension.clip_enabled_checkbox.value = info_dict["clip_enabled"]

        if "axis" in info_dict:
            extension.clip_axis_select.value = info_dict["axis"]

        extension._restoring = False

        return extension
