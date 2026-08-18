"""
Slice3D extension for Scivianna.

This module provides 3D clip plane controls for slicing geometry.
Works with both Panel3D and Panel2D (when using VTK backend).
"""

from typing import TYPE_CHECKING, Any, Dict, Union

import panel as pn
import panel_material_ui as pmui

from scivianna.extension.extension import Extension
from scivianna.icon import get_icon
from scivianna.logging_config import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from scivianna.panel.panel_2d import Panel2D
    from scivianna.panel.panel_3d import Panel3D
    from scivianna.panel.visualisation_panel import VisualizationPanel
    from scivianna.plotter_2d.polygon.vtk_2d import VTK2DPolygonPlotter
    from scivianna.plotter_3d.vtk_3d_plotter import Plotter3D
    from scivianna.slave import ComputeSlave

pn.extension(notifications=True)


class Slice3D(Extension):
    """Extension to control the clip plane for slicing geometry.

    This extension provides interactive controls to enable/disable clipping,
    select the clipping axis (X, Y, Z), and adjust the clip plane position.
    
    Works with both Panel3D and Panel2D (when using VTK backend).
    """

    def __init__(self, slave: "ComputeSlave", plotter: Union["Plotter3D", "VTK2DPolygonPlotter"], panel: Union["Panel3D", "Panel2D"]):
        """Constructor of the slice 3D extension.

        Parameters
        ----------
        slave : ComputeSlave
            Slave computing the displayed data
        plotter : Plotter3D or VTK2DPolygonPlotter
            Plotter with VTK visualization (3D or 2D VTK backend)
        panel : Panel3D or Panel2D
            Panel to which the extension is attached
        """
        from scivianna.panel.panel_2d import Panel2D
        super().__init__(
            "Slice Plane",
            get_icon("view_in_ar"),
            slave,
            plotter,
            panel,
        )

        self.is_2d = isinstance(panel, Panel2D)

        self.description = """
The slice plane extension lets you clip the geometry to see inside.

Controls:
- Enable/disable clipping with the checkbox
- Select axis (X, Y, Z) for the clip plane normal
- Adjust position with the slider
- Keyboard shortcuts in viewer: C (toggle), X/Y/Z (axis)
"""

        # Clip enabled checkbox
        self.plane_enabled_checkbox = pmui.Checkbox(
            label="Enable slice plane", value=False, width=280, visible = not self.is_2d
        )
        self.plane_enabled_checkbox.param.watch(self._on_plane_enabled_change, "value")

        # Clip enabled checkbox
        self.clip_enabled_checkbox = pmui.Checkbox(label="Enable clipping", value=False, width=280, visible = not self.is_2d)
        self.clip_enabled_checkbox.param.watch(self._on_clip_enabled_change, "value")

        # Edges visible checkbox
        self.edges_visible_checkbox = pmui.Checkbox(
            label="Set edges visible", value=True, width=280
        )
        self.edges_visible_checkbox.param.watch(self._on_edges_visible_change, "value")

        # Info option checkbox
        self.info_enabled_checkbox = pmui.Checkbox(
            label="Enable info option", value=True, width=280
        )
        self.info_enabled_checkbox.param.watch(self._on_info_enabled_change, "value")

        # Axis selector
        self.clip_axis_select = pmui.Select(
            label="Clip axis", options=["x", "y", "z"], value="z", width=280, visible = not self.is_2d
        )
        self.clip_axis_select.param.watch(self._on_clip_axis_change, "value")

        # Force 2D view
        self.view_2d_checkbox = pmui.Checkbox(
            label="Force 2D view", value = self.is_2d, width=280
        )
        self.view_2d_checkbox.param.watch(self._on_2d_change, "value")

        # Watch for clip plane changes on the underlying VTK plotter
        self._get_vtk_plotter().param.watch(self._on_plane_change, "clip_origin")
        self._get_vtk_plotter().param.watch(self._on_plane_change, "clip_normal")

    def _on_2d_change(self, event):
        self.plotter.plotter.set_view_2d_mode(self.view_2d_checkbox.value)

    def _get_vtk_plotter(self):
        """Returns the underlying VTK plotter instance.
        
        Returns
        -------
        VTKPlotter
            The underlying vtk_js plotter
        """
        return self.plotter.plotter

    def _on_plane_enabled_change(self, event):
        """Handle plane enabled checkbox change."""
        self.plotter.set_plane_enabled(event.new)

    def _on_clip_enabled_change(self, event):
        """Handle clip enabled checkbox change."""
        self.plotter.set_clip_enabled(event.new)

    def _on_clip_axis_change(self, event):
        """Handle clip axis selector change."""
        self.plotter.set_clip_axis(event.new)

    def _on_edges_visible_change(self, event):
        """Handle edges visible checkbox change."""
        self.plotter.set_edges_visible(event.new)

    def _on_info_enabled_change(self, event):
        """Handle info enabled checkbox change."""
        self.plotter.set_info(event.new)

    def _on_plane_change(self, event):
        vtk_plotter = self._get_vtk_plotter()
        logger.debug(
            "Clip plane - origin: %s, normal: %s",
            vtk_plotter.clip_origin,
            vtk_plotter.clip_normal,
        )

    def make_gui(self) -> pn.viewable.Viewable:
        """Returns a panel viewable to display in the extension tab.

        Returns
        -------
        pn.viewable.Viewable
            Viewable to display in the extension tab
        """
        return pn.Column(
            self.edges_visible_checkbox,
            self.info_enabled_checkbox,
            self.plane_enabled_checkbox,
            self.clip_enabled_checkbox,
            self.clip_axis_select,
            self.view_2d_checkbox
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
            "edges_visible": self.edges_visible_checkbox.value,
            "info_enabled": self.info_enabled_checkbox.value,
            "axis": self.clip_axis_select.value,
            "2d": self.view_2d_checkbox.value
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

        if "edges_visible" in info_dict:
            extension.edges_visible_checkbox.value = info_dict["edges_visible"]

        if "info_enabled" in info_dict:
            extension.info_enabled_checkbox.value = info_dict["info_enabled"]

        if "axis" in info_dict:
            extension.clip_axis_select.value = info_dict["axis"]

        if "2d" in info_dict:
            extension.view_2d_checkbox.value = info_dict["2d"]

        extension._restoring = False

        return extension

    def on_updated_data(self, data):
        """Called when the data is updated.

        If more than 100k cells, disables the info option once with a warning message.

        Parameters
        ----------
        data : Data2D or Data3D
            The updated data object.
        """
        # Get number of cells from the data object
        if hasattr(data, 'cell_ids'):
            n_cells = len(data.cell_ids)
        else:
            n_cells = getattr(self.plotter, 'n_cells', 0)
        
        # Only disable info once for high cell count (check the flag, not the current state)
        info_disabled_flag = getattr(self.plotter, '_info_disabled_high_cell_count', False)
        if n_cells > 100_000 and not info_disabled_flag:
            self.plotter._info_disabled_high_cell_count = True
            self.info_enabled_checkbox.value = False
            self.plotter.set_info(False)
            warning_message = f"Info option disabled due to high cell count ({n_cells} cells). This feature harms performance with large datasets."
            logger.warning(
                warning_message
            )
            pn.state.notifications.warning(warning_message)
