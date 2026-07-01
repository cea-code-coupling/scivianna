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
        self.clip_enabled_checkbox = pmui.Checkbox(
            name="Enable clipping",
            value=True,
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

        # Position slider (0 to 1, normalized)
        self.clip_position_slider = pmui.FloatSlider(
            name="Position",
            start=0.0,
            end=1.0,
            step=0.01,
            value=0.5,
            width=280
        )
        self.clip_position_slider.param.watch(self._on_clip_position_change, "value")

        # Store current state
        self._clip_bounds = [0.0, 1.0]  # Will be updated from geometry bounds

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

    def _on_clip_position_change(self, event):
        """Handle clip position slider change.
        
        Maps normalized slider value (0-1) to geometry bounds.
        """
        if hasattr(self.plotter, 'plotter'):
            vtk_plotter = self.plotter.plotter
            
            # Get current normal direction
            normal = vtk_plotter.clip_normal
            
            # Map slider value to bounds along the normal axis
            t = event.new
            position = self._clip_bounds[0] * (1 - t) + self._clip_bounds[1] * t
            
            # Move clip plane along its normal
            current_origin = list(vtk_plotter.clip_origin)
            
            # Find the dominant axis of the normal
            max_idx = 0
            max_val = abs(normal[0])
            for i in range(1, 3):
                if abs(normal[i]) > max_val:
                    max_val = abs(normal[i])
                    max_idx = i
            
            # Update origin along dominant axis
            current_origin[max_idx] = position
            vtk_plotter.clip_origin = current_origin

    def update_clip_bounds(self):
        """Update clip plane bounds from geometry.
        
        Called when geometry changes to update the slider range.
        """
        if hasattr(self.plotter, 'plotter'):
            # Get geometry bounds from the VTK plotter's data
            # For now, use default bounds - these would be updated from actual geometry
            self._clip_bounds = [0.0, 1.0]

    def make_gui(self) -> pn.viewable.Viewable:
        """Returns a panel viewable to display in the extension tab.

        Returns
        -------
        pn.viewable.Viewable
            Viewable to display in the extension tab
        """
        return pn.Column(
            self.clip_enabled_checkbox,
            self.clip_axis_select,
            self.clip_position_slider,
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
        self.clip_enabled_checkbox.value = True
        self.clip_axis_select.value = "z"
        self.clip_position_slider.value = 0.5
        self.update_clip_bounds()

    def to_json(self) -> dict:
        """Returns a dictionary with the information required to rebuild the extension.

        Returns
        -------
        dict
            Information dictionary
        """
        return {
            "enabled": self.clip_enabled_checkbox.value,
            "axis": self.clip_axis_select.value,
            "position": self.clip_position_slider.value,
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

        if "enabled" in info_dict:
            extension.clip_enabled_checkbox.value = info_dict["enabled"]

        if "axis" in info_dict:
            extension.clip_axis_select.value = info_dict["axis"]

        if "position" in info_dict:
            extension.clip_position_slider.value = info_dict["position"]

        extension._restoring = False

        return extension
