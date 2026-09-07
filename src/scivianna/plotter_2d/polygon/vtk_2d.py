"""
VTK-based 2D polygon plotter for Scivianna.

This module provides a 2D polygon renderer using VTK/vtk.js via the scivianna_vtk
plotter component with 2D mode enabled.
"""
from typing import Callable, List, Tuple

import numpy as np
import panel as pn
import pyvista as pv

import shapely

from scivianna.data.data2d import Data2D
from scivianna.logging_config import get_logger
from scivianna.plotter_2d.generic_plotter import Plotter2D
from scivianna.utils.polygonize_tools import polygons_to_polydata
from scivianna_vtk.plotter import VTKPlotter
from scivianna.utils.color_tools import beautiful_color_maps

logger = get_logger(__name__)


class VTK2DPolygonPlotter(Plotter2D):
    """2D geometry plotter based on VTK/vtk.js with 2D top-down view mode."""

    def __init__(self):
        """Creates the VTKPlotter component with 2D mode enabled."""
        # Create the VTK plotter and enable 2D mode
        self.plotter = VTKPlotter(sizing_mode="stretch_both", margin=0)
        self.plotter.set_view_2d_mode(True)
        self.plotter.set_clip_enabled(False)
        self.plotter.set_edges_visible(True)
        self.plotter.set_info(True)

        # Store reference to current polydata for updates
        self._current_polydata = None

        # Color mapper state
        self._colorbar_visible = False
        self._colorbar_min = 0.0
        self._colorbar_max = 1.0
        self._colormap_name = "BuRd"

        # Callbacks
        self.on_mouse_move_callback = None
        self.on_clic_callback = None

        # Watch for hover and click events
        self.plotter.param.watch(self._on_hover, "hover_position")
        self.plotter.param.watch(self._on_click, "clicks")

        # Store axes information
        self._u = np.array([1.0, 0.0, 0.0])
        self._v = np.array([0.0, 1.0, 0.0])
        self._origin = np.array([0.0, 0.0, 0.0])

        # Flag to track if info was disabled due to high cell count
        self._info_disabled_high_cell_count = False

    def _on_hover(self, event):
        """Handle hover events and forward to callback."""
        if self.on_mouse_move_callback is not None:
            pos = self.plotter.hover_position
            cell_id = self.plotter.hover_cell_id
            if not (any(np.isnan(pos)) or cell_id is None):
                self.on_mouse_move_callback(
                    screen_location=(None, None),  # VTK doesn't provide screen coords
                    space_location=tuple(pos),
                    cell_id=cell_id,
                )

    def _on_click(self, event):
        """Handle click events and forward to callback."""
        if self.on_clic_callback is not None:
            pos = self.plotter.hover_position
            cell_id = self.plotter.hover_cell_id
            if not (any(np.isnan(pos)) or cell_id is None):
                self.on_clic_callback(
                    screen_location=(None, None),
                    space_location=tuple(pos),
                    cell_id=cell_id,
                )

    def display_borders(self, display: bool):
        """
        Display or hide the figure borders and axis.

        Parameters
        ----------
        display : bool
            Display if true, hides otherwise.
        """
        # VTK 2D plotter doesn't have traditional borders
        # Could potentially show axes or grid if needed
        pass

    def update_colorbar(self, display: bool, value_range: Tuple[float, float]):
        """
        Display or hide the color bar, update its range if provided.

        Parameters
        ----------
        display : bool
            Display or hide the color bar.
        value_range : Tuple[float, float]
            New colormap range (min, max).
        """
        self._colorbar_visible = display

        if display and value_range is not None and value_range[0] is not None and value_range[1] is not None:
            self._colorbar_min = value_range[0]
            self._colorbar_max = value_range[1]
            self.plotter.set_colorbar_enabled(True)
            self.plotter.set_colorbar_range(self._colorbar_min, self._colorbar_max)
        else:
            self.plotter.set_colorbar_enabled(False)

    def set_color_map(self, color_map_name: str):
        """
        Set the colorbar color map name.

        Note: Colors are fetched from the Data2D object, not computed by the plotter.
        Changing the colormap requires recomputing the Data2D with the new colormap.

        Parameters
        ----------
        color_map_name : str
            Color map name (stored for reference).
        """
        self._colormap_name = color_map_name
        self.plotter.set_colorbar_colors(np.array(beautiful_color_maps[color_map_name]) / 255.)


    def plot_2d_frame(self, data: Data2D):
        """
        Add a new plot to the figure from a set of polygons.

        Parameters
        ----------
        data : Data2D
            Data2D object containing the geometry to plot.
        """
        if data is None:
            raise ValueError("Provided polygons is None, an error occured before, please check the terminal.")
        if len(data.cell_ids) == 0:
            logger.warning("No polygons to plot")
            return

        # Convert polygons to PolyData (colors are taken from data.cell_colors)
        polydata = polygons_to_polydata(
            data.get_polygons(),
            data.cell_values,
            data.cell_colors,
            data.cell_edge_colors
        )

        # Store reference
        self._current_polydata = polydata

        # Update plotter
        self.plotter.update_polydata(polydata)
        self.plotter.set_view_2d_mode(True)

    def update_2d_frame(self, data: Data2D):
        """
        Update the plot with new geometry and data.

        Parameters
        ----------
        data : Data2D
            Data2D object containing the updated geometry and data.
        """
        print("updating frame")
        self.plot_2d_frame(data)

    def update_colors(self, data: Data2D):
        """
        Update only the colors of the displayed polygons.

        Parameters
        ----------
        data : Data2D
            Data2D object containing the updated color data.
        """
        if self._current_polydata is None:
            # No existing data, do full plot
            self.plot_2d_frame(data)
            return

        colors_array = np.array(data.cell_colors, dtype=float) / 255.0
        colors_edge_array = np.array(data.cell_edge_colors, dtype=float) / 255.0

        if (
            self._current_polydata.cell_data["rgb"].shape != colors_array.shape
            or self._current_polydata.cell_data["cell_value"].shape != np.asarray(data.cell_values).shape
        ):
            self.plot_2d_frame(data)
            return

        self._current_polydata.cell_data["cell_value"] = np.array(data.cell_values, dtype=float)
        self._current_polydata.cell_data["rgb"] = colors_array
        self._current_polydata.cell_data["edge_rgb"] = colors_edge_array[:, :3]

        # Update plotter
        self.plotter.update_colors(self._current_polydata)
        self.plotter.set_view_2d_mode(True)

    def _set_callback_on_range_update(self, callback):
        """
        Set a callback to update the x and y ranges in the GUI.

        Parameters
        ----------
        callback : Callable
            Function that takes x0, x1, y0, y1 as arguments.
        """
        # VTK plotter doesn't have built-in range update events like Bokeh
        # This would need to be implemented via custom JS or camera events
        logger.debug("Range update callback not implemented for VTK 2D plotter")

    def make_panel(self) -> pn.viewable.Viewable:
        """
        Make the Panel viewable displayed in the web app.

        Returns
        -------
        pn.viewable.Viewable
            Displayed viewable.
        """
        return self.plotter

    def _disable_interactions(self, disable: bool):
        """
        Disable the plot interactions for multi-panel web-app resizing.

        Parameters
        ----------
        disable : bool
            Disable if True, enable if False.
        """
        # VTK plotter interactions are handled differently
        # Could potentially lock camera or disable interaction in JS
        pass

    def set_axes(
        self,
        u: Tuple[float, float, float],
        v: Tuple[float, float, float],
        origin: Tuple[float, float, float],
    ):
        """
        Store the u, v axes of the current plot.

        Parameters
        ----------
        u : Tuple[float, float, float]
            Horizontal axis direction vector.
        v : Tuple[float, float, float]
            Vertical axis direction vector.
        origin : Tuple[float, float, float]
            Frame center.
        """
        self._u = np.array(u)
        self._v = np.array(v)
        self._origin = np.array(origin)

        # In 2D mode, the view is top-down (XY plane)
        # The u, v, origin define the coordinate transformation for the data
        self.plotter.hover_u_vector = list(u)
        self.plotter.hover_v_vector = list(v)
        self.plotter.hover_origin = list(origin)

    def enable_highlight(self, enable: bool = True):
        """
        Enable hover highlight.

        Parameters
        ----------
        enable : bool, optional
            Highlight enabled, by default True.
        """
        # VTK plotter has built-in hover, controlled by JS side
        pass

    def get_mouse_location(self) -> Tuple[float, float, float]:
        """
        Return the current mouse location.

        Returns
        -------
        Tuple[float, float, float]
            Mouse location in 3D space.
        """
        return tuple(self.plotter.hover_position)

    def provide_on_mouse_move_callback(self, callback: Callable):
        """
        Store a function to call every time the user moves the mouse on the plot.

        Parameters
        ----------
        callback : Callable
            Function to call with arguments (screen_location, space_location, cell_id).
        """
        self.on_mouse_move_callback = callback

    def provide_on_clic_callback(self, callback: Callable):
        """
        Store a function to call every time the user clicks on the plot.

        Parameters
        ----------
        callback : Callable
            Function to call with arguments (screen_location, space_location, cell_id).
        """
        self.on_clic_callback = callback

    def get_resolution(self) -> Tuple[int, int]:
        """
        Return the current plot resolution.

        Returns
        -------
        Tuple[int, int]
            Resolution (width, height) or (None, None) if not available.
        """
        # VTK plotter doesn't expose resolution directly
        return None, None

    @property
    def n_cells(self) -> int:
        """Returns the number of cells in the current polydata.

        Returns
        -------
        int
            Number of cells, or 0 if no data loaded.
        """
        if self._current_polydata is not None:
            return self._current_polydata.n_cells
        return 0

    def set_plane_enabled(self, enabled: bool):
        """
        Enable or disable the slice plane.

        Parameters
        ----------
        enabled : bool
            Enable the slice plane if True.
        """
        self.plotter.set_plane_enabled(enabled)

    def set_clip_enabled(self, enabled: bool):
        """
        Enable or disable clipping.

        Parameters
        ----------
        enabled : bool
            Enable clipping if True.
        """
        self.plotter.set_clip_enabled(enabled)

    def set_clip_axis(self, axis: str, sign: int = 1):
        """
        Set the clip plane axis.

        Parameters
        ----------
        axis : str
            Axis name ('x', 'y', or 'z').
        sign : int, optional
            Axis direction sign, by default 1.
        """
        self.plotter.set_clip_axis(axis, sign)

    def set_edges_visible(self, visible: bool):
        """
        Set edges visibility.

        Parameters
        ----------
        visible : bool
            Show edges if True.
        """
        self.plotter.set_edges_visible(visible)

    def set_info(self, enabled: bool):
        """
        Enable or disable info display.

        Parameters
        ----------
        enabled : bool
            Enable info display if True.
        """
        self.plotter.set_info(enabled)
