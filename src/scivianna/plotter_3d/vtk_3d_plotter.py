from typing import Tuple, Callable

import numpy as np

from scivianna_vtk.plotter import VTKPlotter
from scivianna.data.data3d import Data3D

class Plotter3D:
    """Unfinished 3D plotter to get the coupling working"""

    def __init__(
        self,
    ):
        self.plotter = VTKPlotter(
            sizing_mode="stretch_both",
            margin=0
        )
        self.plotter.set_clip_enabled = True

    def plot(
        self,
        data: Data3D
    ):
        """Adds a new plot to the figure from a set of polygons

        Parameters
        ----------
        data : Data3D
            Data3D object containing the geometry and data to plot
        """
        print(f"Plotting {data}")
        if data is not None:
            data.update_cell_data()
            self.plotter.update_polydata(data.polydata)

    def update_plot(
        self,
        data: Data3D
    ):
        """Updates plot to the figure

        Parameters
        ----------
        data : Data3D
            Data3D object containing the geometry and data to plot
        """
        if data is not None:
            data.update_cell_data()
            self.plotter.update_colors(data.polydata)

    def make_panel(self,):
        """Returns teh viewable displayed in the 3D panel

        Returns
        -------
        pn.viewable.Viewable
            3D potter
        """
        return self.plotter

    def update_colorbar(self, display: bool, range: Tuple[float, float]):
        """Displays or hide the color bar, if display, updates its range

        Parameters
        ----------
        display : bool
            Display or hides the color bar
        range : Tuple[float, float]
            New colormap range
        """
        pass

    def set_color_map(self, color_map_name: str):
        """Sets the colorbar color map name

        Parameters
        ----------
        color_map_name : str
            Color map name
        """
        pass

    def provide_on_mouse_move_callback(self, callback: Callable):
        """Stores a function to call everytime the user moves the mouse on the plot.
        Functions arguments are location, cell_id.

        Parameters
        ----------
        callback : Callable
            Function to call.
        """
        self.on_mouse_move_callback = callback

        self.plotter.param.watch(
            lambda event: self.send_event(callback),
            "hover_position"
        )

    def provide_on_clic_callback(self, callback: Callable):
        """Stores a function to call everytime the user clics on the plot.
        Functions arguments are location, cell_id.

        Parameters
        ----------
        callback : Callable
            Function to call.
        """
        self.on_clic_callback = callback

    def move_slice_to(
        self,
        u: float,
        v: float,
        u_min: float = None,
        u_max: float = None,
        v_min: float = None,
        v_max: float = None,
        w: float = None
    ):
        """Moves the slice to the given location

        Parameters
        ----------
        u : float
            First axis location
        v : float
            Second axis location
        u_min : float, optional
            First axis min value, by default None
        u_max : float, optional
            First axis max value, by default None
        v_min : float, optional
            Second axis min value, by default None
        v_max : float, optional
            Second axis max value, by default None
        w : float, optional
            Normal axis location, by default None
        """
        w_vector = None
        origin = None

        if u is not None and v is not None:
            w_vector = np.cross(u, v)

        if u_min is not None and v_min is not None and w is not None:
            origin = u_min * u + v_min * v + w * w_vector

        self.plotter.set_clip_plane(origin, w_vector)

    def send_event(self, callback):
        if self.plotter.hover_cell_id is not None:
            callback(
                screen_location=(
                    None,
                    None
                ),
                space_location=self.plotter.hover_position, 
                cell_id=self.plotter.hover_cell_id
            )
