from typing import Tuple, Callable, Optional

import numpy as np

from scivianna.data.data3d import Data3D

class Plotter3D:
    """Unfinished 3D plotter to get the coupling working"""

    def __init__(
        self,
    ):
        self.on_axes_change_callback = None

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
        raise NotImplementedError()

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
        raise NotImplementedError()

    def make_panel(self,):
        """Returns the viewable displayed in the 3D panel

        Returns
        -------
        pn.viewable.Viewable
            3D potter
        """
        raise NotImplementedError()

    def update_colorbar(self, display: bool, range: Tuple[float, float]):
        """Displays or hide the color bar, if display, updates its range

        Parameters
        ----------
        display : bool
            Display or hides the color bar
        range : Tuple[float, float]
            New colormap range
        """
        raise NotImplementedError()

    def set_color_map(self, color_map_name: str):
        """Sets the colorbar color map name

        Parameters
        ----------
        color_map_name : str
            Color map name
        """
        raise NotImplementedError()

    def provide_on_mouse_move_callback(self, callback: Callable):
        """Stores a function to call everytime the user moves the mouse on the plot.
        Functions arguments are location, cell_id.

        Parameters
        ----------
        callback : Callable
            Function to call.
        """
        raise NotImplementedError()

    def provide_on_clic_callback(self, callback: Callable):
        """Stores a function to call everytime the user clics on the plot.
        Functions arguments are location, cell_id.

        Parameters
        ----------
        callback : Callable
            Function to call.
        """
        raise NotImplementedError()

    def provide_on_axes_change_callback(self, callback: Callable):
        """Stores a function to call everytime the user changes the axes.
        Functions arguments are the new axis values (u, v, origin, size_u, size_v).

        Parameters
        ----------
        callback : Callable
            Function to call.
        """
        raise NotImplementedError()

    def move_slice_to(
        self,
        u: np.ndarray = None,
        v: np.ndarray = None,
        origin: Tuple[float, float, float] = None,
    ):
        """Moves the slice to the given location

        Parameters
        ----------
        u : np.ndarray, optional
            First axis direction vector
        v : np.ndarray, optional
            Second axis direction vector
        origin : Tuple[float, float, float], optional
            Physical 3D position of the slice center (center_u*u + center_v*v + w*w), by default None
        """
        raise NotImplementedError()

    def get_slice_normal(self) -> np.ndarray:
        """Returns the normal of the slice plane

        Returns
        -------
        np.ndarray
            Normal of the slice plane
        """
        raise NotImplementedError()

    def set_slice_origin(self, origin: Tuple[float, float, float]):
        """Sets the origin of the slice plane

        Parameters
        ----------
        origin : Tuple[float, float, float]
            Origin of the slice plane
        """
        raise NotImplementedError()

    def get_slice_origin(self) -> np.ndarray:
        """Returns the origin of the slice plane

        Returns
        -------
        np.ndarray
            Origin of the slice plane
        """
        raise NotImplementedError()

    def get_uv(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the u and v axes of the slice plane

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            u and v axes of the slice plane
        """
        raise NotImplementedError()
