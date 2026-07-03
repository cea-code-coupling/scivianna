from typing import IO, Any, Dict, List, Tuple, Union
from scivianna.data.data2d import Data2D
from scivianna.plotter_2d.generic_plotter import Plotter2D
from scivianna.plotter_2d.grid.grid_tools import get_grids

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors as plt_colors
from scivianna.utils.color_tools import beautiful_color_maps

import numpy as np
import panel as pn


class Matplotlib2DGridPlotter(Plotter2D):
    """2D geometry plotter based on the bokeh python module"""

    display_edges = False
    """Display grid cells edges"""

    def __init__(
        self,
    ):
        """Creates the bokeh Figure and ColumnDataSources"""
        self.figure = plt.figure()
        self.ax = plt.axes()

        # self.colorbar = self.figure.colorbar(None)

        plt.gca().set_aspect("equal")

        self.colormap_name = "BuRd"
        self.display_colorbar = False
        self.colorbar_range = (0.0, 1.0)

    def display_borders(self, display: bool):
        """Display or hides the figure borders and axis

        Parameters
        ----------
        display : bool
            Display if true, hides otherwise
        """
        if display:
            plt.axis("on")  # Hide the axis
        else:
            plt.axis("off")  # Hide the axis

    def update_colorbar(self, display: bool, value_range: Tuple[float, float]):
        """Displays or hide the color bar, if display, updates its range

        Parameters
        ----------
        display : bool
            Display or hides the color bar
        value_range : Tuple[float, float]
            New colormap range
        """
        self.display_colorbar = display
        self.colorbar_range = value_range

    def set_color_map(self, color_map_name: str):
        """Sets the colorbar color map name

        Parameters
        ----------
        color_map_name : str
            Color map name
        """
        self.colormap_name = color_map_name

    def plot_2d_frame(
        self,
        data: Data2D,
    ):
        """Adds a new plot to the figure from a set of polygons

        Parameters
        ----------
        data : Data2D
            Data2D object containing the geometry to plot
        """
        self.plot_2d_frame_in_axes(data, self.ax, {})

    def plot_2d_frame_in_axes(
        self,
        data: Data2D,
        axes: matplotlib.axes.Axes,
        plot_options: Dict[str, Any] = {},
    ):
        """Adds a new plot to the figure from a set of polygons

        Parameters
        ----------
        data : Data2D
            Data2D object containing the geometry to plot
        axes : matplotlib.axes.Axes
            Axes in which plot the figure
        plot_options : Dict[str, Any])
            Color options to be passed on to the actual plot function, such as edgecolor, facecolor, linewidth, markersize, alpha.
        """
        x_values = data.u_values
        y_values = data.v_values

        img, view, grid, val_grid = get_grids(data, self.display_edges)

        axes.pcolormesh(x_values, y_values, view)
        
        if self.display_colorbar:
            plt.colorbar(
                cm.ScalarMappable(
                    norm=plt_colors.Normalize(
                        self.colorbar_range[0], self.colorbar_range[1]
                    ),
                    cmap=LinearSegmentedColormap.from_list(self.colormap_name, (np.array(beautiful_color_maps[self.colormap_name])/255).tolist(), N=len(beautiful_color_maps[self.colormap_name])),
                ),
                ax=axes,
            )


    def update_2d_frame(
        self,
        data: Data2D,
    ):
        """Updates plot to the figure

        Parameters
        ----------
        data : Data2D
            Data2D object containing the data to update
        """
        self.plot_2d_frame(
            data,
        )

    def update_colors(self, data: Data2D,):
        """Updates the colors of the displayed polygons

        Parameters
        ----------
        data : Data2D
            Data2D object containing the data to update
        """
        self.plot_2d_frame(
            data,
        )

    def _set_callback_on_range_update(self, callback: IO):
        """Sets a callback to update the x and y ranges in the GUI.

        Parameters
        ----------
        callback : IO
            Function that takes x0, x1, y0, y1 as arguments
        """
        raise NotImplementedError()

    def make_panel(self) -> pn.viewable.Viewable:
        """Makes the Holoviz panel viewable displayed in the web app.

        Returns
        -------
        pn.viewable.Viewable
            Displayed viewable
        """
        raise NotImplementedError()

    def _disable_interactions(self, disable: bool):
        """Disables de plot interactions for multi panel web-app resizing

        Parameters
        ----------
        disable : bool
            Disable if True, enable if False
        """
        raise NotImplementedError()

    def get_resolution(self) -> Tuple[float, float]:
        """Returns the current plot resolution to display. For resolution based codes, it will be replaced by the value present in the gui

        Returns
        -------
        Tuple[float, float]
            Resolution if possible, else (None, None)
        """
        return None, None

    def export(self, file_name: str, title="Bokeh 2D plot"):
        """Exports the plot in a file

        Parameters
        ----------
        file_name : str
            Export file path
        """
        self.figure.suptitle(title)
        self.figure.tight_layout()
        self.figure.savefig(file_name, dpi=1500)

    def set_axes(self, u:Tuple[float, float, float], v:Tuple[float, float, float], w:float):
        """Stores the u v axes of the current plot

        Parameters
        ----------
        u : Tuple[float, float, float]
            Horizontal axis direction vector
        v : Tuple[float, float, float]
            Vertical axis direction vector
        w : float
            Normal vector coordinate
        """
        pass