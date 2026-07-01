from typing import Tuple, Callable

from scivianna.component.vtk_plotter import VTKPlotter
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
            print(data.polydata)
            print(len(data.cell_ids))
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

    def provide_on_clic_callback(self, callback: Callable):
        """Stores a function to call everytime the user clics on the plot.
        Functions arguments are location, cell_id.

        Parameters
        ----------
        callback : Callable
            Function to call.
        """
        self.on_clic_callback = callback
