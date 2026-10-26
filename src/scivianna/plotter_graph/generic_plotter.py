from typing import Callable, Optional, Tuple

import numpy as np

from scivianna.data.data_graph import DataGraph


class PlotterGraph:
    """Abstract base class for plotting a :class:`DataGraph`.

    A graph plotter renders a set of 2D plots (nodes) connected by directed
    links, typically used to visualize the topology of coupled simulations.
    Concrete implementations provide the rendering backend and implement the
    methods below; they are consumed by ``PanelGraph`` in the same way that
    :class:`Plotter2D`/ :class:`Plotter3D` backends are consumed by their
    panels.

    Attributes
    ----------
    on_axes_change_callback : Callable, optional
        Callback invoked when the user changes a node's axes (u, v, origin,
        size_u, size_v), registered via
        :meth:`provide_on_axes_change_callback`.
    """

    def __init__(
        self,
    ):
        self.on_axes_change_callback = None

    def plot(self, data: DataGraph):
        """Adds a new graph to the figure from a set of nodes and links

        Parameters
        ----------
        data : DataGraph
            DataGraph object containing the node geometries and links to plot
        """
        raise NotImplementedError()

    def update_plot(self, data: DataGraph):
        """Updates the displayed graph in place

        Parameters
        ----------
        data : DataGraph
            DataGraph object with updated node geometries and links
        """
        raise NotImplementedError()

    def make_panel(
        self,
    ):
        """Returns the viewable displayed in the graph panel

        Returns
        -------
        pn.viewable.Viewable
            Graph plotter widget
        """
        raise NotImplementedError()

    def update_colorbar(self, display: bool, value_range: Tuple[float, float]):
        """Displays or hide the color bar, if display, updates its range

        Parameters
        ----------
        display : bool
            Display or hides the color bar
        value_range : Tuple[float, float]
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

    def provide_on_node_selection_change_callback(self, callback: Callable):
        """Stores a function to call everytime the selection of nodes changes.
        Functions arguments are the ids of the selected nodes (empty list when
        nothing is selected).

        Parameters
        ----------
        callback : Callable
            Function to call.
        """
        raise NotImplementedError()

    def provide_on_nodes_moved_callback(self, callback: Callable):
        """Stores a function to call everytime one or several nodes are moved on the canvas.
        Functions arguments are a list of {"id", "x", "y"} dicts, one per moved node.

        Parameters
        ----------
        callback : Callable
            Function to call.
        """
        raise NotImplementedError()
