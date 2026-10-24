"""
PanelGraph - graph visualization panel for Scivianna.

This module implements PanelGraph, a visualization panel that renders a
:class:`~scivianna.data.data_graph.DataGraph` (a set of 2D plots connected by
directed links) as an interactive ReactFlow canvas where the nodes can be
dragged, rotated and selected in groups.

Unlike :class:`Panel2D`, the graph panel is not bound to a compute slave: the
graph (node geometries, links and layout) is provided directly with
:meth:`set_graph` or at construction time.
"""

from typing import Callable, List, Optional

import param

import panel as pn
import panel_material_ui as pmui

from scivianna.data.data_graph import DataGraph
from scivianna.extension.extension import Extension
from scivianna.logging_config import get_logger
from scivianna.plotter_graph.generic_plotter import PlotterGraph
from scivianna.plotter_graph.reactflow_plotter import ReactFlowPlotter

logger = get_logger(__name__)


class PanelGraph(pn.viewable.Viewer):
    """Graph visualisation panel associated to a :class:`DataGraph`.

    Attributes
    ----------
    plotter : PlotterGraph
        Graph plotter displaying and updating the graph.
    current_data : DataGraph
        Displayed graph and its properties.
    """

    panel_name: str
    """ Panel name
    """
    plotter: PlotterGraph
    """ Graph plotter displaying and updating the graph
    """
    current_data: DataGraph
    """ Displayed graph and its properties.
    """

    def __init__(
        self,
        name: str = "",
        graph: Optional[DataGraph] = None,
        extensions: List[Extension] = [],
        **plotter_kwargs,
    ):
        """Graph panel constructor

        Parameters
        ----------
        name : str
            Name of the panel.
        graph : DataGraph, optional
            Graph to display. Can be given later with ``set_graph``.
        extensions : List[Extension]
            Reserved for future use (extensions are not wired into the graph
            panel yet).
        **plotter_kwargs
            Forwarded to :class:`ReactFlowPlotter` (e.g. ``max_size``,
            ``scale``, ``color_map_name``).
        """
        super().__init__()

        self.panel_name = name
        self.extensions = list(extensions)
        self.plotter = ReactFlowPlotter(**plotter_kwargs)
        self.current_data: Optional[DataGraph] = None

        # title = pmui.Typography("**" + (name or "Graph") + "**", margin=0)
        self.main_frame = pmui.Column(
            # title,
            self.plotter.make_panel(),
            sizing_mode="stretch_both",
        )

        if graph is not None:
            self.set_graph(graph)

    def set_graph(self, graph: DataGraph):
        """Displays a new graph in the panel.

        Parameters
        ----------
        graph : DataGraph
            Graph to display (node geometries, links and layout).
        """
        if self.current_data is None:
            self.plotter.plot(graph)
        else:
            self.plotter.update_plot(graph)
        self.current_data = graph

    def set_color_map(self, color_map_name: str):
        """Sets the color map used to shade the cells of every node.

        Parameters
        ----------
        color_map_name : str
            Color map name (see ``scivianna.utils.color_tools.color_maps``).
        """
        self.plotter.set_color_map(color_map_name)

    def provide_on_mouse_move_callback(self, callback: Callable):
        """Stores a function to call everytime the user moves the mouse on a node.

        Parameters
        ----------
        callback : Callable
            Function to call with (screen_location, space_location, cell_id).
        """
        self.plotter.provide_on_mouse_move_callback(callback)

    def provide_on_clic_callback(self, callback: Callable):
        """Stores a function to call everytime the user clics on a node.

        Parameters
        ----------
        callback : Callable
            Function to call with (screen_location, space_location, cell_id).
        """
        self.plotter.provide_on_clic_callback(callback)

    def __panel__(self):
        """Returns the panel layout: title + ReactFlow canvas."""
        return self.main_frame
