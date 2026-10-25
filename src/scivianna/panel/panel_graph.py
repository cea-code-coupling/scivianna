"""
PanelGraph - graph visualization panel for Scivianna.

This module implements PanelGraph, a visualization panel that renders a
:class:`~scivianna.data.data_graph.DataGraph` (a set of 2D plots connected by
directed links) as an interactive ReactFlow canvas where the nodes can be
dragged, rotated and selected in groups.

Like :class:`Panel2D`, it is a :class:`VisualizationPanel`: the figure sits in an
:class:`Overlay`, extensions can be attached, updates are applied on the next UI
tick, and the panel can be duplicated and saved to / restored from a json dict.
Unlike :class:`Panel2D`, it is not bound to a compute slave (``slave`` is None): the
graph (node geometries, links and layout) is provided directly, at construction
time or with :meth:`set_graph`.
"""

import copy
from typing import Callable, Dict, List, Optional, Tuple, Type

import panel as pn
import param

from scivianna.data.data_graph import DataGraph
from scivianna.enums import UpdateEvent
from scivianna.extension.extension import Extension
from scivianna.logging_config import get_logger
from scivianna.panel.visualisation_panel import VisualizationPanel
from scivianna.plotter_graph.generic_plotter import PlotterGraph
from scivianna.plotter_graph.reactflow_plotter import ReactFlowPlotter
from scivianna.extension.field_selector import FieldSelector
from scivianna.extension.file_loader import FileLoader
from scivianna.extension.ai_assistant import AIAssistant
from scivianna.extension.axes import Axes
from scivianna.slave import ComputeSlave

logger = get_logger(__name__)

pn.config.inline = True

# Base default extensions: a graph has no field, slice or file to select, so none of the
# ComputeSlave based extensions of Panel2D apply.
default_extensions_graph: List[Type[Extension]] = []

PLOTTER_OPTIONS = (
    "max_size",
    "scale",
    "fit_each",
    "pad_px",
    "color_scale",
    "show_colorbar",
    "colorbar_title",
    "colorbar_position",
    "colorbar_orientation",
    "colorbar_bins",
)
"""ReactFlowPlotter constructor options saved by :meth:`PanelGraph.to_json`."""

default_extensions_bokeh = [FileLoader, Axes, AIAssistant]

class PanelGraph(VisualizationPanel):
    """Graph visualisation panel associated to a :class:`DataGraph`."""

    plotter: PlotterGraph
    """ Graph plotter displaying and updating the graph
    """
    current_data: Optional[DataGraph]
    """ Displayed graph and its properties.
    """
    colormap = param.String()

    def __init__(
        self,
        name: str = "",
        graph: Optional[DataGraph] = None,
        extensions: Optional[List[Type[Extension]]] = None,
        extension_jsons: Optional[Dict[Type[Extension], dict]] = None,
        colormap: str = "BuRd",
        **plotter_kwargs,
    ):
        """Graph panel constructor

        Parameters
        ----------
        name : str
            Name of the panel.
        graph : DataGraph, optional
            Graph to display. Can be given later with :meth:`set_graph`.
        extensions : List[Type[Extension]], optional
            Extensions to add to the panel. None: no extension.
        extension_jsons : dict[Type[Extension], dict], optional
            Panel extensions initial options
        colormap : str
            Colormap name for coloring the nodes, defaults to "BuRd". (``color_map_name``
            is accepted as an alias, for backward compatibility.)
        **plotter_kwargs
            Forwarded to :class:`ReactFlowPlotter` (e.g. ``max_size``, ``scale``,
            ``color_scale``, ``show_colorbar``, ``node_padding``, ``rotation_origin``).
        """
        colormap = plotter_kwargs.pop("color_map_name", colormap)

        #
        #   Initializing attributes
        #
        self.field_change_callback: Callable = None
        """Function to call when the field is changed (never triggered: a graph has no field)"""

        self.on_axes_change_callback: Callable = None
        """Function to call when the axes are changed (never triggered: axes are fixed per node)"""

        self._pending_updates: Dict = {}
        """Pending visual updates to apply on next tick"""

        self.current_data = None

        #
        #   Plotter creation
        #
        self.plotter = ReactFlowPlotter(color_map_name=colormap, **plotter_kwargs)
        
        if extensions is None:
            extensions = default_extensions_bokeh

        super().__init__(
            None,
            name,
            list(default_extensions_graph if extensions is None else extensions),
            extension_jsons or {},
        )

        self.colormap = colormap

        #
        #   First plot
        #
        if graph is not None:
            self.plotter.plot(graph)
            self.current_data = graph

        try:
            pn.state.on_session_created(self.recompute)
        except Exception:
            pass

    def __panel__(self):
        """Layout of the panel: the extensions sidebar (if any extension) and the graph."""
        if not self.extensions:
            return pn.Row(self.figure, margin=0, sizing_mode="stretch_both")
        return super().__panel__()

    @property
    def main_frame(self):
        """Layout displaying the extensions sidebar and the graph (kept for backward compatibility)."""
        return self.__panel__()

    #
    # #
    # #     Updates
    # #
    #
    @pn.io.hold()
    def _apply_update(self):
        """Apply pending visual updates to the plotter.
        Called via add_next_tick_callback to ensure UI thread safety.
        """
        pending, self._pending_updates = self._pending_updates, {}

        if pending.get("colormap") is not None:
            self.plotter.set_color_map(pending["colormap"])

        if pending.get("data") is not None:
            graph = pending["data"]
            if self.current_data is None:
                self.plotter.plot(graph)
            else:
                self.plotter.update_plot(graph)
            self.current_data = graph

        self.marked_to_recompute = False

        # Force notebook refresh
        pn.io.push_notebook(self.figure)

    def _schedule_update(self):
        """Schedule a visual update on the next UI tick."""
        self.marked_to_recompute = True
        if pn.state.curdoc is not None:
            pn.state.curdoc.add_next_tick_callback(self._apply_update)
        else:
            # No running session (script, notebook, tests): nothing to be thread safe with.
            self._apply_update()

    def set_graph(self, graph: DataGraph):
        """Displays a new graph in the panel.

        The nodes present in both the displayed graph and the new one are updated in
        place; if nodes or links appear or disappear, the canvas is rebuilt.

        Parameters
        ----------
        graph : DataGraph
            Graph to display (node geometries, links and layout).
        """
        self._pending_updates["data"] = graph
        self._schedule_update()

    def recompute(self, *args, **kwargs):
        """Re-applies the current graph to the plotter (e.g. when a session is created)."""
        if self.current_data is None:
            return
        self._pending_updates.setdefault("data", self.current_data)
        self._schedule_update()

    def set_colormap(self, colormap: str):
        """Sets the current color map

        Parameters
        ----------
        colormap : str
            Color map name
        """
        if colormap != self.colormap:
            self.colormap = colormap
            self._pending_updates["colormap"] = colormap
            self._schedule_update()

    #
    # #
    # #     API shared with the other panels
    # #
    #
    def duplicate(self, keep_name: bool = False) -> "PanelGraph":
        """Get a copy of the panel. The new panel displays a copy of the current graph, with
        the same plotter options and extensions.

        Parameters
        ----------
        keep_name : bool
            New panel name is the same as the current, if not, a number iterates at the end of the name

        Returns
        -------
        PanelGraph
            Copy of the graph panel
        """
        new_panel = PanelGraph(
            name=self.panel_name if keep_name else self.get_new_name(),
            graph=copy.deepcopy(self.current_data) if self.current_data is not None else None,
            extensions=[e for e in self.extension_classes],
            colormap=self.colormap,
            **self._plotter_kwargs(),
        )
        new_panel.copy_index = self.copy_index

        if isinstance(self.update_event, list):
            new_panel.update_event = self.update_event.copy()
        else:
            new_panel.update_event = self.update_event
        new_panel.sync_field = self.sync_field

        return new_panel

    def get_slave(self) -> Optional[ComputeSlave]:
        """Returns the current panel code slave: a graph panel has none.

        Returns
        -------
        Optional[ComputeSlave]
            None
        """
        return self.slave

    def provide_on_axes_change_callback(self, callback: Callable):
        """Stores a function to call everytime the axes are changed.
        Axes are fixed per node in a graph plot, so it is never triggered; stored for
        interface compatibility.

        Parameters
        ----------
        callback : Callable
            Function to call.
        """
        self.on_axes_change_callback = callback
        self.plotter.provide_on_axes_change_callback(callback)

    def recompute_at(self, position: Tuple[float, float, float], cell_id: str):
        """A graph has no slice location: nothing to recompute. Kept for interface compatibility.

        Parameters
        ----------
        position : Tuple[float, float, float]
            Ignored
        cell_id : str
            Ignored
        """
        pass

    def set_field(self, field_name: str):
        """A graph has no field to select. Kept for interface compatibility.

        Parameters
        ----------
        field_name : str
            Ignored
        """
        logger.debug("%s - A graph panel has no field to select (requested %s).", self.panel_name, field_name)

    def set_coordinates(self, *args, **kwargs):
        """Axes are fixed per node in a graph plot. Kept for interface compatibility."""
        pass

    #
    # #
    # #     Save / restore
    # #
    #
    def _plotter_kwargs(self) -> Dict:
        """Current options of the plotter, in the form accepted by the constructor."""
        kwargs = {key: getattr(self.plotter, key) for key in PLOTTER_OPTIONS}
        canvas = self.plotter.canvas
        kwargs.update(canvas.options)
        kwargs["node_padding"] = canvas.node_padding
        return kwargs

    def to_json(self) -> Dict:
        """Returns a dictionnary with the information about the visualization panel

        The displayed graph is not included, it is given back to :meth:`from_json`.

        Returns
        -------
        Dict
            Information dict
        """
        return {
            "name": self.panel_name,
            "colormap": self.colormap,
            "plotter_kwargs": self._plotter_kwargs(),
            "sync_field": self.sync_field,
            "update_event": self.update_event,
        }

    @classmethod
    def from_json(
        cls,
        info_dict: Dict,
        slave: Optional[ComputeSlave] = None,
        data: Optional[DataGraph] = None,
        extensions: Optional[List[Type[Extension]]] = None,
        extension_jsons: Optional[Dict[Type[Extension], dict]] = None,
    ) -> "PanelGraph":
        """Restores the visualization panel from its information dict

        Parameters
        ----------
        info_dict : Dict
            Dictionnary containing all required information to restore the panel
        slave : ComputeSlave, optional
            Ignored, a graph panel has no slave (kept for the signature shared with Panel2D)
        data : DataGraph, optional
            Initial state graph
        extensions : List[Type[Extension]], optional
            Extensions to add to the panel
        extension_jsons : dict[Type[Extension], dict], optional
            Panel extensions initial options

        Returns
        -------
        PanelGraph
            Restored panel
        """
        panel = PanelGraph(
            name=info_dict["name"],
            graph=data,
            extensions=extensions,
            extension_jsons=extension_jsons,
            colormap=info_dict.get("colormap", "BuRd"),
            **info_dict.get("plotter_kwargs", {}),
        )
        panel.sync_field = info_dict.get("sync_field", False)
        panel.update_event = info_dict.get("update_event", UpdateEvent.RECOMPUTE)
        return panel