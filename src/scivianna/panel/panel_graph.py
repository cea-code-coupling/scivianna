"""
PanelGraph - graph visualization panel for Scivianna.

This module implements PanelGraph, a visualization panel that renders a
:class:`~scivianna.data.data_graph.DataGraph` (a set of 2D plots connected by
directed links) as an interactive ReactFlow canvas where the nodes can be
dragged, rotated and selected in groups.

Like :class:`Panel2D`, it is a :class:`VisualizationPanel`: the figure sits in an
:class:`Overlay`, extensions can be attached, updates are applied on the next UI
tick, and the panel can be duplicated and saved to / restored from a json dict.
The graph (node geometries, links and layout) is requested to a
:class:`~scivianna.slave.ComputeSlave` whose code interface implements
:class:`~scivianna.interface.generic_interface.GraphInterface`, in the same way
:class:`Panel2D` uses a :class:`Geometry2D` interface.
"""

import copy
from typing import Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import panel as pn
import param

from scivianna.data.data_graph import DataGraph
from scivianna.enums import UpdateEvent, VisualizationMode
from scivianna.extension.extension import Extension
from scivianna.interface.generic_interface import GraphInterface
from scivianna.logging_config import get_logger
from scivianna.panel.visualisation_panel import VisualizationPanel
from scivianna.plotter_graph.generic_plotter import PlotterGraph
from scivianna.plotter_graph.reactflow_plotter import ReactFlowPlotter
from scivianna.extension.field_selector import FieldSelector
from scivianna.extension.file_loader import FileLoader
from scivianna.extension.ai_assistant import AIAssistant
from scivianna.extension.graph import Graph
from scivianna.slave import ComputeSlave

logger = get_logger(__name__)

pn.config.inline = True

# Base default extensions: a graph has no slice to select, so the Slice3D
# extension of Panel2D does not apply.
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

# The graph panel has no 2D slice: the Axes extension (built for a Plotter2D)
# does not apply.
default_extensions_bokeh = [FileLoader, FieldSelector, Graph, AIAssistant]


class PanelGraph(VisualizationPanel):
    """Graph visualisation panel associated to a code."""

    plotter: PlotterGraph
    """ Graph plotter displaying and updating the graph
    """
    current_data: Optional[DataGraph]
    """ Displayed graph and its properties.
    """
    colormap = param.String()

    def __init__(
        self,
        slave: ComputeSlave,
        name: str = "",
        extensions: Optional[List[Type[Extension]]] = None,
        extension_jsons: Optional[Dict[Type[Extension], dict]] = None,
        data: Optional[DataGraph] = None,
        displayed_field: Optional[str] = None,
        colormap: str = "BuRd",
        **plotter_kwargs,
    ):
        """Graph panel constructor

        Parameters
        ----------
        slave : ComputeSlave
            ComputeSlave object to which request the plots. Its code interface
            must implement :class:`GraphInterface`.
        name : str
            Name of the panel.
        extensions : List[Type[Extension]], optional
            Extensions to add to the panel. None: no extension.
        extension_jsons : dict[Type[Extension], dict], optional
            Panel extensions initial options
        data : DataGraph, optional
            Graph with which the panel is initialized. If not given, it is
            computed from the slave.
        displayed_field : str, optional
            Name of the field to display initially. None: the first label of
            the interface.
        colormap : str
            Colormap name for coloring the nodes, defaults to "BuRd". (``color_map_name``
            is accepted as an alias, for backward compatibility.)
        **plotter_kwargs
            Forwarded to :class:`ReactFlowPlotter` (e.g. ``max_size``, ``scale``,
            ``color_scale``, ``show_colorbar``, ``node_padding``, ``rotation_origin``).
        """
        code_interface: Type[GraphInterface] = slave.code_interface
        assert issubclass(
            code_interface, GraphInterface
        ), f"A PanelGraph can only be given a GraphInterface slave, received {code_interface}."

        colormap = plotter_kwargs.pop("color_map_name", colormap)

        #
        #   Initializing attributes
        #
        self.field_change_callback: Callable = None
        """Function to call when the field is changed"""

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
            slave,
            name,
            list(default_extensions_graph if extensions is None else extensions),
            extension_jsons or {},
        )

        #
        #   First plot
        #
        if displayed_field is None:
            labels = self.slave.get_labels()
            displayed_field = labels[0] if labels else ""
        self.displayed_field = displayed_field
        for extension in self.extensions:
            extension.on_field_change(displayed_field)

        self.colormap = colormap

        if data is None:
            data_ = self.compute_fn()
        else:
            logger.info(
                "PanelGraph %s initialized with DataGraph object, restoring %d nodes",
                self.panel_name,
                len(data.datas),
            )
            data_ = data

        self.plotter.plot(data_)
        self.current_data = data_

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

    def _do_recompute(self):
        """Perform the actual recompute and schedule visual update."""
        if not self.marked_to_recompute:
            return

        logger.info(
            "%s - Recomputing graph, field: %s",
            self.panel_name,
            self.displayed_field,
        )

        data = self.compute_fn()

        if data is not None:
            # Build pending updates directly
            self._pending_updates = {"data": data}

            # Schedule visual update
            self._schedule_update()

    def compute_fn(self) -> DataGraph:
        """Request the slave to compute a new graph, and updates the data to display

        Returns
        -------
        DataGraph
            Graph data.
        """
        options = {}
        for extension in self.extensions:
            ext_options = extension.provide_options()
            options.update(ext_options)

        if self.panel_coupling_extension is not None:
            coupling_options = self.panel_coupling_extension.provide_options()
            for key, value in coupling_options.items():
                options[key] = value

        computed_data = self.slave.compute_graph(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
            None,
            self.displayed_field,
            options,
            caller=self.panel_name,
        )

        if computed_data is None:
            logger.error(
                "Got None from computed data on %s, returning past values", self.panel_name
            )
            return None

        computed_data, graph_updated = computed_data

        # The extensions color the data on their way to the plotter. A DataGraph
        # holds one Data2D per node: each node is passed through the hook (the
        # ReactFlowPlotter colors the nodes itself from the global value range).
        for data in computed_data.datas.values():
            for extension in self.extensions:
                extension.on_updated_data(data)

        # The hooks may have left cell_values as a plain list: the plotters expect arrays.
        for data in computed_data.datas.values():
            if not isinstance(data.cell_values, np.ndarray):
                data.cell_values = np.array(data.cell_values)

        return computed_data

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
        """Recomputes the figure based on the new parameters.
        Public method that triggers the recompute pipeline.
        """
        if self.current_data is None:
            return
        self.marked_to_recompute = True
        if pn.state.curdoc is not None:
            pn.state.curdoc.add_next_tick_callback(self._do_recompute)
        else:
            self._do_recompute()

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
            slave=self.slave.duplicate(),
            name=self.panel_name if keep_name else self.get_new_name(),
            data=copy.deepcopy(self.current_data) if self.current_data is not None else None,
            extensions=[e for e in self.extension_classes],
            displayed_field=self.displayed_field,
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

    def get_slave(self) -> ComputeSlave:
        """Returns the current panel code slave.

        Returns
        -------
        ComputeSlave
            Panel slave
        """
        return self.slave

    def provide_on_mouse_move_callback(self, callback: Callable):
        """Stores a function to call everytime the user moves the mouse on a node.
        Functions arguments are location, cell_id.

        Parameters
        ----------
        callback : Callable
            Function to call.
        """
        self.plotter.provide_on_mouse_move_callback(callback)

    def provide_on_clic_callback(self, callback: Callable):
        """Stores a function to call everytime the user clics on a node.
        Functions arguments are location, cell_id.

        Parameters
        ----------
        callback : Callable
            Function to call.
        """
        self.plotter.provide_on_clic_callback(callback)

    def provide_field_change_callback(self, callback: Callable):
        """Stores a function to call everytime the displayed field is changed.
        The functions takes a string as argument.

        Parameters
        ----------
        callback : Callable
            Function to call.
        """
        self.field_change_callback = callback

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
        """Updates the plotted field

        Parameters
        ----------
        field_name : str
            New field to display
        """
        if self.displayed_field != field_name:
            self.displayed_field = field_name

            if field_name not in self.slave.get_labels():
                logger.warning(
                    f"\n\nRequested field {field_name} : field unavailable, available values : {self.slave.get_labels()}.\n\n"
                )
            else:
                for extension in self.extensions:
                    extension.on_field_change(field_name)

                if pn.state.curdoc is not None:
                    pn.state.curdoc.add_next_tick_callback(self.recompute)
                else:
                    self.recompute()

                if self.field_change_callback is not None:
                    self.field_change_callback(field_name)

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
            "displayed_field": self.displayed_field,
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
        slave : ComputeSlave
            Panel associated slave (its code interface must implement GraphInterface)
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
            slave=slave,
            name=info_dict["name"],
            data=data,
            extensions=extensions,
            extension_jsons=extension_jsons,
            displayed_field=info_dict.get("displayed_field"),
            colormap=info_dict.get("colormap", "BuRd"),
            **info_dict.get("plotter_kwargs", {}),
        )
        panel.sync_field = info_dict.get("sync_field", False)
        panel.update_event = info_dict.get("update_event", UpdateEvent.RECOMPUTE)
        return panel
