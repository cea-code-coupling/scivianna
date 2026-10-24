"""ReactFlow backend for :class:`PlotterGraph`.

``ReactFlowPlotter`` renders a :class:`~scivianna.data.data_graph.DataGraph` as
an interactive ReactFlow canvas built by the ``scivianna-graph`` module: every
node is a Bokeh figure filled with scivianna's :class:`Bokeh2DPolygonPlotter`,
hover-only, fitted to its content with an equal aspect ratio.

Extra features provided by ``scivianna-graph`` (see its README):

* nodes can be plugged on any side (link directions top/bottom/left/right);
* each node can be rotated (``DataGraph.rotations``), figure included, about an
  optional fixed pivot (``rotation_origin`` flow option);
* dragging an empty area selects several nodes at once so they can be moved
  together.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import panel as pn

from scivianna.data.data2d import Data2D
from scivianna.data.data_graph import DataGraph
from scivianna.plotter_2d.polygon.bokeh import Bokeh2DPolygonPlotter
from scivianna.plotter_graph.generic_plotter import PlotterGraph
from scivianna.utils.color_tools import interpolate_cmap_at_values


try:
    from scivianna_graph import GraphEdge, GraphModel, GraphNode, SciviannaGraph
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "The ReactFlow graph plotter requires the 'scivianna-graph' package. "
        "Install it with `pip install -e path/to/scivianna-graph`."
    ) from exc

from scivianna.logging_config import get_logger
logger = get_logger(__name__)


class ReactFlowPlotter(PlotterGraph):
    """Displays a :class:`DataGraph` as a ReactFlow canvas of linked figures.

    Parameters
    ----------
    max_size : int, default 320
        Pixel size of the longest side of the largest figure content. With the
        default shared scale (see ``scale``), every other figure is drawn at the
        same pixels-per-unit, so a 5x5 polygon is smaller than a 1x50 one.
    scale : float, optional
        Pixels per data unit, shared by all figures. If None (default), the scale
        is chosen so that the longest content side among all nodes is ``max_size``
        pixels: relative sizes are kept when the Data2D are in the same unit.
    fit_each : bool, default False
        If True and ``scale`` is None, each figure is instead fitted to
        ``max_size`` on its own (a 5x5 and a 1x50 polygon then share the same
        longest side). Relative sizes are lost.
    pad_px : int, default 2
        Pixel margin around the content (on each side, whatever the content size)
        so polygon edge lines are not clipped.
    color_map_name : str, default "BuRd"
        Color map used to shade the cells of every node (see
        ``scivianna.utils.color_tools.color_maps``). All nodes share one global
        value range so their colors are comparable.
    **flow_kwargs
        Forwarded to ``scivianna_graph.SciviannaGraph`` as display options, e.g.
        ``show_minimap=True``, ``rotation_origin=(0.5, 1.0)``.
    """

    def __init__(
        self,
        max_size: int = 320,
        scale: Optional[float] = None,
        pad_px: int = 2,
        color_map_name: str = "BuRd",
        fit_each: bool = False,
        **flow_kwargs,
    ):
        super().__init__()
        self.max_size = max_size
        self.scale = scale
        self.pad_px = pad_px
        self.fit_each = fit_each
        self.color_map_name = color_map_name
        self._scale: float = 1.0
        """Pixels per data unit used for the current graph (see ``scale``)."""

        self.on_mouse_move_callback = None
        self.on_clic_callback = None

        self.graph: Optional[DataGraph] = None
        self.plotters: Dict[str, Bokeh2DPolygonPlotter] = {}
        """Scivianna plotter of each node, by node key."""
        self.panes: Dict[str, pn.pane.Bokeh] = {}
        """Panel pane wrapping each node figure, by node key."""

        # The scivianna-graph frontend sizes each node to its content and applies
        # rotation itself, so no extra node CSS is needed here. flow_kwargs are
        # display options (min_zoom, max_zoom, show_minimap, ...).
        self.graph_builder = SciviannaGraph(**flow_kwargs)
        self.canvas = self.graph_builder.canvas

    # ------------------------------------------------------------------ PlotterGraph API
    def plot(self, data: DataGraph):
        """Builds the figures, the nodes and the edges from a DataGraph.

        Parameters
        ----------
        data : DataGraph
            DataGraph object containing the node geometries and links to plot
        """
        self.graph = data
        self._scale = self._graph_scale(data)
        model = GraphModel()
        for key in data.datas:
            node = self._make_node(key, data)
            model.add_node(node)
        for i, link in enumerate(data.links):
            model.add_edge(self._make_edge(link))
        self.graph_builder.set_model(model)

    def update_plot(self, data: DataGraph):
        """Updates the displayed graph in place.

        Nodes present in both the old and the new graph are updated in place
        (colors, then full frame if the geometry changed); nodes that appear
        or disappear rebuild the whole canvas.

        Parameters
        ----------
        data : DataGraph
            DataGraph object with updated node geometries and links
        """
        if self.graph is None:
            self.plot(data)
            return

        old_keys = set(self.graph.datas.keys())
        new_keys = set(data.datas.keys())
        if old_keys != new_keys or self._links_signature(self.graph.links) != self._links_signature(data.links):
            # Structure changed: rebuild the whole canvas.
            self.plot(data)
            return

        # The shared scale depends on every node: when one geometry changes it can
        # change the size of all the others.
        old_scale = self._scale
        self._scale = self._graph_scale(data)
        scale_changed = not np.isclose(old_scale, self._scale)

        for key in new_keys:
            plotter = self.plotters[key]
            old_data = self.graph.datas[key]
            new_data = data.datas[key]
            if self._same_geometry(old_data, new_data):
                plotter.update_colors(new_data)
                if not scale_changed:
                    continue
            else:
                plotter.update_2d_frame(new_data)
            self._fit_figure(plotter, new_data)
            pane = self.panes[key]
            pane.param.update(width=int(plotter.figure.width), height=int(plotter.figure.height))

        # Nodes are sized from their figures: tell the canvas about the new sizes.
        self.graph_builder.update_view_sizes()

        self.graph = data

    def make_panel(self):
        """Returns the ReactFlow viewable."""
        return self.canvas

    def update_colorbar(self, display: bool, value_range: Tuple[float, float]):
        """Colorbars are not displayed in graph nodes (they would break the
        frame == content fit). This method is a no-op kept for interface
        compatibility.
        """
        pass

    def set_color_map(self, color_map_name: str):
        """Re-shades every node with a new color map.

        Parameters
        ----------
        color_map_name : str
            Color map name (see ``scivianna.utils.color_tools.color_maps``).
        """
        self.color_map_name = color_map_name
        if self.graph is None:
            return
        for key, data in self.graph.datas.items():
            plotter = self.plotters[key]
            self._apply_colors(data)
            plotter.update_colors(data)

    def provide_on_mouse_move_callback(self, callback: Callable):
        """Stores a function to call everytime the user moves the mouse on a node.
        Functions arguments are location, cell_id (see ``Bokeh2DPolygonPlotter``).

        Parameters
        ----------
        callback : Callable
            Function to call.
        """
        self.on_mouse_move_callback = callback
        if self.graph is not None:
            for plotter in self.plotters.values():
                plotter.provide_on_mouse_move_callback(callback)

    def provide_on_clic_callback(self, callback: Callable):
        """Stores a function to call everytime the user clics on a node.
        Functions arguments are location, cell_id (see ``Bokeh2DPolygonPlotter``).

        Parameters
        ----------
        callback : Callable
            Function to call.
        """
        self.on_clic_callback = callback
        if self.graph is not None:
            for plotter in self.plotters.values():
                plotter.provide_on_clic_callback(callback)

    def provide_on_axes_change_callback(self, callback: Callable):
        """Axes are fixed per node in a graph plot (each figure is fitted to
        its content and tools are disabled). This method stores the callback
        for interface compatibility but it is never triggered.

        Parameters
        ----------
        callback : Callable
            Function to call.
        """
        self.on_axes_change_callback = callback

    # Per-node slice access (the graph plotter keeps each node fitted to its
    # own content, so these are no-ops kept for interface compatibility).
    def move_slice_to(self, plot_name: str, u=None, v=None, origin=None):
        pass

    def get_slice_normal(self, plot_name: str) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0])

    def set_slice_origin(self, plot_name: str, origin: Tuple[float, float, float]):
        pass

    def get_slice_origin(self, plot_name: str) -> np.ndarray:
        return np.zeros(3)

    def get_uv(self, plot_name: str) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])

    # ------------------------------------------------------------------ figure building
    def _make_node(self, key: str, data_graph: DataGraph) -> GraphNode:
        """Builds the bokeh figure of a node and wraps it in a GraphNode."""
        data = data_graph.datas[key]
        plotter = self._make_plotter(data)
        pane = pn.pane.Bokeh(
            plotter.figure,
            sizing_mode="fixed",
            width=int(plotter.figure.width),
            height=int(plotter.figure.height),
            margin=0,
        )
        self.plotters[key] = plotter
        self.panes[key] = pane

        if self.on_mouse_move_callback is not None:
            plotter.provide_on_mouse_move_callback(self.on_mouse_move_callback)
        if self.on_clic_callback is not None:
            plotter.provide_on_clic_callback(self.on_clic_callback)

        return GraphNode(
            id=key,
            position=tuple(data_graph.locations[key]),
            label=data_graph.labels.get(key, key),
            view=pane,
            rotation=float(data_graph.rotations.get(key, 0.0)),
        )

    def _make_edge(self, link) -> GraphEdge:
        """Builds a GraphEdge from a DataGraph Link."""
        return GraphEdge(
            source=link.source_data,
            target=link.target_data,
            source_direction=link.source_direction,
            target_direction=link.target_direction,
            label=link.label,
            marker_end={"type": "arrowclosed"},
        )

    def _make_plotter(self, data: Data2D) -> Bokeh2DPolygonPlotter:
        """Fills a scivianna Bokeh plotter with the data, then makes it hover-only and fitted."""
        plotter = Bokeh2DPolygonPlotter()
        self._apply_colors(data)
        plotter.plot_2d_frame(data)

        fig = plotter.figure

        # Hover only: drop pan / zoom / save / reset ... that bokeh adds by default.
        hover = plotter.hover_tool
        fig.toolbar.tools = [hover]
        fig.toolbar.active_drag = None
        fig.toolbar.active_scroll = None
        fig.toolbar.active_tap = None
        fig.toolbar.active_inspect = hover
        fig.toolbar_location = None

        # The colorbar would add a side panel and break the "frame == content" fit.
        plotter.update_colorbar(False, (None, None))
        fig.right = [r for r in fig.right if r is not plotter.figure_color_bar]

        # No border / outline / axis padding: the outer size of the figure is the
        # frame size, so the content fills it exactly (a small data padding added
        # in _fit_figure keeps polygon edge lines from being clipped).
        fig.min_border = 0
        fig.outline_line_color = None
        fig.xaxis.visible = False
        fig.yaxis.visible = False

        # Fixed ranges: no tool can change them. ``_fit_figure`` sets them to the
        # data bounds and disables match_aspect so the frame is not re-sized.
        from bokeh.models import Range1d
        fig.x_range = Range1d(0, 1, bounds="auto")
        fig.y_range = Range1d(0, 1, bounds="auto")

        self._fit_figure(plotter, data)
        return plotter

    def _apply_colors(self, data: Data2D) -> None:
        """Shades the cells of ``data`` with the current color map (in place)."""
        min_value = np.nanmin([
            np.asarray(d.cell_values, dtype=float).min() for d in self.graph.datas.values()
        ])
        max_value = np.nanmax([
            np.asarray(d.cell_values, dtype=float).max() for d in self.graph.datas.values()
        ])

        values = np.asarray(data.cell_values, dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            data.cell_colors = np.zeros((len(values), 4)) + 255
            return
        vmin, vmax = min_value, max_value
        if vmax - vmin < 1e-12:
            normalized = np.where(np.isfinite(values), 0.5, np.nan)
        else:
            normalized = (values - vmin) / (vmax - vmin)
        data.cell_colors = interpolate_cmap_at_values(self.color_map_name, normalized).astype(np.int16)

    @staticmethod
    def _data_bounds(data: Data2D) -> Tuple[float, float, float, float]:
        """Returns (x_min, x_max, y_min, y_max) of the polygon exteriors (holes are inside)."""
        polygons = data.get_polygons()
        if len(polygons) == 0:
            return 0.0, 1.0, 0.0, 1.0
        xs = np.concatenate([np.asarray(p.exterior_polygon.x_coords, dtype=float) for p in polygons])
        ys = np.concatenate([np.asarray(p.exterior_polygon.y_coords, dtype=float) for p in polygons])
        return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())

    def _fit_figure(self, plotter: Bokeh2DPolygonPlotter, data: Data2D) -> None:
        """Sets the figure ranges and pixel size so that the frame is exactly the content.

        The ranges are the data bounds plus ``pad_px`` pixels on each side, so the
        plotted polygons fill the frame (edge lines included). A single
        pixels-per-unit scale is used for both axes, keeping the data's aspect
        ratio. ``match_aspect`` is disabled: with the frame already matching the
        content it would only re-size the frame on every render (the source of
        figures appearing larger than their content and overlapping the node
        border).
        """
        x0, x1, y0, y1 = self._data_bounds(data)
        dx = max(x1 - x0, 1e-12)
        dy = max(y1 - y0, 1e-12)

        scale = self._figure_scale(dx, dy)

        # ``pad_px`` pixels of padding on each side, converted to data units. (It is
        # a fixed number of pixels: it must not grow with the extent of the content,
        # or large/thin polygons end up with big blank bands.)
        pad_x = pad_y = self.pad_px / max(scale, 1e-9)

        width = max(int(round((dx + 2 * pad_x) * scale)), 1)
        height = max(int(round((dy + 2 * pad_y) * scale)), 1)

        fig = plotter.figure
        # Ranges are the data bounds plus a hair of padding: the content fills
        # the frame (edge lines included) instead of being cropped at the border.
        fig.x_range.update(start=x0 - pad_x, end=x1 + pad_x)
        fig.y_range.update(start=y0 - pad_y, end=y1 + pad_y)

        # The frame already matches the content, so let bokeh size it verbatim
        # instead of recomputing an aspect-fit (which would grow the frame).
        fig.match_aspect = False
        fig.sizing_mode = "fixed"
        fig.frame_width = width
        fig.frame_height = height
        fig.width = width
        fig.height = height

    # ------------------------------------------------------------------ helpers
    def _graph_scale(self, graph: DataGraph) -> float:
        """Pixels per data unit shared by every figure of the graph.

        The given ``scale`` if any, else the one for which the longest content side
        over all nodes is ``max_size`` pixels.
        """
        if self.scale is not None:
            return float(self.scale)
        extents = []
        for data in graph.datas.values():
            x0, x1, y0, y1 = self._data_bounds(data)
            extents.append(max(x1 - x0, y1 - y0))
        return self.max_size / max(max(extents, default=1.0), 1e-12)

    def _figure_scale(self, dx: float, dy: float) -> float:
        """Pixels per data unit of one figure whose content is dx x dy."""
        if self.scale is not None:
            return float(self.scale)
        if self.fit_each:
            return self.max_size / max(dx, dy)
        return self._scale

    @staticmethod
    def _links_signature(links: List) -> str:
        return repr([(l.source_data, l.target_data, l.source_direction, l.target_direction) for l in links])

    @staticmethod
    def _same_geometry(a: Data2D, b: Data2D) -> bool:
        """Returns True if both Data2D have the same polygons (same cells, same coords)."""
        if len(a.polygons) != len(b.polygons):
            return False
        for pa, pb in zip(a.polygons, b.polygons):
            if pa.cell_id != pb.cell_id:
                return False
            if not np.array_equal(pa.exterior_polygon.x_coords, pb.exterior_polygon.x_coords):
                return False
            if not np.array_equal(pa.exterior_polygon.y_coords, pb.exterior_polygon.y_coords):
                return False
        return True