"""Tests for the Graph extension (node padding, node labels) and the graph coloring."""

import copy

import numpy as np
import pytest
pytest.importorskip("scivianna_graph")

import scivianna.utils

scivianna.utils._testing = True

from test_panel_graph import make_graph_panel

from scivianna.data.data2d import Data2D
from scivianna.data.data_graph import DataGraph, Link
from scivianna.enums import VisualizationMode
from scivianna.extension.field_selector import set_graph_colors_list
from scivianna.extension.graph import Graph
from scivianna.utils.polygonize_tools import PolygonCoords, PolygonElement


def get_labels_on_canvas(panel):
    """Returns {node id: label} of the nodes currently on the canvas.

    An absent label and an empty one both mean "no label" for the frontend,
    so they are normalized to None here.
    """
    return {
        node["id"]: (label if label else None)
        for node in panel.plotter.canvas.nodes
        for label in [node.get("data", {}).get("label")]
    }


@pytest.mark.graph
def test_graph_extension_exists():
    """The Graph extension is loaded by default and wired to the panel."""
    panel, cleanup = make_graph_panel()
    try:
        graph_ext = next(e for e in panel.extensions if isinstance(e, Graph))

        assert graph_ext.title == "Graph customization"
        assert graph_ext.icon is not None
        assert graph_ext.slave is panel.slave
        assert graph_ext.plotter is panel.plotter
        assert graph_ext.panel is panel
    finally:
        cleanup()


@pytest.mark.graph
def test_node_padding_updates_canvas():
    """Changing the padding input updates the canvas node_padding."""
    panel, cleanup = make_graph_panel()
    try:
        graph_ext = next(e for e in panel.extensions if isinstance(e, Graph))

        assert graph_ext.node_padding_inp.value == float(panel.plotter.canvas.node_padding)

        graph_ext.node_padding_inp.value = 25

        assert panel.plotter.canvas.node_padding == 25
    finally:
        cleanup()


@pytest.mark.graph
def test_hide_labels():
    """Toggling the checkbox hides all node labels, untoggling restores them."""
    panel, cleanup = make_graph_panel()
    try:
        graph_ext = next(e for e in panel.extensions if isinstance(e, Graph))

        initial = get_labels_on_canvas(panel)
        assert initial["generator"] == "Generator"
        assert initial["factory_1"] == "Factory 1"

        graph_ext.hide_labels_chk.value = True
        assert all(label is None or label == "" for label in get_labels_on_canvas(panel).values())

        graph_ext.hide_labels_chk.value = False
        assert get_labels_on_canvas(panel) == initial
    finally:
        cleanup()


@pytest.mark.graph
def test_hide_labels_survives_rebuild():
    """Hidden labels stay hidden when the canvas is rebuilt from scratch."""
    panel, cleanup = make_graph_panel()
    try:
        graph_ext = next(e for e in panel.extensions if isinstance(e, Graph))

        graph_ext.hide_labels_chk.value = True
        panel.plotter.plot(copy.deepcopy(panel.current_data))

        assert all(label is None or label == "" for label in get_labels_on_canvas(panel).values())
    finally:
        cleanup()


@pytest.mark.graph
def test_colorbar_position_select():
    """Changing the position select moves the colorbar on the canvas."""
    panel, cleanup = make_graph_panel()
    try:
        graph_ext = next(e for e in panel.extensions if isinstance(e, Graph))

        # The field selector displays the colorbar on first update, bottom-right by default.
        colorbar = panel.plotter.graph_builder.canvas.colorbar
        assert colorbar is not None
        assert colorbar["position"] == "bottom-right"

        graph_ext.colorbar_position_select.value = "top-left"

        assert panel.plotter.colorbar_position == "top-left"
        assert panel.plotter.graph_builder.canvas.colorbar["position"] == "top-left"
    finally:
        cleanup()


@pytest.mark.graph
def test_colorbar_orientation_select():
    """Changing the orientation select rotates the colorbar on the canvas."""
    panel, cleanup = make_graph_panel()
    try:
        graph_ext = next(e for e in panel.extensions if isinstance(e, Graph))

        colorbar = panel.plotter.graph_builder.canvas.colorbar
        assert colorbar is not None
        assert colorbar["orientation"] == "vertical"

        graph_ext.colorbar_orientation_select.value = "horizontal"

        assert panel.plotter.colorbar_orientation == "horizontal"
        assert panel.plotter.graph_builder.canvas.colorbar["orientation"] == "horizontal"
    finally:
        cleanup()


@pytest.mark.graph
def test_extension_json_round_trip():
    """The extension state (padding, labels, colorbar placement) survives to_json / from_json."""
    panel, cleanup = make_graph_panel()
    try:
        graph_ext = next(e for e in panel.extensions if isinstance(e, Graph))

        graph_ext.node_padding_inp.value = 15
        graph_ext.hide_labels_chk.value = True
        graph_ext.colorbar_position_select.value = "top-left"
        graph_ext.colorbar_orientation_select.value = "horizontal"

        info = graph_ext.to_json()
        assert info["node_padding"] == 15
        assert info["hide_labels"] is True
        assert info["colorbar_position"] == "top-left"
        assert info["colorbar_orientation"] == "horizontal"

        restored = Graph.from_json(
            graph_ext,
            {
                "node_padding": 5,
                "hide_labels": False,
                "colorbar_position": "top-right",
                "colorbar_orientation": "vertical",
            },
        )
        assert restored.node_padding_inp.value == 5
        assert restored.hide_labels_chk.value is False
        assert restored.colorbar_position_select.value == "top-right"
        assert restored.colorbar_orientation_select.value == "vertical"
        assert panel.plotter.canvas.node_padding == 5
        assert panel.plotter.colorbar_position == "top-right"
        assert panel.plotter.colorbar_orientation == "vertical"
    finally:
        cleanup()


@pytest.mark.graph
@pytest.mark.parametrize("color_map", ["BuRd", "viridis"])
def test_heat_network_colors(color_map):
    """The heat network example is colored with a global value range, as expected.

    The values span 140 (factory_2) to 190 (generator band 2 / pipe inlets). With
    the default BuRd map: low values are blue, high values red, and every node of
    the same value shares the exact same color. viridis is checked too: it is a
    sequential map (dark -> light), so the same-value / monotonicity properties
    hold while the endpoint checks don't.
    """
    from scivianna_example.graph import build_graph

    graph = build_graph()
    set_graph_colors_list(graph, color_map)

    # Every node got colors (and edge colors) for all its cells.
    for key, data in graph.datas.items():
        assert len(data.cell_colors) == len(data.cell_ids)
        assert len(data.cell_edge_colors) == len(data.cell_ids)

    def color_of(node: str, index: int = 0):
        """Return the RGB color of a node's cell as an int tuple."""
        return tuple(int(c) for c in graph.datas[node].cell_colors[index])

    low = color_of("factory_2")   # value 140 -> minimum of the range
    high = color_of("generator", 2)  # value 190 -> maximum of the range

    if color_map == "BuRd":
        # The colormapped endpoints of BuRd.
        assert low[2] > low[0], f"the lowest value should be blue, found {low}"
        assert high[0] > high[2], f"the highest value should be red, found {high}"

        # A middle value is strictly between the two endpoints, on the same axis.
        mid = color_of("factory_1")  # value 150
        assert low[2] < mid[2] < high[2] or low[0] < mid[0] < high[0], \
            f"the middle value should sit between the endpoints, found {mid}"

    else:
        # viridis is sequential: the minimum is dark, the maximum is light.
        assert sum(low[:3]) < 3 * 120, f"the lowest value should be dark, found {low}"
        assert sum(high[:3]) > 3 * 150, f"the highest value should be light, found {high}"

    # Same value -> same color, across different nodes.
    assert color_of("generator", 0) == color_of("factory_2")      # both 140
    assert color_of("generator", 2) == color_of("pipe_1", 0)      # both 190

    # The colors are monotonic along a pipe: values decrease from inlet to outlet
    # (190 -> 170). For BuRd the pipe sits on the red side of the diverging map,
    # so each segment moves closer to the white middle. For viridis (sequential)
    # the whole pipe is in the bright part of the map, where RGB distance to any
    # fixed reference is not monotonic: check the value order directly instead.
    pipe_1 = graph.datas["pipe_1"]
    if color_map == "BuRd":
        for k in range(len(pipe_1.cell_ids) - 1):
            dist_k = np.linalg.norm(np.asarray(pipe_1.cell_colors[k], dtype=float) - 255)
            dist_k1 = np.linalg.norm(np.asarray(pipe_1.cell_colors[k + 1], dtype=float) - 255)
            assert dist_k > dist_k1, \
                f"colors should move toward the white middle along the cooling pipe, " \
                f"segment {k}: {dist_k:.1f} -> {dist_k1:.1f}"
    else:
        from scivianna.utils.color_tools import interpolate_cmap_at_values

        # Each color is the colormap at that cell's normalized value.
        t = (np.asarray(pipe_1.cell_values, dtype=float) - 140.0) / (190.0 - 140.0)
        expected = np.asarray(interpolate_cmap_at_values(color_map, t), dtype=float)
        got = np.asarray(pipe_1.cell_colors, dtype=float)
        assert np.allclose(got, expected, atol=1), "pipe colors should follow the value order"


@pytest.mark.graph
@pytest.mark.parametrize("color_map", ["BuRd", "viridis"])
def test_graph_colors_forced_range_and_centering(color_map):
    """Forced min/max bounds and centering on zero change the global color range.

    With a forced max below the data maximum, values above it saturate at the
    top of the colormap; with a forced min only, the minimum stays at the data
    value; centering on zero makes the range symmetric around 0.
    """
    from scivianna_example.graph import build_graph
    from scivianna.utils.color_tools import interpolate_cmap_at_values

    # Global data range is [140, 190] (see test_heat_network_colors).
    graph = build_graph()
    set_graph_colors_list(graph, color_map)
    color_default = tuple(int(c) for c in graph.datas["generator"].cell_colors[2])

    # Force the max below the data maximum: values above it saturate at the top.
    graph = build_graph()
    vmin, vmax = set_graph_colors_list(graph, color_map, min_value=140.0, max_value=180.0)
    assert (vmin, vmax) == (140.0, 180.0)
    # The top of the colormap is not unique to the maximum: with BuRd every value
    # >= 180 maps to it, so check saturation against the top color directly.
    top_color = tuple(int(c) for c in interpolate_cmap_at_values(color_map, np.array([1.0]))[0])
    assert tuple(int(c) for c in graph.datas["generator"].cell_colors[2]) == top_color  # 190 >= 180
    assert tuple(int(c) for c in graph.datas["pipe_1"].cell_colors[0]) == top_color     # 190 >= 180

    # Force only the min: the max stays at the data maximum.
    graph = build_graph()
    vmin, vmax = set_graph_colors_list(graph, color_map, min_value=150.0)
    assert (vmin, vmax) == (150.0, 190.0)

    # Center on zero: the range is symmetric around 0, so it spans [-190, 190].
    graph = build_graph()
    vmin, vmax = set_graph_colors_list(graph, color_map, center_colormap_on_zero=True)
    assert (vmin, vmax) == (-190.0, 190.0)


def _string_node(name: str, values: list[str]) -> Data2D:
    """A single-cell-per-value node with string cell values."""
    cells = [
        PolygonElement(
            exterior_polygon=PolygonCoords([0, 1, 1, 0], [0, 0, 1, 1]),
            holes=[],
            cell_id=f"{name}_{k}",
        )
        for k in range(len(values))
    ]
    data = Data2D.from_polygon_list(cells)
    data.cell_values = np.array(values)
    return data


@pytest.mark.graph
@pytest.mark.parametrize("color_map", ["BuRd", "viridis"])
def test_graph_colors_from_string(color_map):
    """set_graph_colors_list colors string values like set_colors_list does.

    Each distinct string value gets one color (a hash of the string), shared by
    every cell holding that value across all nodes; the seed changes the colors
    and no numeric range is returned (string coloring has no scale).
    """
    # Two nodes sharing the same three labels: node_a holds A, B, C once each,
    # node_b repeats A and B so the cross-node "same value -> same color" check
    # is meaningful.
    graph = DataGraph(
        datas={
            "node_a": _string_node("node_a", ["A", "B", "C"]),
            "node_b": _string_node("node_b", ["A", "B"]),
        },
        locations={"node_a": (0.0, 0.0), "node_b": (100.0, 0.0)},
        links=[Link(source_data="node_a", target_data="node_b", source_direction="right", target_direction="left")],
    )

    # String mode is opt-in: the default stays FROM_VALUE and would fail on strings.
    with pytest.raises(ValueError):
        set_graph_colors_list(graph, color_map)

    vmin, vmax = set_graph_colors_list(graph, color_map, coloring_mode=VisualizationMode.FROM_STRING)

    # No numeric range exists for string values.
    assert (vmin, vmax) == (None, None)

    def color_of(node: str, index: int):
        return tuple(int(c) for c in graph.datas[node].cell_colors[index])

    # Every cell got a color and an edge color.
    for key, data in graph.datas.items():
        assert len(data.cell_colors) == len(data.cell_ids)
        assert len(data.cell_edge_colors) == len(data.cell_ids)

    # Same value -> same color, across nodes; different values -> different colors.
    assert color_of("node_a", 0) == color_of("node_b", 0)   # both "A"
    assert color_of("node_a", 1) == color_of("node_b", 1)   # both "B"
    assert color_of("node_a", 0) != color_of("node_a", 1)   # "A" vs "B"
    assert color_of("node_a", 2) not in (color_of("node_a", 0), color_of("node_a", 1))  # "C"

    # The seed changes the colors.
    graph = DataGraph(
        datas={"node_a": _string_node("node_a", ["A", "B", "C"])},
        locations={"node_a": (0.0, 0.0)},
        links=[],
    )
    set_graph_colors_list(graph, color_map, coloring_mode=VisualizationMode.FROM_STRING, seed=42)
    colors_42 = [tuple(int(c) for c in row) for row in graph.datas["node_a"].cell_colors]

    graph = DataGraph(
        datas={"node_a": _string_node("node_a", ["A", "B", "C"])},
        locations={"node_a": (0.0, 0.0)},
        links=[],
    )
    set_graph_colors_list(graph, color_map, coloring_mode=VisualizationMode.FROM_STRING, seed=43)
    colors_43 = [tuple(int(c) for c in row) for row in graph.datas["node_a"].cell_colors]

    assert colors_42 != colors_43
