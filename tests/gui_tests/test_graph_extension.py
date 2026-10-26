"""Tests for the Graph extension (node padding, node labels) and the graph coloring."""

import copy

import numpy as np
import pytest
pytest.importorskip("scivianna_graph")

import scivianna.utils

scivianna.utils._testing = True

from test_panel_graph import make_graph_panel

from scivianna.extension.field_selector import set_graph_colors_list
from scivianna.extension.graph import Graph


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
def test_extension_json_round_trip():
    """The extension state (padding, labels) survives to_json / from_json."""
    panel, cleanup = make_graph_panel()
    try:
        graph_ext = next(e for e in panel.extensions if isinstance(e, Graph))

        graph_ext.node_padding_inp.value = 15
        graph_ext.hide_labels_chk.value = True

        info = graph_ext.to_json()
        assert info["node_padding"] == 15
        assert info["hide_labels"] is True

        restored = Graph.from_json(graph_ext, {"node_padding": 5, "hide_labels": False})
        assert restored.node_padding_inp.value == 5
        assert restored.hide_labels_chk.value is False
        assert panel.plotter.canvas.node_padding == 5
    finally:
        cleanup()


@pytest.mark.graph
def test_heat_network_colors():
    """The heat network example is colored with a global value range, as expected.

    The values span 140 (factory_2) to 190 (generator band 2 / pipe inlets). With
    the default BuRd map: low values are blue, high values red, and every node of
    the same value shares the exact same color.
    """
    from scivianna_example.graph import build_graph

    graph = build_graph()
    set_graph_colors_list(graph, "BuRd")

    # Every node got colors (and edge colors) for all its cells.
    for key, data in graph.datas.items():
        assert len(data.cell_colors) == len(data.cell_ids)
        assert len(data.cell_edge_colors) == len(data.cell_ids)

    def color_of(node: str, index: int = 0):
        return tuple(int(c) for c in graph.datas[node].cell_colors[index])

    # The colormapped endpoints of BuRd.
    low = color_of("factory_2")   # value 140 -> minimum of the range
    high = color_of("generator", 2)  # value 190 -> maximum of the range
    assert low[2] > low[0], f"the lowest value should be blue, found {low}"
    assert high[0] > high[2], f"the highest value should be red, found {high}"

    # Same value -> same color, across different nodes.
    assert color_of("generator", 0) == color_of("factory_2")      # both 140
    assert color_of("generator", 2) == color_of("pipe_1", 0)      # both 190

    # A middle value is strictly between the two endpoints, on the same axis.
    mid = color_of("factory_1")  # value 150
    assert low[2] < mid[2] < high[2] or low[0] < mid[0] < high[0], \
        f"the middle value should sit between the endpoints, found {mid}"

    # The colors are monotonic along a pipe: values decrease from inlet to outlet
    # (190 -> 170), all on the red side of BuRd, so each segment moves closer to
    # the white middle.
    pipe_1 = graph.datas["pipe_1"]
    for k in range(len(pipe_1.cell_ids) - 1):
        dist_k = np.linalg.norm(np.asarray(pipe_1.cell_colors[k], dtype=float) - 255)
        dist_k1 = np.linalg.norm(np.asarray(pipe_1.cell_colors[k + 1], dtype=float) - 255)
        assert dist_k > dist_k1, "colors should move toward the white middle along the cooling pipe"
