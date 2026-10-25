"""Tests for the Graph extension (node padding, node labels)."""

import copy

import pytest

import scivianna.utils

scivianna.utils._testing = True

from test_panel_graph import make_graph_panel

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
