"""Tests for the graph node callbacks (selection change, nodes moved).

The frontend messages are simulated by feeding plain dicts to
``SciviannaGraph._handle_msg``, the same way the mouse callback tests feed data
to ``Bokeh2DPolygonPlotter.send_event``.
"""

import pytest

import scivianna.utils

scivianna.utils._testing = True

from test_interface import make_panel_2d

from scivianna.extension.extension import Extension
from scivianna.panel.panel_graph import PanelGraph
from scivianna.slave import ComputeSlave
from scivianna_example.graph.heat_network_interface import HeatNetworkInterface


class TrackingGraphExtension(Extension):
    """Extension tracking node selection and move events, like DummyTestExtension."""

    def __init__(self, slave, plotter, panel):
        super().__init__("Tracking graph", "", slave, plotter, panel)

        self.selection_history = []
        self.moved_history = []

    def on_node_selection_change(self, selected_nodes):
        self.selection_history.append(selected_nodes)

    def on_nodes_moved(self, moved_nodes):
        self.moved_history.append(moved_nodes)


def make_graph_panel_with_tracking():
    """Creates a PanelGraph with a tracking extension.

    Returns
    -------
    tuple
        (panel, tracking_ext, cleanup) where cleanup terminates the slave process.
    """
    slave = ComputeSlave(HeatNetworkInterface)
    panel = PanelGraph(name="Graph callbacks", slave=slave, extensions=[TrackingGraphExtension])

    def cleanup():
        slave.terminate()

    return panel, next(e for e in panel.extensions if isinstance(e, TrackingGraphExtension)), cleanup


def send_message(panel, payload):
    """Sends a frontend message to the canvas, as Panel would deliver it."""
    panel.plotter.graph_builder._handle_msg(payload)


@pytest.mark.graph
def test_selection_changed_reaches_extension():
    """A selection_changed message is dispatched to the extension's on_node_selection_change."""
    panel, tracking, cleanup = make_graph_panel_with_tracking()
    try:
        send_message(panel, {"type": "selection_changed", "ids": ["pipe_1", "generator"]})

        assert len(tracking.selection_history) == 1
        assert tracking.selection_history[0] == ["pipe_1", "generator"]
    finally:
        cleanup()


@pytest.mark.graph
def test_selection_cleared_reaches_extension():
    """Clearing the selection (empty ids) is dispatched too."""
    panel, tracking, cleanup = make_graph_panel_with_tracking()
    try:
        send_message(panel, {"type": "selection_changed", "ids": []})

        assert tracking.selection_history == [[]]
    finally:
        cleanup()


@pytest.mark.graph
def test_nodes_moved_reaches_extension_and_model():
    """A nodes_moved message updates the model positions and reaches on_nodes_moved."""
    panel, tracking, cleanup = make_graph_panel_with_tracking()
    try:
        send_message(panel, {"type": "nodes_moved", "nodes": [{"id": "pipe_2", "x": 10.0, "y": -50.0}]})

        assert len(tracking.moved_history) == 1
        assert tracking.moved_history[0] == [{"id": "pipe_2", "x": 10.0, "y": -50.0}]
        assert panel.plotter.graph_builder.model.get_node("pipe_2").position == (10.0, -50.0)
    finally:
        cleanup()


@pytest.mark.graph
def test_nodes_moved_multi_selection():
    """A multi-selection drag reports every dragged node in one call."""
    panel, tracking, cleanup = make_graph_panel_with_tracking()
    try:
        send_message(
            panel,
            {
                "type": "nodes_moved",
                "nodes": [
                    {"id": "pipe_1", "x": 1.0, "y": 2.0},
                    {"id": "pipe_3", "x": 3.0, "y": 4.0},
                ],
            },
        )

        assert len(tracking.moved_history) == 1
        assert {node["id"] for node in tracking.moved_history[0]} == {"pipe_1", "pipe_3"}
    finally:
        cleanup()


@pytest.mark.graph
def test_nodes_moved_unknown_id_skipped():
    """An unknown node id is skipped without crashing nor calling the callback."""
    panel, tracking, cleanup = make_graph_panel_with_tracking()
    try:
        send_message(panel, {"type": "nodes_moved", "nodes": [{"id": "does_not_exist", "x": 0.0, "y": 0.0}]})

        assert tracking.moved_history == []
    finally:
        cleanup()


@pytest.mark.graph
def test_unknown_message_type_ignored():
    """Messages of unknown type are ignored."""
    panel, tracking, cleanup = make_graph_panel_with_tracking()
    try:
        send_message(panel, {"type": "something_else"})

        assert tracking.selection_history == []
        assert tracking.moved_history == []
    finally:
        cleanup()


@pytest.mark.graph
def test_2d_panel_unaffected_by_new_hooks():
    """Panel2D still builds with the new extension hooks (they are no-ops there)."""
    panel, _, cleanup = make_panel_2d()
    try:
        assert hasattr(panel.extensions[0], "on_node_selection_change")
        assert hasattr(panel.extensions[0], "on_nodes_moved")
    finally:
        cleanup()
