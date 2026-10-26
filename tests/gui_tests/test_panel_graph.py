"""Tests for PanelGraph: building, updating and serializing the graph panel."""

import copy

import numpy as np
import pytest
pytest.importorskip("scivianna_graph")

import scivianna.utils

scivianna.utils._testing = True

from scivianna.data.data_graph import DataGraph
from scivianna.extension.ai_assistant import AIAssistant
from scivianna.extension.field_selector import FieldSelector
from scivianna.extension.file_loader import FileLoader
from scivianna.extension.graph import Graph
from scivianna.interface.csv_result import CSVInterface
from scivianna.panel.panel_graph import PanelGraph
from scivianna.slave import ComputeSlave
from scivianna_example.graph.heat_network_interface import HeatNetworkInterface


def make_graph_panel():
    """Creates a PanelGraph instance with the heat network example interface.

    Returns
    -------
    tuple
        (panel, cleanup) where cleanup terminates the slave process.
    """
    slave = ComputeSlave(HeatNetworkInterface)
    panel = PanelGraph(name="Graph test", slave=slave)

    def cleanup():
        slave.terminate()

    return panel, cleanup


@pytest.mark.graph
def test_build_panel():
    """The panel builds the example graph: 7 nodes, 6 links."""
    panel, cleanup = make_graph_panel()
    try:
        assert isinstance(panel.current_data, DataGraph)
        assert len(panel.current_data.datas) == 7
        assert len(panel.current_data.links) == 6

        # Every node holds values (filled by the slave through get_graph_value_dict).
        for data in panel.current_data.datas.values():
            assert len(data.cell_values) == len(data.cell_ids)
    finally:
        cleanup()


@pytest.mark.graph
def test_default_extensions():
    """The graph panel loads its default extensions, including Graph."""
    panel, cleanup = make_graph_panel()
    try:
        extension_classes = [e.__class__ for e in panel.extensions]
        assert FileLoader in extension_classes
        assert FieldSelector in extension_classes
        assert Graph in extension_classes
        assert AIAssistant in extension_classes
    finally:
        cleanup()


@pytest.mark.graph
def test_recompute_keeps_node_set():
    """recompute re-requests the slave and keeps the same nodes."""
    panel, cleanup = make_graph_panel()
    try:
        old_keys = set(panel.current_data.datas.keys())

        panel.recompute()

        assert set(panel.current_data.datas.keys()) == old_keys
    finally:
        cleanup()


@pytest.mark.graph
def test_set_graph_updates_in_place():
    """set_graph displays a new graph (here a copy of the current one) without error."""
    panel, cleanup = make_graph_panel()
    try:
        new_graph = copy.deepcopy(panel.current_data)

        panel.set_graph(new_graph)

        assert set(panel.current_data.datas.keys()) == set(new_graph.datas.keys())
    finally:
        cleanup()


@pytest.mark.graph
def test_to_json_from_json_round_trip():
    """The panel information survives a to_json / from_json round trip."""
    panel, cleanup = make_graph_panel()
    try:
        info = panel.to_json()

        slave = ComputeSlave(HeatNetworkInterface)
        restored = PanelGraph.from_json(info, slave=slave)
        try:
            assert restored.panel_name == info["name"]
            assert restored.displayed_field == "Temperature"
            assert len(restored.current_data.datas) == 7
        finally:
            slave.terminate()
    finally:
        cleanup()


@pytest.mark.graph
def test_duplicate():
    """duplicate builds a new panel displaying the same graph, with its own slave."""
    panel, cleanup = make_graph_panel()
    try:
        dup = panel.duplicate()
        try:
            assert len(dup.current_data.datas) == 7
            assert dup.slave is not panel.slave
        finally:
            dup.slave.terminate()
    finally:
        cleanup()


@pytest.mark.graph
def test_rejects_non_graph_interface():
    """A PanelGraph can only be built on a GraphInterface slave."""
    slave = ComputeSlave(CSVInterface, allow_errors=True)
    try:
        with pytest.raises(AssertionError):
            PanelGraph(name="Not a graph", slave=slave)
    finally:
        slave.terminate()
