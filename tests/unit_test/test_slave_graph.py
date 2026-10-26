"""Tests for the GraphInterface commands of ComputeSlave (compute_graph, get_graph_value_dict)."""

import numpy as np
import pytest
pytest.importorskip("scivianna_graph")

from scivianna.interface.csv_result import CSVInterface
from scivianna.interface.generic_interface import GraphInterface
from scivianna.slave import ComputeSlave
from scivianna_example.graph.heat_network_interface import HeatNetworkInterface


@pytest.mark.graph
def test_compute_graph_fills_cell_values():
    """compute_graph returns the graph and fills every node's cell_values from get_graph_value_dict."""
    slave = ComputeSlave(HeatNetworkInterface)
    try:
        graph, graph_updated = slave.compute_graph(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
            None,
            "Temperature",
            {},
        )

        # The example graph is static: the structure never changes.
        assert graph_updated is False
        assert len(graph.datas) == 7
        assert len(graph.links) == 6

        # Every node has values matching the interface's own mapping, cell by cell.
        expected = slave.get_graph_value_dict(
            "Temperature", [cell for data in graph.datas.values() for cell in data.cell_ids], {}
        )
        for key, data in graph.datas.items():
            assert len(data.cell_values) == len(data.cell_ids)
            for cell_id, value in zip(data.cell_ids, data.cell_values):
                assert value == expected[cell_id]
    finally:
        slave.terminate()


@pytest.mark.graph
def test_get_graph_value_dict_round_trip():
    """get_graph_value_dict returns the interface's values for the requested cells."""
    slave = ComputeSlave(HeatNetworkInterface)
    try:
        graph, _ = slave.compute_graph(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
            None,
            "Temperature",
            {},
        )

        pipe_1 = graph.datas["pipe_1"]
        cells = list(pipe_1.cell_ids)
        values = slave.get_graph_value_dict("Temperature", cells, {})

        assert set(values.keys()) == set(cells)
        for cell_id, value in zip(cells, pipe_1.cell_values):
            assert values[cell_id] == value
    finally:
        slave.terminate()


@pytest.mark.graph
def test_compute_graph_rejects_non_graph_interface():
    """A slave whose interface is not a GraphInterface raises on compute_graph."""
    slave = ComputeSlave(CSVInterface, allow_errors=True)
    try:
        val = slave.compute_graph(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
            None,
            "Temperature",
            {},
        )
    except TypeError:
        assert True, "TypeError caught"
    else:
        assert val is None, "compute_graph should not succeed on a non GraphInterface slave"
    finally:
        slave.terminate()
