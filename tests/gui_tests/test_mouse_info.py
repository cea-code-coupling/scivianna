"""Tests for get_mouse_info on each implemented plotter backend.

The mouse state is simulated the same way as in test_mouse_europe.py: by writing
directly into the plotter's ColumnDataSource (or, for VTK, into the pyvista
plotter params) and then calling get_mouse_info. The _mouse_inside flag is set
manually to simulate a MouseEnter event.
"""

import pytest

import scivianna.utils

scivianna.utils._testing = True

from test_interface import make_panel_2d


# ---------------------------------------------------------------------------
# Bokeh 2D polygon plotter (the default Panel2D backend)
# ---------------------------------------------------------------------------


def _simulate_hover_polygon(plotter, index: int, x=0.5, y=0.3, z=0.0):
    """Sets the Bokeh polygon plotter mouse state as if hovering over cell at ``index``."""
    plotter._mouse_inside = True
    plotter.source_mouse.data = {
        "sx": [100.0],
        "sy": [50.0],
        "x": [x],
        "y": [y],
        "z": [z],
        "index": [index],
        "u": [x],
        "v": [y],
        "snap_sx": [100.0],
        "snap_sy": [50.0],
    }


def _simulate_leave_polygon(plotter):
    """Simulates the mouse leaving the figure."""
    plotter._mouse_inside = False


@pytest.mark.default
def test_bokeh_polygon_mouse_info_hovering():
    """get_mouse_info returns coordinates, cell id and value when hovering a cell."""
    panel, _, cleanup = make_panel_2d()
    try:
        plotter = panel.plotter
        _simulate_hover_polygon(plotter, index=1, x=0.5, y=0.7, z=0.0)

        info = plotter.get_mouse_info()

        assert "0.500" in info
        assert "0.700" in info
        assert "Cell: 1" in info
    finally:
        cleanup()


@pytest.mark.default
def test_bokeh_polygon_mouse_info_not_hovering():
    """get_mouse_info returns an empty string when the mouse is not over the figure."""
    panel, _, cleanup = make_panel_2d()
    try:
        plotter = panel.plotter
        _simulate_leave_polygon(plotter)

        assert plotter.get_mouse_info() == ""
    finally:
        cleanup()


@pytest.mark.default
def test_bokeh_polygon_mouse_info_out_of_bounds():
    """get_mouse_info still returns coordinates when the index is out of bounds (no cell info)."""
    panel, _, cleanup = make_panel_2d()
    try:
        plotter = panel.plotter
        _simulate_hover_polygon(plotter, index=999, x=1.0, y=2.0, z=3.0)

        info = plotter.get_mouse_info()

        assert "1.000" in info
        assert "Cell:" not in info
    finally:
        cleanup()


# ---------------------------------------------------------------------------
# Bokeh 2D grid plotter (Mandelbrot example, display_polygons=False)
# ---------------------------------------------------------------------------


@pytest.mark.default
def test_bokeh_grid_mouse_info_hovering():
    """get_mouse_info returns coordinates and cell id when hovering a grid cell."""
    from scivianna_example.mandelbrot import mandelbrot

    layout, slaves = mandelbrot.make_panel(None, return_slaves=True)
    try:
        grid_panel = layout.visualisation_panels["Mandelbrot grid"]
        plotter = grid_panel.plotter

        # Simulate hovering cell (i=2, j=3)
        plotter._mouse_inside = True
        plotter.source_mouse.data = {
            "sx": [10.0],
            "sy": [20.0],
            "x": [0.5],
            "y": [0.6],
            "z": [0.0],
            "i": [2],
            "j": [3],
        }

        info = plotter.get_mouse_info()

        assert "0.500" in info
        assert "0.600" in info
        if plotter.cell_name_grid is not None:
            expected_cell = plotter.cell_name_grid[3, 2]
            assert f"Cell: {expected_cell}" in info
    finally:
        for slave in slaves:
            slave.terminate()


@pytest.mark.default
def test_bokeh_grid_mouse_info_not_hovering():
    """get_mouse_info returns an empty string when the mouse is not over the grid figure."""
    from scivianna_example.mandelbrot import mandelbrot

    layout, slaves = mandelbrot.make_panel(None, return_slaves=True)
    try:
        grid_panel = layout.visualisation_panels["Mandelbrot grid"]
        plotter = grid_panel.plotter

        plotter._mouse_inside = False

        assert plotter.get_mouse_info() == ""
    finally:
        for slave in slaves:
            slave.terminate()


# ---------------------------------------------------------------------------
# VTK 2D polygon plotter (scivianna_vtk backend)
# ---------------------------------------------------------------------------


@pytest.mark.pyvista
def test_vtk_2d_mouse_info_hovering():
    """get_mouse_info returns coordinates, cell id and value when hovering a VTK cell."""
    pytest.importorskip("scivianna_vtk")

    from scivianna_example.vtk_example.demo_3d import get_panel as vtk_get_panel

    layout, slaves = vtk_get_panel(None, return_slaves=True)
    try:
        panel_2d = layout.visualisation_panels["VTK slice"]
        plotter = panel_2d.plotter

        # Simulate hover state on the underlying pyvista plotter
        plotter.plotter.hover_position = [1.5, 2.5, 0.0]
        plotter.plotter.hover_cell_id = "cell_42"
        plotter.plotter.hover_cell_value = 99.7

        info = plotter.get_mouse_info()

        assert "1.500" in info
        assert "2.500" in info
        assert "Cell: cell_42" in info
        assert "Value: 99.7" in info
    finally:
        for slave in slaves:
            slave.terminate()


@pytest.mark.pyvista
def test_vtk_2d_mouse_info_no_hover():
    """get_mouse_info returns an empty string when no cell is hovered (NaN position)."""
    pytest.importorskip("scivianna_vtk")

    import numpy as np

    from scivianna_example.vtk_example.demo_3d import get_panel as vtk_get_panel

    layout, slaves = vtk_get_panel(None, return_slaves=True)
    try:
        panel_2d = layout.visualisation_panels["VTK slice"]
        plotter = panel_2d.plotter

        plotter.plotter.hover_position = [float("nan"), float("nan"), float("nan")]
        plotter.plotter.hover_cell_id = None
        plotter.plotter.hover_cell_value = None

        assert plotter.get_mouse_info() == ""
    finally:
        for slave in slaves:
            slave.terminate()


# ---------------------------------------------------------------------------
# ReactFlow graph plotter (scivianna-graph)
# ---------------------------------------------------------------------------


@pytest.mark.graph
def test_reactflow_mouse_info_hovering_node():
    """get_mouse_info returns the node key and cell info when hovering a graph node."""
    pytest.importorskip("scivianna_graph")

    from scivianna.panel.panel_graph import PanelGraph
    from scivianna.slave import ComputeSlave
    from scivianna_example.graph.heat_network_interface import HeatNetworkInterface

    slave = ComputeSlave(HeatNetworkInterface)
    panel = PanelGraph(name="mouse_info", slave=slave)
    try:
        plotter = panel.plotter

        # Pick the first node and simulate a hover on its Bokeh plotter
        node_key = list(plotter.plotters.keys())[0]
        node_plotter = plotter.plotters[node_key]

        cell_names = node_plotter.source_polygons.data["cell_names"]
        assert len(cell_names) > 0, "Node should have cells"

        node_plotter._mouse_inside = True
        node_plotter.source_mouse.data = {
            "sx": [5.0],
            "sy": [5.0],
            "x": [0.3],
            "y": [0.7],
            "z": [0.0],
            "index": [0],
            "u": [0.3],
            "v": [0.7],
            "snap_sx": [5.0],
            "snap_sy": [5.0],
        }

        info = plotter.get_mouse_info()

        assert info.startswith(f"[{node_key}]")
        assert "0.300" in info
        assert "Cell:" in info
    finally:
        slave.terminate()


@pytest.mark.graph
def test_reactflow_mouse_info_no_node_hovered():
    """get_mouse_info returns an empty string when no node is hovered."""
    pytest.importorskip("scivianna_graph")

    from scivianna.panel.panel_graph import PanelGraph
    from scivianna.slave import ComputeSlave
    from scivianna_example.graph.heat_network_interface import HeatNetworkInterface

    slave = ComputeSlave(HeatNetworkInterface)
    panel = PanelGraph(name="mouse_info", slave=slave)
    try:
        plotter = panel.plotter

        # All nodes report "not inside" (default state)
        for node_plotter in plotter.plotters.values():
            node_plotter._mouse_inside = False

        assert plotter.get_mouse_info() == ""
    finally:
        slave.terminate()


@pytest.mark.graph
def test_reactflow_mouse_info_only_active_node_reported():
    """Only the node with _mouse_inside=True is reported, not the first one in the dict."""
    pytest.importorskip("scivianna_graph")

    from scivianna.panel.panel_graph import PanelGraph
    from scivianna.slave import ComputeSlave
    from scivianna_example.graph.heat_network_interface import HeatNetworkInterface

    slave = ComputeSlave(HeatNetworkInterface)
    panel = PanelGraph(name="mouse_info", slave=slave)
    try:
        plotter = panel.plotter
        keys = list(plotter.plotters.keys())
        assert len(keys) >= 2, "Need at least two nodes for this test"

        # The LAST node is hovered, not the first
        target_key = keys[-1]
        target_plotter = plotter.plotters[target_key]
        cell_names = target_plotter.source_polygons.data["cell_names"]

        target_plotter._mouse_inside = True
        target_plotter.source_mouse.data = {
            "sx": [1.0],
            "sy": [2.0],
            "x": [0.9],
            "y": [0.1],
            "z": [0.0],
            "index": [0],
            "u": [0.9],
            "v": [0.1],
            "snap_sx": [1.0],
            "snap_sy": [2.0],
        }

        # All other nodes are NOT hovered
        for key in keys:
            if key != target_key:
                plotter.plotters[key]._mouse_inside = False

        info = plotter.get_mouse_info()

        assert info.startswith(f"[{target_key}]")
        assert "0.900" in info
    finally:
        slave.terminate()
