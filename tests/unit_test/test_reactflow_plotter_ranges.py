"""Figure ranges of the ReactFlowPlotter follow the data when a node's geometry changes."""

import pytest

pytest.importorskip("scivianna_graph")

from scivianna.data.data_graph import DataGraph
from scivianna.plotter_graph.reactflow_plotter import ReactFlowPlotter
from scivianna_example.graph.heat_network import Pipe, pipe_data


def _graph(length: float, width: float) -> DataGraph:
    datas = {"a": pipe_data(Pipe("a", length=length, width=width)), "b": pipe_data(Pipe("b", length=8.0))}
    return DataGraph(datas, {"a": (0, 0), "b": (500, 0)}, [])


@pytest.mark.graph
def test_ranges_have_no_bounds():
    """BokehJS freezes bounds="auto" at the first range and clamps later ranges into it:
    a node whose data grows would keep drawing its old extent (cropped polygons)."""
    plotter = ReactFlowPlotter()
    plotter.plot(_graph(4.0, 1.0))
    fig = plotter.plotters["a"].figure
    assert fig.x_range.bounds is None
    assert fig.y_range.bounds is None


@pytest.mark.graph
def test_update_plot_grows_ranges_and_size():
    """A larger geometry widens the ranges and the pane, and the canvas node gets the new size."""
    plotter = ReactFlowPlotter()
    plotter.plot(_graph(4.0, 1.0))
    fig = plotter.plotters["a"].figure
    old_w, old_h = fig.width, fig.height

    plotter.update_plot(_graph(8.0, 3.0))

    assert fig.x_range.start < -1.5 and fig.x_range.end > 1.5
    assert fig.y_range.start < 0.0 and fig.y_range.end > 8.0
    assert fig.width > old_w and fig.height > old_h
    node = next(n for n in plotter.canvas.nodes if n["id"] == "a")
    assert node["data"]["view_width"] == fig.width
    assert node["data"]["view_height"] == fig.height
