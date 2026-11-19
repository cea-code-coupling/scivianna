"""ReactFlowPlotter keeps its figures on in-place updates and forgets removed nodes on rebuild."""

import pytest

pytest.importorskip("scivianna_graph")

from bokeh.document import Document

from scivianna.data.data_graph import DataGraph
from scivianna.plotter_graph.reactflow_plotter import ReactFlowPlotter
from scivianna_example.graph.heat_network import Pipe, pipe_data


def _graph(keys=("a", "b"), length: float = 4.0) -> DataGraph:
    """Graph of pipes, one per key, without links."""
    datas = {key: pipe_data(Pipe(key, length=length)) for key in keys}
    return DataGraph(datas, {key: (300.0 * i, 0.0) for i, key in enumerate(keys)}, [])


@pytest.mark.graph
def test_update_in_place_keeps_the_figures():
    """Same nodes: the displayed figures are updated, not replaced."""
    plotter = ReactFlowPlotter()
    plotter.plot(_graph())
    doc = Document()
    doc.add_root(plotter.canvas.get_root(doc))
    fig = plotter.plotters["a"].figure

    plotter.update_plot(_graph(length=6.0))

    assert plotter.plotters["a"].figure is fig
    assert fig.document is doc


@pytest.mark.graph
def test_rebuild_drops_removed_nodes():
    """A rebuild with fewer nodes does not keep the plotters of the removed ones."""
    plotter = ReactFlowPlotter()
    plotter.plot(_graph(("a", "b", "c")))

    plotter.update_plot(_graph(("a", "b")))

    assert set(plotter.plotters) == {"a", "b"}
    assert set(plotter.panes) == {"a", "b"}
