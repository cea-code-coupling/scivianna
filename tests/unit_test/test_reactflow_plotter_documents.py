"""ReactFlowPlotter keeps the browser in sync when its figures change document.

A Bokeh figure belongs to one document; showing the canvas in a second one (another
tab or session served from the same object) moves the figures there, and in-place
updates stop reaching the first display. The plotter must rebuild the figures then.
"""

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


def _show(plotter: ReactFlowPlotter) -> Document:
    """Displays the canvas in a new document, like a browser session does."""
    doc = Document()
    doc.add_root(plotter.canvas.get_root(doc))
    return doc


@pytest.mark.graph
def test_update_in_place_when_figures_are_owned():
    """Same nodes, one display: the figures are updated, not replaced."""
    plotter = ReactFlowPlotter()
    plotter.plot(_graph())
    _show(plotter)
    fig = plotter.plotters["a"].figure

    plotter.update_plot(_graph(length=6.0))

    assert plotter.plotters["a"].figure is fig


@pytest.mark.graph
def test_rebuild_when_shown_in_two_documents():
    """With a second display the first one would miss in-place updates: new figures are built."""
    plotter = ReactFlowPlotter()
    plotter.plot(_graph())
    _show(plotter)
    _show(plotter)
    fig = plotter.plotters["a"].figure

    plotter.update_plot(_graph(length=6.0))

    assert plotter.plotters["a"].figure is not fig


@pytest.mark.graph
def test_rebuild_drops_removed_nodes():
    """A rebuild with fewer nodes does not keep the plotters of the removed ones."""
    plotter = ReactFlowPlotter()
    plotter.plot(_graph(("a", "b", "c")))

    plotter.update_plot(_graph(("a", "b")))

    assert set(plotter.plotters) == {"a", "b"}
    assert set(plotter.panes) == {"a", "b"}


@pytest.mark.graph
def test_failed_update_forces_a_rebuild(monkeypatch):
    """An update that fails half way forgets the graph, so the next one starts from scratch."""
    plotter = ReactFlowPlotter()
    plotter.plot(_graph())

    def boom(*args, **kwargs):
        raise RuntimeError("update failed")

    monkeypatch.setattr(plotter.graph_builder, "update_view_sizes", boom)
    with pytest.raises(RuntimeError):
        plotter.update_plot(_graph(length=6.0))
    assert plotter.graph is None

    monkeypatch.undo()
    fig = plotter.plotters["a"].figure
    plotter.update_plot(_graph(length=6.0))
    assert plotter.plotters["a"].figure is not fig
