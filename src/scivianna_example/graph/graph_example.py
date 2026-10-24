"""Graph example entry point.

Run with:
    panel serve graph_example.py --show
or from a notebook:
    make_graph_panel().main_frame.show()
"""

from scivianna.panel.panel_graph import PanelGraph
from scivianna_example.graph.graph import make_graph
# from scivianna_example.graph.aled import make_graph


def make_graph_panel(**plotter_kwargs) -> PanelGraph:
    """Builds the graph panel of the example.

    Parameters
    ----------
    **plotter_kwargs
        Forwarded to :class:`ReactFlowPlotter` (e.g. ``max_size``, ``scale``,
        ``color_map_name``).

    Returns
    -------
    PanelGraph
        The graph panel displaying the example DataGraph.
    """
    return PanelGraph(name="Graph example", graph=make_graph(), **plotter_kwargs)


if __name__ == "__main__":
    make_graph_panel().main_frame.show()
