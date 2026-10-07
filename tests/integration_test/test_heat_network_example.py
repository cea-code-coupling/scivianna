"""Ensuring that the heat network graph example builds end to end."""

import pytest

import scivianna.utils

scivianna.utils._testing = True


@pytest.mark.graph
def test_heat_network_example():
    """The example panel builds its graph through the slave and terminates cleanly."""
    from scivianna_example.graph import make_graph_panel

    panel = make_graph_panel()

    try:
        assert len(panel.current_data.datas) == 7
        assert len(panel.current_data.links) == 6
        assert panel.displayed_field == "Temperature"

        # A recompute goes back to the slave and keeps the graph.
        panel.recompute()
        assert len(panel.current_data.datas) == 7
    finally:
        panel.get_slave().terminate()


if __name__ == "__main__":
    test_heat_network_example()
