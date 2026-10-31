from pathlib import Path
from typing import TYPE_CHECKING

import panel as pn
import panel_material_ui as pmui

import scivianna.utils
from scivianna.extension.extension import Extension
from scivianna.icon import get_icon
from scivianna.plotter_graph.generic_plotter import PlotterGraph

if TYPE_CHECKING:
    from scivianna.panel.visualisation_panel import VisualizationPanel
    from scivianna.slave import ComputeSlave


class Graph(Extension):
    """Extension to customize the graph canvas (node padding)."""

    def __init__(self, slave: "ComputeSlave", plotter: PlotterGraph, panel: "VisualizationPanel"):
        """Constructor of the extension, saves the slave and the panel

        Parameters
        ----------
        slave : ComputeSlave
            Slave computing the displayed data
        plotter : PlotterGraph
            Figure plotter
        panel : VisualizationPanel
            Panel to which the extension is attached
        """
        assert isinstance(plotter, PlotterGraph), "Graph extension is built for a PlotterGraph only"
        super().__init__(
            "Graph customization",
            get_icon("flow_graph"),
            slave,
            plotter,
            panel,
        )

        self.description = """
The graph extension allows you to customize the canvas layout.

You can change the padding between a node's border and its content (label and figure),
hide all the node labels at once, and re-fit the viewport to the drawn nodes.
"""

        self.iconsize = "1.0em"

        self.node_padding_inp = pmui.FloatInput(
            label="Node padding",
            value=float(self.plotter.canvas.node_padding),
            start=0,
            end=200,
            step=1,
            width=125,
            margin=5,
            align="center",
        )

        self.hide_labels_chk = pmui.Checkbox(
            label="Hide node labels",
            value=False,
            margin=5,
        )

        self.fit_view_button = pmui.Button(
            label="Fit view",
            width=125,
            icon=(Path(scivianna.__file__).parent / "icon" / "fit_screen.svg").read_text().strip(),
            margin=5,
        )
        self.fit_view_button.on_click(self._fit_view)

        def update_node_padding(*args, **kwargs):
            if self._restoring:
                return
            self.plotter.canvas.node_padding = self.node_padding_inp.value
            # The node size depends on the padding: the canvas must re-read the
            # figures sizes so the nodes follow.
            self.plotter.graph_builder.update_view_sizes()

        def update_hide_labels(*args, **kwargs):
            if self._restoring:
                return
            self.plotter.set_labels_visible(not self.hide_labels_chk.value)

        self.node_padding_inp.param.watch(update_node_padding, "value")
        self.hide_labels_chk.param.watch(update_hide_labels, "value")

    def _fit_view(self, *args, **kwargs):
        """Re-fits the canvas viewport to the drawn nodes."""
        self.plotter.fit_view()

    def make_gui(self) -> pn.viewable.Viewable:
        """Returns a panel viewable to display in the extension tab.

        Returns
        -------
        pn.viewable.Viewable
            Viewable to display in the extension tab
        """
        return pmui.Column(
            pmui.Typography("Node padding (px)"),
            self.node_padding_inp,
            self.hide_labels_chk,
            self.fit_view_button,
            margin=0,
        )

    def to_json(self) -> dict:
        """Returns a dictionary with the information required to rebuild the extension.

        Returns
        -------
        dict
            Information dictionary
        """
        return {
            "node_padding": self.node_padding_inp.value,
            "hide_labels": self.hide_labels_chk.value,
        }

    @classmethod
    def from_json(cls, extension: "Graph", info_dict: dict) -> "Graph":
        """Restores the extension from its information dict.

        Parameters
        ----------
        extension : Graph
            Extension instance to restore
        info_dict : dict
            Dictionary containing extension state information

        Returns
        -------
        Graph
            Restored extension
        """
        extension._restoring = True
        extension.node_padding_inp.value = info_dict.get("node_padding", 0)
        extension.plotter.canvas.node_padding = extension.node_padding_inp.value
        extension.hide_labels_chk.value = info_dict.get("hide_labels", False)
        extension._restoring = False
        return extension
