
from pathlib import Path

from bokeh.plotting import curdoc

import panel as pn
import panel_material_ui as pmui
from typing import TYPE_CHECKING

import scivianna
from scivianna.extension.extension import Extension
from scivianna.plotter_2d.generic_plotter import Plotter2D
from scivianna.slave import ComputeSlave

if TYPE_CHECKING:
    from scivianna.panel.visualisation_panel import VisualizationPanel
    from scivianna.layout.generic_layout import GenericLayout


class CouplingExtension(Extension):
    """Extension to start coupling simulations."""

    def __init__(
        self,
        layout: "GenericLayout",
        slave: ComputeSlave,
        plotter: Plotter2D,
        panel: "VisualizationPanel"
    ):
        """Constructor of the extension, saves the slave and the panel

        Parameters
        ----------
        slave : ComputeSlave
            Slave computing the displayed data
        plotter : Plotter2D
            Figure plotter
        panel : VisualizationPanel
            Panel to which the extension is attached
        """
        super().__init__(
            "Time management",
            "timer",
            slave,
            plotter,
            panel,
        )

        self.description = """
This extension allows you run coupling simulations.
"""

        self.run_button = pn.widgets.ButtonIcon(
            icon=(Path(scivianna.__file__).parent / "icon" / "player-play.svg").read_text().strip(),
            description="Start automatic update",
            height=30,
            width=30,
            align="center",
        )
        self.display_last = pmui.Checkbox(
            name = "Display last time", 
            value = True,
            width = 280
        )
        self.time_slider = pmui.DiscreteSlider(
            name = "Time selector", 
            value = 0., 
            options = [0.], 
            description="Time at which display the results",
            width = 260,
            show_value=False,
            size="small"
        )

        self.layout = layout

        self.run_button.on_click(self.request_recompute)
        self.time_slider.param.watch(self.request_recompute, "value")

        self.curdoc = curdoc()

        pn.state.curdoc = self.curdoc

        self.layout_param_card = pn.Column(
            self.run_button,
            self.display_last,
            self.time_slider,
            width=300,
            margin=0,
        )

    def make_gui(
        self,
    ) -> pn.viewable.Viewable:
        """Returns a panel viewable to display in the extension tab.

        Returns
        -------
        pn.viewable.Viewable
            Viewable to display in the extension tab
        """
        return pn.Column(
                pmui.Typography("Start/pause real time display"),
                self.layout_param_card,
                margin=(0, 0, 10, 10),
            )

    def request_recompute(self, event):
        """Request a recompute task on all panels, which will trigger the addition of a periodict update on the panels

        Parameters
        ----------
        event : bool
            If the call is from a button press or release
        """
        if event:
            if event.obj == self.run_button:
                if self.layout.periodic_recompute_added:
                    self.run_button.icon = "player-play"
                    self.layout.stop_periodic_update()
                else:
                    self.run_button.icon = "player-pause"
                    self.layout.add_periodic_update()
            else:
                self.layout.recompute()

    def provide_options(self):
        option_dict = super().provide_options()
        option_dict["time"] = self.time_slider.value

        return option_dict
    
    def add_time_value(self, value: float):
        self.time_slider.options = self.time_slider.options + [round(value, 8)]
        if self.display_last.value:
            self.time_slider.value = round(value, 8)
