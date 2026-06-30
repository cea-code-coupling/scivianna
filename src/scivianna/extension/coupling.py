
from pathlib import Path

from bokeh.plotting import curdoc

import panel as pn
import panel_material_ui as pmui
import param
from typing import TYPE_CHECKING

import scivianna
from scivianna.extension.extension import Extension
from scivianna.plotter_2d.generic_plotter import Plotter2D
from scivianna.slave import ComputeSlave

if TYPE_CHECKING:
    from scivianna.panel.visualisation_panel import VisualizationPanel
    from scivianna.layout.generic_layout import GenericLayout


class CouplingExtension(Extension, param.Parameterized):
    """Extension to start coupling simulations."""
    show_play_button = param.Boolean(default=False, doc="Show/hide the play button")

    def __init__(
        self,
        layout: "GenericLayout",
        slave: ComputeSlave,
        plotter: Plotter2D,
        panel: "VisualizationPanel",
        **params
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
        param.Parameterized.__init__(self, **params)
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
        self.start_description = pmui.Typography("Start/pause real time display", visible = False)
        self.run_button = pn.widgets.ButtonIcon(
            icon=(Path(scivianna.__file__).parent / "icon" / "player-play.svg").read_text().strip(),
            description="Start automatic update",
            height=30,
            width=30,
            align="center",
            visible = False
        )
        self.display_last = pmui.Checkbox(
            label = "Display last time",
            value = True,
            width = 280,
            visible = False
        )
        self.time_slider = pmui.DiscreteSlider(
            label = "Time selector",
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

        if pn.state.curdoc is None:
            self.curdoc = curdoc()
            pn.state.curdoc = self.curdoc

        self.layout_param_card = pn.Column(
            self.run_button,
            self.display_last,
            self.time_slider,
            width=300,
            margin=0,
        )

        # Link button visibility to the parameter
        self.param.watch(self._update_button_visibility, "show_play_button")

    def _update_button_visibility(self, event):
        """Update button visibility when the parameter changes."""
        self.run_button.visible = event.new
        self.display_last.visible = event.new

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
                self.start_description,
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
                self.layout.recompute_all()

    def provide_options(self) -> dict[str, any]:
        """Provide the option extension option dictionnary

        Returns
        -------
        Dict[str, Any]
            Extension option dictionnary
        """
        option_dict = super().provide_options()
        option_dict["time"] = self.time_slider.value

        return option_dict

    def add_time_value(self, value: float):
        """Adds a new time value to the slide widget

        Parameters
        ----------
        value : float
            Time value to add
        """
        if not round(value, 8) in self.time_slider.options:
            self.time_slider.options = self.time_slider.options + [round(value, 8)]
        if self.display_last.value:
            self.time_slider.value = round(value, 8)

    def to_json(self) -> dict:
        """Returns a dictionary with the information required to rebuild the extension.
        The show_play_button is willingly not saved as it doesn't make sense to restore it.

        Returns
        -------
        dict
            Information dictionary
        """
        return {
            "time_slider_options": self.time_slider.options,
            "time_slider_value": self.time_slider.value,
            "display_last_value": self.display_last.value
        }

    @classmethod
    def from_json(cls, extension: "CouplingExtension", info_dict: dict) -> "CouplingExtension":
        """Restores the extension from its information dict.
        The show_play_button is willingly not saved as it doesn't make sense to restore it.

        Parameters
        ----------
        extension : CouplingExtension
            Extension instance to restore
        info_dict : dict
            Dictionary containing extension state information

        Returns
        -------
        CouplingExtension
            Restored extension
        """
        extension._restoring = True
        try:
            if "time_slider_options" in info_dict:
                extension.time_slider.options = info_dict["time_slider_options"]
            if "time_slider_value" in info_dict:
                extension.time_slider.value = info_dict["time_slider_value"]
            if "display_last_value" in info_dict:
                extension.display_last.value = info_dict["display_last_value"]
        finally:
            extension._restoring = False
        return extension
