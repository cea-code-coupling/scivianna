
from pathlib import Path

import panel as pn
from typing import TYPE_CHECKING

import scivianna
from scivianna.extension.extension import Extension
from scivianna.plotter_2d.generic_plotter import Plotter2D
from scivianna.slave import ComputeSlave
from scivianna.utils.interface_tools import GenericInterfaceEnum

if TYPE_CHECKING:
    from scivianna.panel.visualisation_panel import VisualizationPanel
    from scivianna.layout.generic_layout import GenericLayout


class LayoutExtension(Extension):
    """Extension to load files and send them to the slave."""

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
        layout : GenericLayout
            Extension associated layout
        slave : ComputeSlave
            Slave computing the displayed data
        plotter : Plotter2D
            Figure plotter
        panel : VisualizationPanel
            Panel to which the extension is attached
        """
        super().__init__(
            "Interface layout",
            "dashboard",
            slave,
            plotter,
            panel,
        )

        self.layout = layout

        self.description = """
This extension allows you to split panels to visualize several fields/geometries at the same time.
"""
        interface_options = [
            val.value if isinstance(val, GenericInterfaceEnum) else str(val)
            for val in self.layout.available_interfaces.keys()
        ]

        self.interface_selector = pn.widgets.Select(
            label="Code",
            options=interface_options,
            value=interface_options[
                list(self.layout.available_interfaces.values())
                    .index(self.layout.visualisation_panels[self.layout.current_frame].slave.code_interface)
            ],
            width = 280,
            margin = 0
        )

        """
            Current edited frame selector
        """
        self.frame_selector = pn.widgets.Select(
            label="Visualizer selector",
            options=list(self.layout.visualisation_panels.keys()),
            value=list(self.layout.visualisation_panels.keys())[0],
            width = 280,
            margin = 0
        )

        self.interface_selector.param.watch(self.layout.change_code_interface, "value")
        self.frame_selector.param.watch(self.set_to_frame, "value")

        self.duplicate_horizontally_button = pn.widgets.ButtonIcon(
            icon=(Path(scivianna.__file__).parent / "icon" / "columns-2.svg").read_text().strip(), description="Duplicate horizontally", height=30, width=30
        )
        self.duplicate_vertitally_button = pn.widgets.ButtonIcon(
            icon=(Path(scivianna.__file__).parent / "icon" / "layout-rows.svg").read_text().strip(), description="Duplicate vertically", height=30, width=30
        )
        self.split_new_horizontally_button = pn.widgets.ButtonIcon(
            icon=(Path(scivianna.__file__).parent / "icon" / "column-insert-right.svg").read_text().strip(),
            description="Split horizontally",
            height=30,
            width=30,
        )
        self.split_new_vertically_button = pn.widgets.ButtonIcon(
            icon=(Path(scivianna.__file__).parent / "icon" / "row-insert-bottom.svg").read_text().strip(),
            description="Split vertically",
            height=30,
            width=30,
        )

        def duplicate_vertically(event: bool):
            """Split the panel vertically, the new panel is a copy of the first

            Parameters
            ----------
            event : bool
                If the call is from a button press or release
            """
            if event:
                self.layout.duplicate(True)

        def duplicate_hozironally(event: bool):
            """Split the panel hozironally, the new panel is a copy of the first

            Parameters
            ----------
            event : bool
                If the call is from a button press or release
            """
            if event:
                self.layout.duplicate(False)

        self.duplicate_horizontally_button.on_click(duplicate_hozironally)
        self.duplicate_vertitally_button.on_click(duplicate_vertically)

        self.layout_param_card = pn.Column(
            self.frame_selector,
            self.interface_selector,
            # pn.Row(
            #     self.duplicate_horizontally_button, self.duplicate_vertitally_button
            # ),
            width=300,
            margin=0,
        )

    def add_widget(self, widget: pn.widgets.Widget):
        self.layout_param_card.append(widget)


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
                self.layout_param_card,
                margin=(0, 0, 10, 10),
            )

    def change_code_interface(self, *args, **kwargs):
        self.layout.change_code_interface()

    def set_to_frame(self, *args, **kwargs):
        self.layout.set_to_frame(self.frame_selector.value)

    @pn.io.hold()
    def change_to_frame(self, frame_name: str):
        self.frame_selector.options = list(self.layout.visualisation_panels.keys())

        if frame_name != self.frame_selector.value:
            frame_code_enum = list(self.layout.available_interfaces.keys())[
                list(self.layout.available_interfaces.values()).index(
                    self.layout.visualisation_panels[frame_name].slave.code_interface
                )
            ]

            self.interface_selector.value = (
                frame_code_enum.value
                if isinstance(frame_code_enum, GenericInterfaceEnum)
                else str(frame_code_enum)
            )

            self.frame_selector.value = frame_name

