import functools 
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Type, Union

import panel as pn
import panel_material_ui as pmui

import scivianna
import scivianna.utils
from scivianna.enums import UpdateEvent
from scivianna.extension.layout import LayoutExtension
from scivianna.extension.coupling import CouplingExtension
import scivianna.interface
from scivianna.interface.generic_interface import GenericInterface
from scivianna.panel.gui import GUI
from scivianna.slave import ComputeSlave
from scivianna.panel.visualisation_panel import VisualizationPanel
from scivianna.utils.interface_tools import (
    GenericInterfaceEnum,
    get_interface_default_panel,
)


card_style = {}


class GenericLayout:
    """Displayable that lets arranging several VisualizationPanel"""

    main_frame: pn.Column
    """ Main frame : gridstack of different VisualizationPanel main_frame
    """

    available_interfaces: Dict[Union[str, GenericInterfaceEnum], Type[GenericInterface]]
    """ Available interface classes to switch from one to another
    """
    load_available_interfaces: Callable = None
    """ Function loading available interfaces. Can be overwriten to add additional interfaces.
    """

    visualisation_panels: Dict[str, VisualizationPanel]
    """ Dictionnary containing all visualisation panels.
    """

    def __init__(
        self,
        visualisation_panels: Dict[str, VisualizationPanel],
        additional_interfaces: Dict[
            Union[str, GenericInterfaceEnum], Type[GenericInterface]
        ] = {},
        add_run_button: bool = False,
    ):
        self.visualisation_panels = visualisation_panels

        if self.load_available_interfaces is None:
            self.available_interfaces = scivianna.interface.INTERFACES
        else:
            self.available_interfaces = self.load_available_interfaces()

        for interface in additional_interfaces:
            if not issubclass(additional_interfaces[interface], GenericInterface):
                raise TypeError(
                    f"Provided interface {interface} is not a GenericInterface, found type {type(additional_interfaces[interface])}"
                )
            self.available_interfaces[interface] = additional_interfaces[interface]

        self.code_interface_to_update = True

        """
            ButtonIcon to split the frames
            List of available icons : https://tabler.io/icons

            border-vertical / border-horizontal
            box-align-left / box-align-right
            columns-2 / column-insert-right
            layout-rows / row-insert-bottom
        """

        #   Adding a play button at the beginning of the side bar
        #   It will trigger a periodic task to update the plot in case of code coupling simulation
        self.periodic_recompute_added = False
        self.current_frame = list(self.visualisation_panels.keys())[0]

        self.panels_to_recompute: List[str] = []

        self.layout_extension = LayoutExtension(add_run_button, self, None, None, None)

        self.button_columns: Dict[str, List[pmui.IconButton]] = {}
        self.side_bars: Dict[str, pn.Column] = {}

        self.gui = GUI([self.layout_extension])

        self.gui.drawer.param.watch(self.outline_panels, "open")
        self.gui.drawer.param.watch(self.toggle_outlines, "open")
        
        for panel in self.visualisation_panels.values():
            self.register_panel(panel)

        self.change_current_frame()
        self.toggle_outlines()

        self.last_hover_id = None
        """Last hovered cell to trigger change if applicable"""

    @pn.io.hold()
    def register_panel(self, panel: VisualizationPanel):
        if isinstance(panel, VisualizationPanel):
            panel.provide_on_clic_callback(self.on_clic_callback)
            panel.provide_on_mouse_move_callback(self.mouse_move_callback)
            panel.provide_field_change_callback(self.field_change_callback)

            # 2: assuming the two first are the button to open and close the tab
            self.button_columns[panel.panel_name] = [b[0] for b in panel.gui.buttons]
            self.side_bars[panel.panel_name] = [b[1] for b in panel.gui.buttons]
            
            for button, side_bar in zip(self.button_columns[panel.panel_name], self.side_bars[panel.panel_name]):
                self.gui.register_new_extension(button, side_bar)

            panel.gui_panel.visible = False

            panel.figure.button.on_click(functools.partial(self.button_change_to_frame, frame_name=panel.panel_name))
        else:
            raise ValueError(f"Tried registering {panel}, only VisualizationPanel instances are accepted.")
        
    @pn.io.hold()
    def change_code_interface(self, event):
        """Replaces the panel to one linked to the code interface

        Parameters
        ----------
        event : Any
            Event to make the function linkable to the gridstack
        """
        current_frame = self.current_frame
        code_interface = self.layout_extension.interface_selector.value
        
        interface_key = list(self.available_interfaces.keys())[
            [
                val.value if isinstance(val, GenericInterfaceEnum) else str(val)
                for val in self.available_interfaces.keys()
            ].index(code_interface)
        ]

        if (
            self.code_interface_to_update
            and self.available_interfaces[interface_key] != self.visualisation_panels[current_frame].slave.code_interface
        ):
            print(
                f"Updating code interface of panel {current_frame} to {code_interface}"
            )

            default_panel = get_interface_default_panel(
                interface_key, title=current_frame
            )

            if default_panel is None:
                # Means the panel is custom and was provided by the user
                new_slave = ComputeSlave(self.available_interfaces[interface_key])

                self.visualisation_panels[current_frame] = VisualizationPanel(
                    slave=new_slave, name=current_frame
                )
            else:
                self.visualisation_panels[current_frame] = default_panel

            self.register_panel(self.visualisation_panels[current_frame])

    @pn.io.hold()
    def change_current_frame(self, *args, **kwargs):
        """Swap the the active panel

        Parameters
        ----------
        event : Any
            Event to make the function linkable to the gridstack
        """
        print("Changing to ", self.current_frame)
        self.layout_extension.change_to_frame(self.current_frame)

        for key in self.side_bars:
            for b in self.button_columns[key]:
                b.visible = key == self.current_frame
        
        self.gui.change_drawer(None, self.gui.active_extension)
        self.outline_panels()

    @pn.io.hold()
    def set_to_frame(self, frame_name: str):
        """Updates the Select widget to the active panel

        Parameters
        ----------
        frame_name : str
            Name of the active panel

        Raises
        ------
        ValueError
            The provided name is not in the Select options
        """
        if frame_name in self.visualisation_panels:
            self.current_frame = frame_name
            self.code_interface_to_update = False

            self.layout_extension.change_to_frame(self.current_frame)
            self.change_current_frame()

            self.code_interface_to_update = True
        else:
            raise ValueError(
                f"Frame {frame_name} not in options, available keys : {list(self.visualisation_panels.keys())}"
            )
        
    def outline_panels(self, *args, **kwargs):
        """Updates the figures outline colors
        """
        colorize = self.gui.drawer.open
        for frame in self.visualisation_panels:
            if colorize:
                self.visualisation_panels[frame].outline_color(
                    "var(--design-primary-color, var(--panel-primary-color))" 
                    if (frame == self.current_frame) 
                    else "lightgray")
            else:
                self.visualisation_panels[frame].outline_color(None)
                

    def duplicate(self, horizontal: bool):
        """Split the panel, the new panel is a copy of the first, all panels are duplicated.

        Parameters
        ----------
        horizontal : bool
            Whether the cut should be horizontal or vertical
        """
        raise NotImplementedError()

    def get_panel(self, panel_name: str) -> VisualizationPanel:
        """Returns the VisualizationPanel associated to the given name

        Parameters
        ----------
        panel_name : str
            Name of the panel

        Returns
        -------
        VisualizationPanel
            Requested panel

        Raises
        ------
        ValueError
            No panel at the given name
        """
        if panel_name not in self.visualisation_panels:
            raise ValueError(
                f"Unknown panel requested: {panel_name}, available panels: {list(self.visualisation_panels.keys())}. Make sure the key requested by the exchanger is defined as panel_name@field_name"
            )

        return self.visualisation_panels[panel_name]

    def set_panel(self, panel_name: str, panel: VisualizationPanel):
        """Updates the VisualizationPanel associated to the given name

        Parameters
        ----------
        panel_name : str
            Name of the panel
        panel:VisualizationPanel
            New panel value
        """
        self.visualisation_panels[panel_name] = panel

    def stop_periodic_update(
        self,
    ):
        """Stops the curdoc a periodic task (every 100 ms) to automatically update the plots."""
        self.periodic_recompute_added = False
        pn.state.curdoc.add_timeout_callback(self.recompute, 100)

    def add_periodic_update(
        self,
    ):
        """Add to the curdoc a periodic task (every 100 ms) to automatically update the plots."""
        self.periodic_recompute_added = True
        pn.state.curdoc.add_timeout_callback(self.recompute, 100)

    def recompute(
        self,
    ):
        """Periodically called function that requests calling async_update_data at the end of current tick."""
        if pn.state.curdoc is not None:
            pn.state.curdoc.add_next_tick_callback(self.async_update_data)
        elif scivianna.utils._testing:
            self.async_update_data()

    @pn.io.hold()
    async def async_update_data(
        self,
    ):
        """Request all panels to update themselves. This function being called between two ticks, it will not trigger collisions between automatic and user update requests."""
        for panel in self.panels_to_recompute:
            self.visualisation_panels[panel].recompute()
            self.visualisation_panels[panel].async_update_data()
        self.panels_to_recompute.clear()
        if self.periodic_recompute_added:
            self.add_periodic_update()

    def mark_to_recompute(self, panels_to_recompute):
        self.panels_to_recompute = panels_to_recompute

    def _make_button_icon(self,) -> pn.widgets.ButtonIcon:
        """Makes a button icon to switch to current panel

        Returns
        -------
        pn.widgets.ButtonIcon
            ButtonIcon
        """
        return pn.widgets.ButtonIcon(
            size="2.5em",
            icon=(Path(scivianna.__file__).parent / "icon" / "layout-sidebar.svg").read_text().strip(),
            visible=False,
            description="Change side bar and coordinate bar to current plot."
        )

    def on_clic_callback(self, 
        screen_location: Tuple[float, float],
        space_location: Tuple[float, float, float],
        cell_id: Union[str, int],
    ):
        """Function called when the mouse moves on the plot

        Parameters
        ----------
        screen_location : Tuple[float, float]
            Mouse location on screen
        space_location : Tuple[float, float, float]
            Mouse targeted location in space
        cell_id : Union[str, int]
            Currently hovered cell
        """
        for panel in self.visualisation_panels.values():
            if panel.update_event == UpdateEvent.CLIC or (isinstance(panel.update_event, list) and UpdateEvent.CLIC in panel.update_event):
                panel.recompute_at(space_location, cell_id)

    def mouse_move_callback(self, 
        screen_location: Tuple[float, float],
        space_location: Tuple[float, float, float],
        cell_id: Union[str, int],
    ):
        """Function called when the mouse moves on the plot

        Parameters
        ----------
        screen_location : Tuple[float, float]
            Mouse location on screen
        space_location : Tuple[float, float, float]
            Mouse targeted location in space
        cell_id : Union[str, int]
            Currently hovered cell
        """
        for panel in self.visualisation_panels.values():
            if panel.update_event == UpdateEvent.MOUSE_POSITION_CHANGE or (isinstance(panel.update_event, list) and UpdateEvent.MOUSE_POSITION_CHANGE in panel.update_event):
                panel.recompute_at(space_location, cell_id)

            if cell_id != self.last_hover_id and\
                    (
                        panel.update_event == UpdateEvent.MOUSE_CELL_CHANGE
                        or (isinstance(panel.update_event, list) and UpdateEvent.MOUSE_CELL_CHANGE in panel.update_event)
                    ):
                self.last_hover_id = cell_id
                panel.recompute_at(space_location, cell_id)

    def field_change_callback(self, new_field: str):
        """Function calling panels update a field change

        Parameters
        ----------
        new_field : str
            New field to set
        """
        for panel in self.visualisation_panels.values():
            if panel.sync_field:
                panel.set_field(new_field)

    def add_time_widget(self,):
        """Adds a time management widget to the layout
        """
        self.time_widget = CouplingExtension(self, None, None, None)

        self.gui.add_extension(self.time_widget)

    @pn.io.hold()
    def toggle_outlines(self, *args, **kwargs):
        """Function called when the sidebar is opened or closed, edits the frames visibility.
        """
        for panel in self.visualisation_panels.values():
            if self.gui.drawer.open:
                panel.figure.show_buttons()
            else:
                panel.figure.hide_buttons()

    def button_change_to_frame(self, event, frame_name: str):
        """Triggers a frame change from clicking on a button
        
        Parameters
        ----------
        event : Any
            Bokeh triggering event
        frame_name : str
            Frame to change to
        """
        # Setting the active extension to the layout which is the first in buttons as added first
        self.gui.active_extension = self.gui.buttons[0]

        self.set_to_frame(frame_name)

    def __panel__(self,):
        """Returns the panel to display
        """
        return self.main_frame
    
    def show(self, *args, **kwargs):
        return self.main_frame.show(*args, **kwargs)

    def servable(self, *args, **kwargs):
        return self.main_frame.servable(*args, **kwargs)