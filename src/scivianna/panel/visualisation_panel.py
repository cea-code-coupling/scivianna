from pathlib import Path
from typing import Callable, Dict, List, Tuple, Type, Union
import panel as pn
import panel_material_ui as pmui

import scivianna
from scivianna.component.overlay_component import Overlay
from scivianna.data.data_container import DataContainer
from scivianna.extension.extension import Extension

from scivianna.interface.generic_interface import Geometry2D

from scivianna.enums import UpdateEvent
from scivianna.slave import ComputeSlave

from scivianna.panel.gui import GUI
from scivianna.plotter_2d.generic_plotter import Plotter2D

pn.config.inline = True

class VisualizationPanel(pn.viewable.Viewer):
    """Visualisation panel associated to a code."""

    panel_name: str
    """ Panel name
    """
    slave: ComputeSlave
    """ Slave to which request the plots
    """
    plotter: Plotter2D
    """ 2D plotter displaying and updating the graph
    """
    main_frame: pmui.Container
    """ Main frame displaying the geometry.
    """
    extensions: List[Extension]
    """ List of extensions attached to the panel
    """
    figure: Overlay
    """Figure in its overlay"""

    current_data: DataContainer
    """ Displayed data and their properties.
    """
    update_event: Union[UpdateEvent, List[UpdateEvent]] = UpdateEvent.RECOMPUTE
    """ On what event does the panel recompute itself
    """
    sync_field: bool = False
    """ The panel updates its field if another panel did
    """

    def __init__(
            self,
            slave: ComputeSlave,
            name="",
            extensions: Union[List[Extension], List[Tuple[Type[Extension], dict]]] = []
        ):
        """Visualization panel constructor

        Parameters
        ----------
        slave : ComputeSlave
            ComputeSlave object to which request the plots.
        name : str
            Name of the panel.
        extensions : Union[List[Extension], List[Tuple[Type[Extension], dict]]]
            List of extensions loaded with the visualizer. Can be either extension classes or tuples of (class, state_dict).
        """
        #
        #   Initializing attributes
        #
        super().__init__()

        self.panel_coupling_extension: Extension = None
        self.panel_name = name
        self.copy_index = 1

        self.slave = slave
        self.slave.allow_errors = True

        self.update_polygons = False
        """Need to update the data at the next async call"""
        self.field_change_callback: Callable = None
        """Function to call when the field is changed"""

        self.__data_to_update: bool = False
        self.__new_data = {}
        """New data to set in the colorbar and in the columndatasources"""

        #
        #   Saving code interface properties
        #
        code_interface: Type[Geometry2D] = self.slave.code_interface

        #
        #   Extensions creation
        #
        self.extension_classes = []
        extension_states = {}

        # Process extensions - can be classes or (class, state) tuples
        for ext in extensions:
            if isinstance(ext, tuple):
                ext_class, ext_state = ext
                self.extension_classes.append(ext_class)
                extension_states[ext_class.__name__] = ext_state
            else:
                self.extension_classes.append(ext)

        # Add interface extensions
        for extension in code_interface.extensions:
            if not issubclass(extension, Extension):
                raise TypeError(f"Extension {extension} declared in {code_interface.extensions} extensions is not a subclass of {Extension}")
            if not extension in self.extension_classes:
                self.extension_classes.append(extension)

        #
        #   Building layout
        #
        self.extensions = []
        for e in self.extension_classes:
            ext = e(self.slave, self.plotter, self)
            # Apply saved state if available
            if e.__name__ in extension_states:
                e.from_json(ext, extension_states[e.__name__])
            self.extensions.append(ext)

        self.gui = GUI(
            self.extensions
        )
        self.gui_panel = self.gui.make_panel()

        self.button = pmui.IconButton(icon=(Path(scivianna.__file__).parent / "icon" / "settings_applications.svg").read_text().strip(), icon_size = "1em", margin=0)
        self.title_typo = pmui.Typography("**"+self.panel_name+"**", margin=0)

        self.figure = Overlay(
            figure = self.plotter.make_panel(),
            button = self.button,
            title = self.title_typo,
            sizing_mode="stretch_both",
            styles={"border": "2px solid lightgray"}
        )
        self.figure.hide_buttons()

        pn.io.push_notebook(self.figure)


        self.periodic_recompute_added = False
        """Coupling periodic update"""
        self.marked_to_recompute = False
        """Recompute requested by a coordinates/field change on API side"""

        for extension in self.extensions:
            self.provide_on_clic_callback(extension.on_mouse_clic)
            self.provide_on_mouse_move_callback(extension.on_mouse_move)

    def duplicate(self, keep_name: bool = False) -> "VisualizationPanel":
        """Get a copy of the panel. A panel of the same type is generated, the current display too, but a new slave process is created.

        Parameters
        ----------
        keep_name : bool
            New panel name is the same as the current, if not, a number iterates at the end of the name

        Returns
        -------
        VisualizationPanel
            Copy of the visualisation panel
        """
        raise NotImplementedError()

    def get_slave(
        self,
    ) -> ComputeSlave:
        """Returns the current panel code slave

        Returns
        -------
        ComputeSlave
            Panel slave
        """
        return self.slave


    #
    # #
    # #     API to provide in the panels
    # #
    #
    def recompute(
        self, *args, **kwargs
    ):
        """Recomputes the figure based on the new bounds and parameters.
        """
        raise NotImplementedError()

    def provide_on_mouse_move_callback(self, callback: Callable):
        """Stores a function to call everytime the user moves the mouse on the plot.
        Functions arguments are location, cell_id.

        Parameters
        ----------
        callback : Callable
            Function to call.
        """
        self.plotter.provide_on_mouse_move_callback(callback)

    def provide_on_clic_callback(self, callback: Callable):
        """Stores a function to call everytime the user clics on the plot.
        Functions arguments are location, cell_id.

        Parameters
        ----------
        callback : Callable
            Function to call.
        """
        self.plotter.provide_on_clic_callback(callback)

    def provide_field_change_callback(self, callback: Callable):
        """Stores a function to call everytime the displayed field is changed.
        the functions takes a string as argument.

        Parameters
        ----------
        callback : Callable
            Function to call.
        """
        self.field_change_callback = callback

    def recompute_at(self, position: Tuple[float, float, float], cell_id: str):
        """Triggers a panel recomputation at the provided location. Called by layout update event.

        Parameters
        ----------
        position : Tuple[float, float, float]
            Location to provide to the slave
        cell_id : str
            cell id to provide to the slave
        """
        raise NotImplementedError()

    def set_field(self, field_name: str):
        """Updates the plotted field

        Parameters
        ----------
        field_name : str
            New field to display
        """
        raise NotImplementedError()

    def set_colormap(self, colormap: str):
        """Sets the current color map

        Parameters
        ----------
        colormap : str
            Color map name
        """
        raise NotImplementedError()

    def outline_color(self, color: str = "lightgray"):
        """Sets the color of the outlined plot

        Parameters
        ----------
        color : str
            HTML color
        """
        if color is None:
            self.figure.styles = {"border": "0px solid lightgray"}
        else:
            self.figure.styles = {"border": f"2px solid {color}"}

    def __panel__(self,):
        return pn.Row(self.gui_panel, self.figure, margin=0, sizing_mode="stretch_both")

    def rename(self, name: str):
        """Rename current panel

        Parameters
        ----------
        name : str
            New name
        """
        self.panel_name = name
        self.title_typo.object = f"## {name}"

    def get_new_name(self) -> str:
        """Returns an new name iterating on the current one

        Returns
        -------
        str
            Name different from current name
        """
        if self.panel_name.endswith(f" - {self.copy_index}"):
            new_name = self.panel_name.replace(
                f" - {self.copy_index}", f" - {self.copy_index + 1}"
            )
        else:
            new_name = f"{self.panel_name} - {self.copy_index + 1}"

        self.copy_index += 1

        return new_name

    def trigger_on_file_load(self, file_path: str, file_label: str):
        for extension in self.extensions:
            extension.on_file_load(file_path, file_label)

    def to_json(self) -> Dict:
        """Returns a dictionnary with the information required to rebuild the visualization panel

        Returns
        -------
        Dict
            Information dict
        """
        raise NotImplementedError()

    @classmethod
    def from_json(self, info_dict: Dict, data: DataContainer) -> "VisualizationPanel":
        """Restores the visualization panel from its information dict

        Parameters
        ----------
        info_dict : Dict
            Dictionnary containing all required information to restore the panel

        Returns
        -------
        VisualizationPanel
            Dict defined visualization panel
        """
        raise NotImplementedError()
