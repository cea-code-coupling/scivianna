from typing import Callable, Dict, List, Tuple, Type, Union
import panel as pn
import pandas as pd

from scivianna.data.data1d import Data1D
from scivianna.enums import UpdateEvent
from scivianna.extension.extension import Extension
from scivianna.extension.line_selector import LineSelector
from scivianna.panel.visualisation_panel import VisualizationPanel
from scivianna.plotter_1d.bokeh_1d_plotter import BokehPlotter1D
from scivianna.plotter_1d.generic_plotter import Plotter1D
from scivianna.slave import ComputeSlave
import scivianna.utils


default_extensions = [LineSelector]

class Panel1D(VisualizationPanel):
    """Visualisation panel associated to a code."""

    plotter: Plotter1D
    """ 1D plotter displaying and updating the graph
    """
    update_event: Union[UpdateEvent, List[UpdateEvent]] = UpdateEvent.RECOMPUTE
    """ On what event does the panel recompute itself
    """
    position: Tuple[float, float, float] = None
    """Position where request the plot"""
    cell_id: str = None
    """cell ID where request the plot"""

    def __init__(
            self,
            slave: ComputeSlave,
            name: str = "",
            extensions: List[Extension] = default_extensions
        ):
        """Visualization panel constructor

        Parameters
        ----------
        slave : ComputeSlave
            Slave used to take the information from.
        name : str
            Name of the panel.
        """
        self.plotter = BokehPlotter1D()

        super().__init__(slave, name, extensions.copy())

        self.copy_index = 1

        self.fields = slave.get_labels()

        self.__data_to_update: bool = False
        """Is it required to update the data, can be set on periodic event or on clic"""
        self.field_change_callback: Callable = None
        """Function to call when the field is changed"""

        self.current_time = 0.0
        self.current_value = None

        self.series: Dict[str, pd.Series] = {}

        def field_changed(event):
            """Function called on field changed

            Parameters
            ----------
            event : Any
                Field changed trigering event
            """
            if self.field_change_callback is not None and\
                    len(self.fields_list) > 0:
                self.field_change_callback(self.fields_list)
            self.recompute(event)

        self.fields_list = self.slave.get_labels()
        self.visible_fields_list = [self.slave.get_labels()[0]]
        self.visible_series_list = []
        self.__new_data = {}

        self.recompute()

        self.periodic_recompute_added = False

    @pn.io.hold()
    def async_update_data(
        self,
    ):
        """Update the figures and buttons based on what was added in self.__new_data. This function is called between two servers ticks to prevent multi-users collisions."""
        if "field_names" in self.__new_data:
            self.visible_fields_list = self.__new_data["field_names"]
            self.__new_data = {}
            self.__get_series(self.visible_fields_list)
            self.__data_to_update = True

        if self.__data_to_update:
            for key in self.series:
                if self.series[key] is not None:
                    # Can be None if the async happens at the same time as the next recompute
                    self.plotter.update_plot(key, self.series[key])

            self.plotter.set_visible(self.visible_series_list)

            self.__data_to_update = False

        # this is necessary only in a notebook context where sometimes we have to force Panel/Bokeh to push an update to the browser
        pn.io.push_notebook(self.figure)

    def recompute(self, *args, **kwargs):
        """Recomputes the figure based on the new bounds and parameters.

        Parameters
        ----------
        event : Any
            Event to make the function linkable to a button
        """
        if len(self.visible_fields_list) > 0:
            self.__data_to_update = True

            self.series.clear()
            self.__get_series(self.visible_fields_list)

            if pn.state.curdoc is not None:
                pn.state.curdoc.add_next_tick_callback(self.async_update_data)
            elif scivianna.utils._testing:
                self.async_update_data()

    def __get_series(self, key: str):
        """Get the serie or series associated to the given key

        Parameters
        ----------
        key : str
            Field to request to the slave
        """
        options = {}
        if self.panel_coupling_extension is not None:
            coupling_options = self.panel_coupling_extension.provide_options()
            for k, value in coupling_options.items():
                options[k] = value

        if isinstance(key, (str, int)):
            keys = [key]
        else:
            keys = key.copy()

        self.visible_series_list.clear()

        for key in keys:
            series = self.slave.get_1D_value(
                position = self.position,
                cell_index = self.cell_id,
                material_name = None,
                field = key,
                options = options
            )

            if isinstance(series, list):
                for serie in series:
                    self.series[serie.name] = serie
                    self.visible_series_list.append(serie.name)
            else:
                self.series[series.name] = series
                self.visible_series_list.append(series.name)

    def duplicate(self, keep_name: bool = False) -> "Panel1D":
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
        new_visualiser = Panel1D(
            self.panel_name,
            extensions=[e for e in self.extension_classes]
        )
        new_visualiser.copy_index = self.copy_index

        return new_visualiser

    def get_slave(
        self,
    ):
        return self.slave

    def recompute_at(self, position: Tuple[float, float, float], cell_id: str):
        """Triggers a panel recomputation at the provided location. Called by layout update event.

        Parameters
        ----------
        position : Tuple[float, float, float]
            Location to provide to the slave
        cell_id : str
            cell id to provide to the slave
        """
        self.position = position
        self.cell_id = cell_id
        self.recompute()

    def set_field(self, field_names: List[str]):
        """Updates the plotted fields

        Parameters
        ----------
        field_name : List[str]
            Fields to display
        allow_wrong_name : bool
            Accept a wrong field (nothing happens)
        """
        fields: List[str] = []
        self.fields_list = self.slave.get_labels()

        if isinstance(field_names, list):
            for field_name in field_names:
                if field_name not in self.fields_list:
                    continue
                else:
                    fields.append(field_name)
        else:
            if field_names not in self.fields_list:
                pass
            else:
                fields.append(field_names)

        if fields != [] and set(fields) != set(self.visible_fields_list):
            self.__new_data["field_names"] = fields

            self.__data_to_update = True
            if pn.state.curdoc is not None:
                pn.state.curdoc.add_next_tick_callback(self.async_update_data)
            elif scivianna.utils._testing:
                self.async_update_data()
            else:
                # A set field was probably called before opening the gui
                self.visible_fields_list = fields

    def provide_field_change_callback(self, callback: Callable):
        """Stores a function to call everytime the displayed field is changed.
        the functions takes a string as argument.

        Parameters
        ----------
        callback : Callable
            Function to call.
        """
        self.field_change_callback = callback

    def provide_on_mouse_move_callback(self, callback: Callable):
        """Stores a function to call everytime the user moves the mouse on the plot.
        Functions arguments are location, cell_id.

        Parameters
        ----------
        callback : Callable
            Function to call.
        """
        pass

    def provide_on_clic_callback(self, callback: Callable):
        """Stores a function to call everytime the user clics on the plot.
        Functions arguments are location, cell_id.

        Parameters
        ----------
        callback : Callable
            Function to call.
        """
        pass

    def set_colormap(self, colormap):
        """Sets the current color map, not used in the case of a Panel1D

        Parameters
        ----------
        colormap : str
            Color map name
        """
        pass

    def to_json(self) -> Dict:
        """Returns a dictionnary with the information required to rebuild the visualization panel

        Returns
        -------
        Dict
            Information dict
        """
        return {
            "name": self.panel_name,
            "position": self.position,
            "cell_id": self.cell_id,
            "field_names": self.fields_list,
            "visible_field_names": self.visible_fields_list,
            "serie_names": self.visible_series_list,
            "sync_field": self.sync_field,
            "update_event": self.update_event,
        }

    @classmethod
    def from_json(
        cls,
        info_dict: Dict,
        slave: ComputeSlave,
        extensions: Union[List[Extension], List[Tuple[Type[Extension], dict]]] = []
    ) -> "Panel1D":
        """Restores the visualization panel from its information dict

        Parameters
        ----------
        info_dict : Dict
            Dictionnary containing all required information to restore the panel
        slave : ComputeSlave
            Panel associated slave
        extensions : Union[List[Extension], List[Tuple[Type[Extension], dict]]]
            GUI extensions, can be extension classes or tuples of (class, state_dict)

        Returns
        -------
        Panel1D
            Restored panel
        """
        panel = Panel1D(
            slave,
            info_dict["name"],
            extensions
        )
        panel.position = info_dict["position"]
        panel.cell_id = info_dict["cell_id"]
        panel.set_field(info_dict["visible_field_names"])
        panel.sync_field = info_dict["sync_field"]
        panel.update_event = info_dict["update_event"]
        return panel
