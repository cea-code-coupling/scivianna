from logging import warning
from typing import Callable, Dict, List, Tuple, Type, Union
import numpy as np
import panel as pn
import param
import os

from scivianna.extension.slice_3d import Slice3D
from scivianna.extension.extension import Extension
from scivianna.extension.field_selector import FieldSelector
from scivianna.extension.file_loader import FileLoader
from scivianna.panel.visualisation_panel import VisualizationPanel

from scivianna.data.data3d import Data3D
from scivianna.interface.generic_interface import Geometry3D

from scivianna.enums import VisualizationMode
from scivianna.slave import ComputeSlave

from scivianna.plotter_3d.vtk_3d_plotter import Plotter3D
from scivianna.constants import MESH, X, Y
import scivianna.utils

profile_time = bool(os.environ["VIZ_PROFILE"]) if "VIZ_PROFILE" in os.environ else 0
if profile_time:
    import time

pn.config.inline = True

default_extensions = [FileLoader, FieldSelector, Slice3D]


class Panel3D(VisualizationPanel):
    """2D Visualisation panel associated to a code."""

    plotter: Plotter3D
    """ 2D plotter displaying and updating the graph
    """
    current_data: Data3D
    """ Displayed data and their properties.
    """
    colormap = param.String()

    def __init__(
        self,
        slave: ComputeSlave,
        name="",
        extensions: List[Extension] = default_extensions,
        data: Data3D = None,
        displayed_field: str = MESH,
        colormap: str = "BuRd",
    ):
        """Visualization panel constructor

        Parameters
        ----------
        slave : ComputeSlave
            ComputeSlave object to which request the plots.
        name : str
            Name of the panel.
        display_polygons : bool
            Display as polygons or as a 2D grid.
        extensions : List[Extension]
            List of extensions to add to the gui.
        data : Data2D
            Data2D object with which the panel is initialized.
        displayed_field : str
            Name of the field to display initially, defaults to MESH.
        colormap : str
            Colormap name for coloring the data, defaults to "BuRd".
        u : Tuple[float, float, float]
            Direction vector for the horizontal axis, defaults to X.
        v : Tuple[float, float, float]
            Direction vector for the vertical axis, defaults to Y.
        u_range : Tuple[float, float]
            Range (min, max) for the horizontal axis, defaults to (0., 1.).
        v_range : Tuple[float, float]
            Range (min, max) for the vertical axis, defaults to (0., 1.).
        w_value : float
            Value along the u ^ v (normal) axis, defaults to 0.5.
        """
        code_interface: Type[Geometry3D] = slave.code_interface
        assert issubclass(
            code_interface, Geometry3D
        ), f"A VisualizationPanel can only be given a Geometry3D interface slave, received {code_interface}."

        #
        #   Initializing attributes
        #
        self.update_polygons = False
        """Need to update the data at the next async call"""

        self.field_change_callback: Callable = None
        """Function to call when the field is changed"""

        self.plotter = Plotter3D()

        super().__init__(slave, name, extensions.copy())

        #
        #   First plot on XY basic range
        #
        self.displayed_field = displayed_field
        for extension in self.extensions:
            extension.on_field_change(displayed_field)

        self.colormap = colormap

        if data is None:
            data_ = self.compute_fn()
        else:
            print(f"Panel3D {self.panel_name} initialized initial Data3D object, restoring {len(data.cell_ids)} cells.")
            data_ = data
            self.update_polygons = True
        
        self.plotter.plot(data_)

        self.current_data = data_

        if (
            slave.get_label_coloring_mode(self.displayed_field) == VisualizationMode.FROM_VALUE
        ):
            self.plotter.update_colorbar(
                True,
                (
                    min([float(e) for e in data_.cell_values]),
                    max([float(e) for e in data_.cell_values]),
                ),
            )
        else:
            self.plotter.update_colorbar(False, (None, None))

        self.w_value = 0.
        self.u = X
        self.v = Y
        self.__data_to_update: bool = False
        self.__new_data = {}
        
        for extension in self.extensions:
            extension.on_range_change(
                (self.plotter.get_slice_origin()[0], None), 
                (self.plotter.get_slice_origin()[1], None), 
                self.w_value
            )
            extension.on_frame_change(*self.plotter.get_uv())

        try:
            pn.state.on_session_created(self.recompute)
        except Exception:
            pass

    @pn.io.hold()
    def async_update_data(
        self,
    ):
        """Update the figures and buttons based on what was added in self.__new_data. This function is called between two servers ticks to prevent multi-users collisions."""
        if self.__data_to_update:
            if profile_time:
                st = time.time()

            if "color_mapper" in self.__new_data:
                self.plotter.update_colorbar(
                    True,
                    (
                        self.__new_data["color_mapper"]["new_low"],
                        self.__new_data["color_mapper"]["new_high"],
                    ),
                )
                self.plotter.set_color_map(self.colormap)
            if "data" in self.__new_data:
                self.current_data: Data3D = self.__new_data["data"]

                if not self.update_polygons:
                    self.plotter.update_plot(self.current_data)
                else:
                    self.plotter.plot(self.current_data)

            self.__data_to_update = False

            if profile_time:
                print(f"Async function : {time.time() - st}")

        if "field_name" in self.__new_data:
            if self.marked_to_recompute:
                self.marked_to_recompute = False
                self.async_update_data()
        else:
            # If marked to recompute, a safe change was applied on a plot parameter, a recompute is requested async
            if self.marked_to_recompute:
                if not scivianna.utils._testing:
                    self.recompute()
                    self.marked_to_recompute = False
                    self.async_update_data()
                else:
                    self.marked_to_recompute = False
                    self.recompute()

        # this is necessary only in a notebook context where sometimes we have to force Panel/Bokeh to push an update to the browser
        pn.io.push_notebook(self.figure)

        self.__new_data = {}

    def compute_fn(
        self,
    ) -> Data3D:
        """Request the slave to compute a new frame, and updates the data to display

        Returns
        -------
        Data3D
            Geometry data.
        """
        options = {key: value for options in [
            e.provide_options() for e in self.extensions
        ] for key, value in options.items()}

        if self.panel_coupling_extension is not None:
            coupling_options = self.panel_coupling_extension.provide_options()
            for key, value in coupling_options.items():
                options[key] = value

        computed_data = self.slave.compute_3D_data(
            self.displayed_field,
            options,
        )

        if computed_data is None:
            print(
                f"\n\n Got None from computed data on {self.panel_name}, returning the past values.\n\n"
            )
            return None

        computed_data, polygons_updated = computed_data

        for extension in self.extensions:
            extension.on_updated_data(computed_data)

        return computed_data

    def recompute(
        self, *args, **kwargs
    ):
        """Recomputes the figure based on the new bounds and parameters.
        """
        if profile_time:
            st = time.time()


        data = self.compute_fn()

        if data is not None:
            self.__new_data = {
                "data": data,
            }

            if (
                self.slave.get_label_coloring_mode(
                    self.displayed_field
                ) == VisualizationMode.FROM_VALUE
            ):
                self.__new_data["color_mapper"] = {
                    "new_low": np.nanmin(np.array(data.cell_values).astype(float)),
                    "new_high": np.nanmax(np.array(data.cell_values).astype(float)),
                }
                self.__new_data["hide_colorbar"] = False
            else:
                self.__new_data["hide_colorbar"] = True

            self.__data_to_update = True

            if profile_time:
                print(f"Plot panel preparing data : {time.time() - st}")

            if pn.state.curdoc is not None:
                pn.state.curdoc.add_next_tick_callback(self.async_update_data)
            elif scivianna.utils._testing:
                self.async_update_data()

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
        new_visualiser = Panel3D(
            slave=self.slave.duplicate(),
            name=self.panel_name,
            extensions=[e for e in self.extension_classes]
        )
        new_visualiser.copy_index = self.copy_index

        if isinstance(self.update_event, list):
            new_visualiser.update_event = self.update_event.copy()
        else:
            new_visualiser.update_event = self.update_event

        new_visualiser.set_field(self.displayed_field)
        new_visualiser.set_colormap(self.colormap)

        return new_visualiser

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

    def provide_on_axes_change_callback(self, callback: Callable):
        """Stores a function to call everytime the axes are changed.
        the functions takes a two numpy arrays and three floats (axes, and umin, vmin, w).

        Parameters
        ----------
        callback : Callable
            Function to call.
        """
        self.plotter.provide_on_axes_change_callback(callback)

    def set_field(self, field_name: str):
        """Updates the plotted field

        Parameters
        ----------
        field_name : str
            New field to display
        """
        if self.displayed_field != field_name:
            self.displayed_field = field_name

            if field_name not in self.slave.get_labels():
                warning(f"\n\nRequested field {field_name} : field unavailable, available values : {self.slave.get_labels()}.\n\n")

            else:

                for extension in self.extensions:
                    extension.on_field_change(field_name)

                if pn.state.curdoc is not None:
                    pn.state.curdoc.add_next_tick_callback(self.recompute)
                elif scivianna.utils._testing:
                    self.recompute()

                if self.field_change_callback is not None:
                    self.field_change_callback(field_name)

    def set_colormap(self, colormap: str):
        """Sets the current color map

        Parameters
        ----------
        colormap : str
            Color map name
        """
        if colormap != self.colormap:
            self.colormap = colormap
            self.__data_to_update = True

            if pn.state.curdoc is not None:
                pn.state.curdoc.add_next_tick_callback(self.recompute)
            elif scivianna.utils._testing:
                self.recompute()

    def to_json(self) -> Dict:
        """Returns a dictionnary with the information required to rebuild the visualization panel

        Returns
        -------
        Dict
            Information dict
        """
        return {
            "name": self.panel_name,
            "displayed_field": self.displayed_field,
            "colormap": self.colormap,
            "sync_field": self.sync_field,
            "update_event": self.update_event,
        }

    @classmethod
    def from_json(
        cls, 
        info_dict: Dict, 
        slave: ComputeSlave,
        data: Data3D,
        extensions: Union[List[Extension], List[Tuple[Type[Extension], dict]]] = []
    ) -> "Panel3D":
        """Restores the visualization panel from its information dict

        Parameters
        ----------
        info_dict : Dict
            Dictionnary containing all required information to restore the panel
        slave : ComputeSlave
            Panel associated slave
        data : Data3D
            Initial state Data3D
        extensions : Union[List[Extension], List[Tuple[Type[Extension], dict]]]
            GUI extensions, can be extension classes or tuples of (class, state_dict)

        Returns
        -------
        Panel3D
            Restored panel
        """        
        panel = Panel3D(
            slave = slave,
            name = info_dict["name"],
            extensions = extensions,
            data = data,
            displayed_field = info_dict["displayed_field"],
            colormap = info_dict["colormap"],
        )
        panel.sync_field = info_dict["sync_field"]
        panel.update_event = info_dict["update_event"]
        return panel

    def set_coordinates(
        self,
        u: Tuple[float, float, float] = None,
        v: Tuple[float, float, float] = None,
        u_min: float = None,
        u_max: float = None,
        v_min: float = None,
        v_max: float = None,
        w: float = None,
    ):
        """Updates the plot coordinates

        Parameters
        ----------
        u : Tuple[float, float, float], optional
            Horizontal axis direction vector, by default None
        v : Tuple[float, float, float], optional
            Vertical axis direction vector, by default None
        u_min : float, optional
            Horizontal axis minimum coordinate, by default None
        u_max : float, optional
            Horizontal axis maximum coordinate, by default None
        v_min : float, optional
            Vertical axis minimum coordinate, by default None
        v_max : float, optional
            Vertical axis maximum coordinate, by default None
        w : float, optional
            Normal axis location, by default None
        """
        u_plotter, v_plotter = self.plotter.get_uv()

        if (
            (u is None or np.isclose(u, u_plotter).all())
            and (v is None or np.isclose(v, v_plotter).all())
            and (w is None or np.isclose(w, self.w_value).all())
        ):
            return

        self.plotter.move_slice_to(
            u, v, u_min, u_max, v_min, v_max, w
        )

        if u is not None and v is not None:
            self.u = u
            self.v = v
        if w is not None:
            self.w_value = w

        for extension in self.extensions:
            extension.on_range_change((u_min, u_max), (v_min, v_max), self.w_value)
            extension.on_frame_change(*self.plotter.get_uv())

    def recompute_at(self, position: Tuple[float, float, float], cell_id: str):
        """Triggers a panel recomputation at the provided location. Called by layout update event.

        Parameters
        ----------
        position : Tuple[float, float, float]
            Location to provide to the slave
        cell_id : str
            cell id to provide to the slave
        """
        w = self.plotter.get_slice_normal()

        w_val = np.dot(position, w)

        for extension in self.extensions:
            extension.on_range_change((position[0], None), (position[1], None), w_val)
            extension.on_frame_change(*self.plotter.get_uv())

        self.plotter.set_slice_origin(position)

        if w_val != self.w_value:
            pn.state.notifications.info(f"w updating to {w_val} in {self.panel_name}", 1000)
            self.w_value = w_val
            self.plotter.set_axes(w, self.w_value)
