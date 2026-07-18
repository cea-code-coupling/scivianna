from logging import warning
from typing import Callable, Dict, List, Tuple, Type, Union
import numpy as np
import panel as pn
import param
import os

from scivianna.extension.extension import Extension
from scivianna.extension.field_selector import FieldSelector
from scivianna.extension.file_loader import FileLoader
from scivianna.extension.axes import Axes
from scivianna.panel.visualisation_panel import VisualizationPanel

try:
    from scivianna.extension.ai_assistant import AIAssistant
    has_agent = True

except ImportError as e:
    has_agent = False

    print(f"Warning : Agent not loaded, received error : {e}")

except ValueError as e:
    has_agent = False
    print(f"Warning : Agent not loaded, received error : {e}")


from scivianna.data.data2d import Data2D
from scivianna.interface.generic_interface import Geometry2D

from scivianna.enums import UpdateEvent, VisualizationMode
from scivianna.slave import ComputeSlave

from scivianna.utils.polygon_sorter import PolygonSorter
from scivianna.plotter_2d.polygon.bokeh import Bokeh2DPolygonPlotter
from scivianna.plotter_2d.grid.bokeh import Bokeh2DGridPlotter
from scivianna.plotter_2d.generic_plotter import Plotter2D
from scivianna.constants import MESH, X, Y
import scivianna.utils

profile_time = bool(os.environ["VIZ_PROFILE"]) if "VIZ_PROFILE" in os.environ else 0
if profile_time:
    import time

pn.config.inline = True

default_extensions = [FileLoader, FieldSelector, Axes]
if has_agent:
    default_extensions.append(AIAssistant)


class Panel2D(VisualizationPanel):
    """2D Visualisation panel associated to a code."""

    plotter: Plotter2D
    """ 2D plotter displaying and updating the graph
    """
    current_data: Data2D
    """ Displayed data and their properties.
    """
    colormap = param.String()

    def __init__(
        self,
        slave: ComputeSlave,
        name="",
        display_polygons: bool = True,
        extensions: List[Extension] = default_extensions,
        data: Data2D = None,
        displayed_field: str = MESH,
        colormap: str = "BuRd",
        u: Tuple[float, float, float] = X,
        v: Tuple[float, float, float] = Y,
        origin: Tuple[float, float, float] = None,
        size_u: float = 1.0,
        size_v: float = 1.0,
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
        origin : Tuple[float, float, float], optional
            Physical 3D position of the slice center
        size_u : float
            Size of the slice along the u axis, defaults to 1.0.
        size_v : float
            Size of the slice along the v axis, defaults to 1.0.
        """
        code_interface: Type[Geometry2D] = slave.code_interface
        assert issubclass(
            code_interface, Geometry2D
        ), f"A VisualizationPanel can only be given a Geometry2D interface slave, received {code_interface}."

        self.__data_to_update: bool = False
        self.__new_data = {}

        #
        #   Initializing attributes
        #
        self.update_polygons = False
        """Need to update the data at the next async call"""
        self.display_polygons = display_polygons

        self.polygon_sorter = PolygonSorter()

        self.field_change_callback: Callable = None
        """Function to call when the field is changed"""

        self.on_axes_change_callback: Callable = None
        """Function to call when the axes are changed"""

        #
        #   Plotter creation
        #
        if self.display_polygons:
            self.plotter = Bokeh2DPolygonPlotter()
        else:
            self.plotter = Bokeh2DGridPlotter()

        super().__init__(slave, name, extensions.copy())

        self.u = u
        self.v = v

        # Store coordinates in new style: origin, size_u, size_v
        # origin must be provided as a physical 3D position
        self.origin = origin if origin is not None else [0.01, 0.01, 0.01]
        
        self.size_u = size_u
        self.size_v = size_v

        #
        #   First plot on XY basic range
        #
        self.displayed_field = displayed_field
        for extension in self.extensions:
            extension.on_field_change(displayed_field)
            extension.on_frame_change(u, v)
            extension.on_range_change(
                self.origin,
                self.size_u,
                self.size_v
            )

        self.colormap = colormap

        if data is None:
            data_ = self.compute_fn(self.u, self.v, self.origin, self.size_u, self.size_v)
        else:
            print(f"Panel2D {self.panel_name} initialized initial Data2D object, restoring {len(data.cell_ids)} cells.")
            data_ = data
            self.update_polygons = True
        
        self.plotter.set_axes(self.u, self.v, self.origin)
        self.plotter.plot_2d_frame(data_)

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

        # Attach the range update callback to the event
        self.plotter._set_callback_on_range_update(self.ranges_callback)

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
                self.current_data: Data2D = self.__new_data["data"]

                if not self.update_polygons:
                    self.plotter.update_colors(self.current_data)
                else:
                    self.plotter.update_2d_frame(self.current_data)

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
        u: Tuple[float, float, float],
        v: Tuple[float, float, float],
        origin: Tuple[float, float, float],
        size_u: float,
        size_v: float,
    ) -> Data2D:
        """Request the slave to compute a new frame, and updates the data to display

        Parameters
        ----------
        u : Tuple[float, float, float]
            Direction vector along the horizontal axis
        v : Tuple[float, float, float]
            Direction vector along the vertical axis
        origin : Tuple[float, float, float]
            Physical 3D position of the slice center
        size_u : float
            Size of the slice along the u axis
        size_v : float
            Size of the slice along the v axis

        Returns
        -------
        Data2D
            Geometry data.
        """
        options = {key: value for options in [
            e.provide_options() for e in self.extensions
        ] for key, value in options.items()}

        if self.panel_coupling_extension is not None:
            coupling_options = self.panel_coupling_extension.provide_options()
            for key, value in coupling_options.items():
                options[key] = value

        computed_data = self.slave.compute_2D_data(
            u,
            v,
            origin,
            size_u,
            size_v,
            None,
            self.displayed_field,
            options,
            caller=self.panel_name,
        )

        if computed_data is None:
            print(
                f"\n\n Got None from computed data on {self.panel_name}, returning the past values.\n\n"
            )
            return None

        computed_data, polygons_updated = computed_data

        for extension in self.extensions:
            extension.on_updated_data(computed_data)

        if polygons_updated or (self.polygon_sorter.sort_indexes is None):
            self.polygon_sorter.sort_from_value(computed_data)
            self.update_polygons = True
        else:
            self.polygon_sorter.sort_list(computed_data)
            self.update_polygons = False

        return computed_data

    def ranges_callback(
        self,
        x0: float,
        x1: float,
        y0: float,
        y1: float,
    ):
        """Updates the bounds FloatInput based on the current frame zoom.

        Parameters
        ----------
        x0 : float
            Horizontal axis minimum value
        x1 : float
            Horizontal axis maximum value
        y0 : float
            Vertical axis minimum value
        y1 : float
            Vertical axis maximum value
        """
        to_update = {"x0": x0, "x1": x1, "y0": y0, "y1": y1}
        self.__new_data = {**self.__new_data, **to_update}
        
        # Convert x0, x1, y0, y1 (center coordinates in physical space) to origin/size_u/size_v
        u_arr = np.array(self.u, dtype=float)
        v_arr = np.array(self.v, dtype=float)
        
        # x0, x1 are the center_u coordinates, y0, y1 are center_v coordinates
        center_u = (x1 + x0) * .5
        center_v = (y0 + y1) * .5
        
        new_size_u = (x1 - x0)
        new_size_v = (y1 - y0)
        
        # Compute origin from center coordinates
        self.origin = center_u * u_arr + center_v * v_arr

        if new_size_u != self.size_u or new_size_v != self.size_v:
            self.size_u = new_size_u
            self.size_v = new_size_v
            
            for extension in self.extensions:
                extension.on_range_change(self.origin, self.size_u, self.size_v)

        if self.update_event == UpdateEvent.RANGE_CHANGE or (
            isinstance(self.update_event, list) and UpdateEvent.RANGE_CHANGE in self.update_event
        ):
            self.marked_to_recompute = True

        if pn.state.curdoc is not None:
            pn.state.curdoc.add_next_tick_callback(self.async_update_data)
        elif scivianna.utils._testing:
            self.async_update_data()

    def get_uv(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Gets the normal direction vectors from the FloatInput objects.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Vectors U, V
        """
        u = self.u / np.linalg.norm(self.u)
        v = self.v / np.linalg.norm(self.v)

        return u, v

    def recompute(
        self, *args, **kwargs
    ):
        """Recomputes the figure based on the new bounds and parameters.
        """
        if profile_time:
            st = time.time()

        u, v = self.get_uv()

        print(
            f"{self.panel_name} - Recomputing for axes {u}, {v}, at origin : {self.origin}, size_u : {self.size_u}, size_v : {self.size_v}, with field {self.displayed_field}"
        )

        data = self.compute_fn(
            u, v, self.origin, self.size_u, self.size_v
        )

        if data is not None:
            if profile_time:
                print(f"Plot panel compute function : {time.time() - st}")
                st = time.time()

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
        new_visualiser = Panel2D(
            slave=self.slave.duplicate(),
            name=self.panel_name,
            display_polygons=self.display_polygons,
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

    def recompute_at(self, position: Tuple[float, float, float], cell_id: str):
        """Triggers a panel recomputation at the provided location. Called by layout update event.

        Parameters
        ----------
        position : Tuple[float, float, float]
            Location to provide to the slave
        cell_id : str
            cell id to provide to the slave
        """
        # Update origin directly to the clicked position
        new_origin = tuple(position)
        
        if new_origin != self.origin:
            self.origin = new_origin
            
            for extension in self.extensions:
                extension.on_range_change(self.origin, self.size_u, self.size_v)

            self.plotter.set_axes(self.u, self.v, self.origin)

            self.__data_to_update = True
            self.marked_to_recompute = True
            if pn.state.curdoc is not None:
                pn.state.curdoc.add_next_tick_callback(self.async_update_data)
            elif scivianna.utils._testing:
                self.async_update_data()

    def set_coordinates(
        self,
        u: Tuple[float, float, float] = None,
        v: Tuple[float, float, float] = None,
        origin: Tuple[float, float, float] = None,
        size_u: float = None,
        size_v: float = None,
    ):
        """Updates the plot coordinates

        Parameters
        ----------
        u : Tuple[float, float, float], optional
            Horizontal axis direction vector, by default None
        v : Tuple[float, float, float], optional
            Vertical axis direction vector, by default None
        origin : Tuple[float, float, float], optional
            Physical 3D position of the slice center, by default None
        size_u : float, optional
            Size of the slice along the u axis, by default None
        size_v : float, optional
            Size of the slice along the v axis, by default None
        """
        self.__data_to_update = True

        update_axes = False
        if u is not None:
            if not type(u) in [tuple, list, np.ndarray]:
                raise TypeError(
                    f"u must have one of the following types: [tuple, list, np.ndarray], found {type(u)}"
                )
            if not len(u) == 3:
                raise ValueError(f"u must be of length 3, found {len(u)}")
            if not np.array_equal(u, self.u):
                self.u = u
                update_axes = True

        if v is not None:
            if not type(v) in [tuple, list, np.ndarray]:
                raise TypeError(
                    f"v must have one of the following types: [tuple, list, np.ndarray], found {type(v)}"
                )
            if not len(v) == 3:
                raise ValueError(f"v must be of length 3, found {len(v)}")
            if not np.array_equal(v, self.v):
                self.v = v
                update_axes = True

        if update_axes:
            for extension in self.extensions:
                extension.on_frame_change(self.u, self.v)

        update_range = False

        # Handle origin parameter (tuple of 3 floats representing the slice center)
        if origin is not None:
            if not type(origin) in [tuple, list, np.ndarray]:
                raise TypeError(f"origin must be a tuple/list/ndarray of 3 floats, found type {type(origin)}")
            if len(origin) != 3:
                raise ValueError(f"origin must be of length 3, found {len(origin)}")
            if not np.allclose(origin, self.origin):
                self.origin = list(origin)
                update_range = True

        # Handle size_u and size_v parameters
        if size_u is not None:
            if not type(size_u) in [float, int]:
                raise TypeError(f"size_u must be a number, found type {type(size_u)}")
            if size_u != self.size_u:
                self.size_u = size_u
                update_range = True
        
        if size_v is not None:
            if not type(size_v) in [float, int]:
                raise TypeError(f"size_v must be a number, found type {type(size_v)}")
            if size_v != self.size_v:
                self.size_v = size_v
                update_range = True

        if update_range:
            for extension in self.extensions:
                extension.on_range_change(self.origin, self.size_u, self.size_v)

            if self.on_axes_change_callback is not None:
                self.on_axes_change_callback(self.u, self.v, self.origin, self.size_u, self.size_v)

        if update_axes or update_range:
            self.plotter.set_axes(self.u, self.v, self.origin)

            self.marked_to_recompute = True

            if pn.state.curdoc is not None:
                pn.state.curdoc.add_next_tick_callback(self.async_update_data)
            elif scivianna.utils._testing:
                self.async_update_data()

            if self.update_event == UpdateEvent.AXES_CHANGE or (isinstance(self.update_event, list) and UpdateEvent.AXES_CHANGE in self.update_event):
                if self.on_axes_change_callback is not None:
                    self.on_axes_change_callback(
                        u=self.u,
                        v=self.v,
                        origin=self.origin,
                        size_u=self.size_u,
                        size_v=self.size_v,
                    )

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
                # Reseting indexes to prevent weird edges
                if pn.state.curdoc is not None:
                    pn.state.curdoc.add_next_tick_callback(self.polygon_sorter.reset_indexes)
                elif scivianna.utils._testing:
                    self.polygon_sorter.reset_indexes()

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
        """Returns a dictionnary with the information about the visualization panel

        Returns
        -------
        Dict
            Information dict
        """
        return {
            "name": self.panel_name,
            "u": self.u,
            "v": self.v,
            "origin": self.origin,
            "size_u": self.size_u,
            "size_v": self.size_v,
            "displayed_field": self.displayed_field,
            "display_polygons": self.display_polygons,
            "colormap": self.colormap,
            "sync_field": self.sync_field,
            "update_event": self.update_event,
        }

    @classmethod
    def from_json(
        cls, 
        info_dict: Dict, 
        slave: ComputeSlave,
        data: Data2D,
        extensions: Union[List[Extension], List[Tuple[Type[Extension], dict]]] = []
    ) -> "Panel2D":
        """Restores the visualization panel from its information dict

        Parameters
        ----------
        info_dict : Dict
            Dictionnary containing all required information to restore the panel
        slave : ComputeSlave
            Panel associated slave
        data : Data2D
            Initial state Data2D
        extensions : Union[List[Extension], List[Tuple[Type[Extension], dict]]]
            GUI extensions, can be extension classes or tuples of (class, state_dict)

        Returns
        -------
        Panel2D
            Restored panel
        """        
        panel = Panel2D(
            slave = slave,
            name = info_dict["name"],
            display_polygons = info_dict["display_polygons"],
            extensions = extensions,
            data = data,
            displayed_field = info_dict["displayed_field"],
            colormap = info_dict["colormap"],
            u = info_dict["u"],
            v = info_dict["v"],
            origin = info_dict.get("origin"),
            size_u = info_dict.get("size_u", 1.0),
            size_v = info_dict.get("size_v", 1.0),
        )
        panel.sync_field = info_dict["sync_field"]
        panel.update_event = info_dict["update_event"]
        return panel

    def provide_on_axes_change_callback(self, callback: Callable):
        """Stores a function to call everytime the axes are changed.
        the functions takes two numpy arrays and three values (u, v, origin, size_u, size_v).

        Parameters
        ----------
        callback : Callable
            Function to call.
        """
        self.on_axes_change_callback = callback