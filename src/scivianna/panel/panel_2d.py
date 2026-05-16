from logging import warning
from typing import Callable, List, Tuple, Type
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
        extensions: List[Extension] = default_extensions
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
        """
        code_interface: Type[Geometry2D] = slave.code_interface
        assert issubclass(
            code_interface, Geometry2D
        ), f"A VisualizationPanel can only be given a Geometry2D interface slave, received {code_interface}."

        #
        #   Initializing attributes
        #
        self.update_polygons = False
        """Need to update the data at the next async call"""
        self.display_polygons = display_polygons

        self.polygon_sorter = PolygonSorter()

        self.field_change_callback: Callable = None
        """Function to call when the field is changed"""

        #
        #   Plotter creation
        #
        if self.display_polygons:
            self.plotter = Bokeh2DPolygonPlotter()
        else:
            self.plotter = Bokeh2DGridPlotter()

        super().__init__(slave, name, extensions.copy())

        self.u = X
        self.v = Y

        self.u_range = (0., 1.)
        self.v_range = (0., 1.)
        self.w_value = 0.5

        #
        #   First plot on XY basic range
        #
        self.displayed_field = MESH
        for extension in self.extensions:
            extension.on_field_change(MESH)

        self.colormap = "BuRd"

        data_ = self.compute_fn(self.u, self.v, self.u_range[0], self.v_range[0], self.u_range[1], self.v_range[1], self.w_value)

        self.plotter.set_axes(self.u, self.v, self.w_value)
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

        self.__data_to_update: bool = False
        self.__new_data = {}

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
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        z: float,
    ) -> Data2D:
        """Request the slave to compute a new frame, and updates the data to display

        Parameters
        ----------
        u : Tuple[float, float, float]
            Direction vector along the horizontal axis
        v : Tuple[float, float, float]
            Direction vector along the vertical axis
        x0 : float
            Lower bound value along the u axis
        y0 : float
            Lower bound value along the v axis
        x1 : float
            Upper bound value along the u axis
        y1 : float
            Upper bound value along the v axis
        z : float
            Value along the u ^ v axis

        Returns
        -------
        Data2D
            Geometry data.
        """
        options = {key: value for options in [
            e.provide_options() for e in self.extensions
        ] for key, value in options.items()}

        computed_data = self.slave.compute_2D_data(
            u,
            v,
            x0,
            x1,
            y0,
            y1,
            z,
            None,
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
        if pn.state.curdoc is not None:
            pn.state.curdoc.add_next_tick_callback(self.async_update_data)
        elif scivianna.utils._testing:
            self.async_update_data()

        self.u_range = (x0, x1)
        self.v_range = (y0, y1)

        for extension in self.extensions:
            extension.on_range_change((x0, x1), (y0, y1), self.w_value)

        if self.update_event == UpdateEvent.RANGE_CHANGE or (
            isinstance(self.update_event, list) and UpdateEvent.RANGE_CHANGE in self.update_event
        ):
            self.marked_to_recompute = True

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
            f"{self.panel_name} - Recomputing for axes {u}, {v}, at range : {self.u}, {self.v}, ({self.w_value}), with field {self.displayed_field}"
        )

        data = self.compute_fn(
            u, v, self.u_range[0], self.v_range[0], self.u_range[1], self.v_range[1], self.w_value
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
        u, v = self.get_uv()

        w = np.cross(u, v)
        w_val = np.dot(position, w)

        for extension in self.extensions:
            extension.on_range_change(self.u_range, self.v_range, w_val)

        if w_val != self.w_value:
            pn.state.notifications.info(f"w updating to {w_val} in {self.panel_name}", 1000)
            self.w_value = w_val
            self.plotter.set_axes(self.u, self.v, self.w_value)

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
        if u_min is not None:
            if not type(u_min) in [float, int]:
                raise TypeError(f"u_min must be a number, found type {type(u_min)}")
            if u_min != self.u_range[0]:
                update_range = True
        else:
            u_min = self.u_range[0]

        if v_min is not None:
            if not type(v_min) in [float, int]:
                raise TypeError(f"v_min must be a number, found type {type(v_min)}")
            if v_min != self.v_range[0]:
                update_range = True
        else:
            v_min = self.v_range[0]

        if u_max is not None:
            if not type(u_max) in [float, int]:
                raise TypeError(f"u_max must be a number, found type {type(u_max)}")
            if u_max != self.u_range[1]:
                update_range = True
        else:
            u_max = self.u_range[1]

        if v_max is not None:
            if not type(v_max) in [float, int]:
                raise TypeError(f"v_max must be a number, found type {type(v_max)}")
            if v_max != self.v_range[1]:
                update_range = True
        else:
            v_max = self.v_range[1]

        if w is not None:
            if not type(w) in [float, int]:
                raise TypeError(f"w must be a number, found type {type(w)}")
            if w != self.w_value:
                self.w_value = w
                update_range = True

        if update_range:
            self.u_range = (u_min, u_max)
            self.v_range = (v_min, v_max)

            for extension in self.extensions:
                extension.on_range_change(self.u_range, self.v_range, self.w_value)

        if update_axes or update_range:
            self.plotter.set_axes(self.u, self.v, self.w_value)
            
            for extension in self.extensions:
                extension.on_frame_change(self.u, self.v)

            self.marked_to_recompute = True

            if pn.state.curdoc is not None:
                pn.state.curdoc.add_next_tick_callback(self.async_update_data)
            elif scivianna.utils._testing:
                self.async_update_data()

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
                print(pn.state.curdoc)
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
