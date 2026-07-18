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

from scivianna.extension.ai_assistant import AIAssistant

from scivianna.data.data2d import Data2D
from scivianna.interface.generic_interface import Geometry2D

from scivianna.enums import UpdateEvent, VisualizationMode
from scivianna.slave import ComputeSlave

from scivianna.utils.polygon_sorter import PolygonSorter
from scivianna.plotter_2d.polygon.bokeh import Bokeh2DPolygonPlotter
from scivianna.plotter_2d.grid.bokeh import Bokeh2DGridPlotter
from scivianna.plotter_2d.generic_plotter import Plotter2D
from scivianna.constants import MESH, X, Y, Z
import scivianna.utils

profile_time = bool(os.environ["VIZ_PROFILE"]) if "VIZ_PROFILE" in os.environ else 0
if profile_time:
    import time

pn.config.inline = True

default_extensions = [FileLoader, FieldSelector, Axes, AIAssistant]


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

        #
        #   Initializing attributes
        #
        self.update_polygons = False
        """Need to update the polygons geometry (vs just colors)"""
        self.display_polygons = display_polygons

        self.polygon_sorter = PolygonSorter()

        self.field_change_callback: Callable = None
        """Function to call when the field is changed"""

        self.on_axes_change_callback: Callable = None
        """Function to call when the axes are changed"""

        self._pending_updates: Dict = {}
        """Pending visual updates to apply on next tick"""

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

        self.figure.param.watch(self.key_pressed, "key")

    def key_pressed(self, *events):
        """Callback function triggered when a key is pressed in the overlay component.
        It updates the `key` parameter of the panel with the last key pressed.
        If the key is 'X', 'Y', or 'Z', it changes the axes accordingly. If the key is 'F', it flips the u axis.

        Parameters
        ----------
        *events : tuple
            Event information, not used in this function.
        """
        if self.figure.key:
            if self.figure.key == "x":
                print(f"Key pressed: X - Changing axes to XZ for panel {self.panel_name}")
                if np.array_equal(self.u, Y) and np.array_equal(self.v, Z):
                    self.set_coordinates(
                        u=(0, -1, 0),
                        v=Z,
                    )
                else:
                    self.set_coordinates(
                        u=Y,
                        v=Z,
                    )
            if self.figure.key == "y":
                print(f"Key pressed: Y - Changing axes to YZ for panel {self.panel_name}")
                if np.array_equal(self.u, X) and np.array_equal(self.v, Z):
                    self.set_coordinates(
                        u=(-1, 0, 0),
                        v=Z,
                    )
                else:
                    self.set_coordinates(
                        u=X,
                        v=Z,
                    )
            if self.figure.key == "z":
                print(f"Key pressed: Z - Setting axes to X and Y for panel {self.panel_name}")
                if np.array_equal(self.u, X) and np.array_equal(self.v, Y):
                    self.set_coordinates(
                        u=(-1, 0, 0),
                        v=Y,
                    )
                else:
                    self.set_coordinates(
                        u=X,
                        v=Y,
                    )
            if self.figure.key == "f":
                print(f"Key pressed: F - Flipping u axis for panel {self.panel_name}")
                self.set_coordinates(
                    u=[-e for e in self.u],
                    v=self.v,
                )
            self.figure.key = ""  # Reset the key after processing

    @pn.io.hold()
    def _apply_update(self):
        """Apply pending visual updates to the plotter. 
        Called via add_next_tick_callback to ensure UI thread safety.
        """
        if profile_time:
            st = time.time()

        # Update colorbar if needed
        if self._pending_updates.get("colorbar"):
            cb = self._pending_updates["colorbar"]
            self.plotter.update_colorbar(True, (cb["new_low"], cb["new_high"]))
            self.plotter.set_color_map(self.colormap)

        # Update data visualization if needed
        if self._pending_updates.get("data"):
            shapes_matching = self.current_data.cell_ids.shape == self._pending_updates["data"].cell_ids.shape
            self.current_data = self._pending_updates["data"]
            if self.update_polygons or not shapes_matching:
                self.plotter.update_2d_frame(self.current_data)
            else:
                self.plotter.update_colors(self.current_data)

        # Clear pending updates
        self._pending_updates = {}
        self.marked_to_recompute = False

        if profile_time:
            print(f"Apply update : {time.time() - st}")

        # Force notebook refresh
        pn.io.push_notebook(self.figure)

    def _schedule_recompute(self):
        """Schedule a recompute on the next UI tick."""
        self.marked_to_recompute = True
        if pn.state.curdoc is not None:
            pn.state.curdoc.add_next_tick_callback(self._do_recompute)
        elif scivianna.utils._testing:
            self._do_recompute()

    def _schedule_update(self):
        """Schedule a visual update on the next UI tick (no data recompute)."""
        if pn.state.curdoc is not None:
            pn.state.curdoc.add_next_tick_callback(self._apply_update)
        elif scivianna.utils._testing:
            self._apply_update()

    def _do_recompute(self):
        """Perform the actual recompute and schedule visual update."""
        if not self.marked_to_recompute:
            return
            
        if profile_time:
            st = time.time()

        u, v = self.get_uv()

        print(
            f"{self.panel_name} - Recomputing for axes {u}, {v}, at origin : {self.origin}, size_u : {self.size_u}, size_v : {self.size_v}, with field {self.displayed_field}"
        )

        data = self.compute_fn(u, v, self.origin, self.size_u, self.size_v)

        if data is not None:
            if profile_time:
                print(f"Plot panel compute function : {time.time() - st}")
                st = time.time()

            # Build pending updates directly
            self._pending_updates = {"data": data}

            if (
                self.slave.get_label_coloring_mode(
                    self.displayed_field
                ) == VisualizationMode.FROM_VALUE
            ):
                self._pending_updates["colorbar"] = {
                    "new_low": np.nanmin(np.array(data.cell_values).astype(float)),
                    "new_high": np.nanmax(np.array(data.cell_values).astype(float)),
                }

            if profile_time:
                print(f"Plot panel preparing data : {time.time() - st}")

            # Schedule visual update
            self._schedule_update()

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
            list(u),
            list(v),
            list(origin),
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
        # Convert x0, x1, y0, y1 (center coordinates in physical space) to origin/size_u/size_v
        u_arr = np.array(self.u, dtype=float)
        v_arr = np.array(self.v, dtype=float)
        w_arr = np.cross(u_arr, v_arr)
        
        center_u = (x1 + x0) * 0.5
        center_v = (y0 + y1) * 0.5
        
        new_size_u = (x1 - x0)
        new_size_v = (y1 - y0)
        
        # Compute origin from center coordinates
        self.origin = center_u * u_arr + center_v * v_arr + np.dot(w_arr, self.origin) * w_arr

        if new_size_u != self.size_u or new_size_v != self.size_v:
            self.size_u = new_size_u
            self.size_v = new_size_v
            
            for extension in self.extensions:
                extension.on_range_change(self.origin, self.size_u, self.size_v)

        # Schedule recompute if range changes trigger it
        if self.update_event == UpdateEvent.RANGE_CHANGE or (
            isinstance(self.update_event, list) and UpdateEvent.RANGE_CHANGE in self.update_event
        ):
            self._schedule_recompute()
        else:
            # Just update the view without full recompute
            self._schedule_update()

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
        Public method that triggers the recompute pipeline.
        """
        self._schedule_recompute()

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
        if not np.allclose(position, self.origin):
            self.origin = position
            
            for extension in self.extensions:
                extension.on_range_change(self.origin, self.size_u, self.size_v)

            self.plotter.set_axes(self.u, self.v, self.origin)

            self._schedule_recompute()

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
            self._schedule_recompute()

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
            self._schedule_recompute()

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