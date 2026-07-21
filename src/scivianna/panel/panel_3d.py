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
from scivianna.constants import MESH, X, Y, DEFAULT_ORIGIN
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
        origin : Tuple[float, float, float], optional
            Physical 3D position of the slice center, defaults to None
        size_u : float
            Size of the slice along the u axis (not used for 3D but kept for API compatibility), defaults to None
        size_v : float
            Size of the slice along the v axis (not used for 3D but kept for API compatibility), defaults to None
        w : float
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

        self.w_inp = 0.
        self.origin = tuple(DEFAULT_ORIGIN)
        self.u = X
        self.v = Y
        self._pending_updates: Dict = {}
        """Pending visual updates to apply on next tick"""
        
        for extension in self.extensions:
            extension.on_range_change(
                self.origin, 
                1.0, 
                1.0
            )
            extension.on_frame_change(*self.plotter.get_uv())

        try:
            pn.state.on_session_created(self.recompute)
        except Exception:
            pass

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
            self.current_data = self._pending_updates["data"]
            if self.update_polygons:
                self.plotter.plot(self.current_data)
            else:
                self.plotter.update_plot(self.current_data)

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

        print(
            f"{self.panel_name} - Recomputing with field {self.displayed_field}"
        )

        data = self.compute_fn()

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
        the functions takes two numpy arrays and three values (u, v, origin, size_u, size_v).

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
            self._schedule_recompute()

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
            Size of the slice along the u axis (not used for 3D but kept for API compatibility), by default None
        size_v : float, optional
            Size of the slice along the v axis (not used for 3D but kept for API compatibility), by default None
        """
        u_plotter, v_plotter = self.plotter.get_uv()

        if (
            (u is None or np.isclose(u, u_plotter).all())
            and (v is None or np.isclose(v, v_plotter).all())
        ):
            return

        if origin is not None:
            self.origin = tuple(origin)

        if u is not None and v is not None:
            self.u = u
            self.v = v

        self.plotter.move_slice_to(
            u, v, 
            self.origin
        )

        for extension in self.extensions:
            extension.on_range_change(self.origin, 1.0, 1.0)
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

        for extension in self.extensions:
            extension.on_range_change(position, 1.0, 1.0)
            extension.on_frame_change(*self.plotter.get_uv())

        self.plotter.set_slice_origin(position)
