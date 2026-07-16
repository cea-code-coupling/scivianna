from typing import Tuple, TYPE_CHECKING
import numpy as np
import panel as pn
import panel_material_ui as pmui
from scivianna.enums import GeometryType
from scivianna.extension.extension import Extension
from scivianna.plotter_2d.generic_plotter import Plotter2D
import scivianna.utils

if TYPE_CHECKING:
    from scivianna.panel.visualisation_panel import VisualizationPanel
    from scivianna.slave import ComputeSlave

icon_svg = """
<svg
   version="1.0"
   width="48pt"
   height="48pt"
   viewBox="0 0 48 48"
   preserveAspectRatio="xMidYMid"
   id="svg6"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:svg="http://www.w3.org/2000/svg">
  <defs
     id="defs10" />
  <g
     transform="matrix(0.1,0,0,-0.1,0,48)"
     stroke="none"
     id="g4">
    <path
       d="m 196,424 c -30,-37 -28,-56 4,-36 19,12 20,9 20,-97 V 181 l -74,-40 c -40,-23 -76,-41 -79,-41 -3,0 -3,8 0,19 6,23 -11,35 -25,18 C 36,131 29,108 25,87 18,51 19,49 54,35 73,27 96,20 105,20 c 22,0 19,25 -5,36 -17,8 -9,15 60,52 l 80,43 80,-43 c 69,-37 77,-44 60,-52 -24,-11 -27,-36 -5,-36 9,0 32,7 51,15 35,14 36,16 29,52 -4,21 -11,44 -17,50 -14,17 -31,5 -25,-18 3,-11 3,-19 0,-19 -3,0 -39,18 -79,41 l -74,40 v 110 c 0,106 1,109 20,97 10,-7 22,-9 26,-5 9,9 -48,76 -66,76 -8,0 -28,-16 -44,-35 z"
       id="path2" />
  </g>
</svg>

"""
class Axes(Extension):
    """Extension to load files and send them to the slave."""

    def __init__(
        self,
        slave: "ComputeSlave",
        plotter: Plotter2D,
        panel: "VisualizationPanel"
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
        assert isinstance(plotter, Plotter2D), "Axes extension is built for a Plotter2D only"
        super().__init__(
            "Axes customization",
            icon_svg,
            slave,
            plotter,
            panel,
        )

        self.description = """
The axes extension allows you to edit the axes vectors or the plot bounds along both axes if applicable.

You can also hide/show the axes on the plot and force a plot update.

The following keys are binded:
-   **X** : Moves the axes to a YZ plane
-   **Y** : Moves the axes to a XZ plane
-   **Z** : Moves the axes to a XY plane
-   **F** : Flips the horizontal axis
"""

        self.iconsize = "1.0em"

        self.borders_displayed = False
        self.axes_updated = False
        self.range_updated = False
        self.__new_data = {}

        self.hide_show_button = pmui.Button(
            label = "Toggle axes",
            description="Display plot tools and axis",
        )
        self.hide_show_button.on_click(self.toggle_axis_visibility)

        # 
        #   Bounds widgets (now using origin/size_u/size_v)
        #     
        self.origin_x_inp = pmui.FloatInput(
            label="origin_x",
            value=0,
            start=-1e6,
            end=1e6,
            step=0.1,
            width=125, margin=5,
            align="center",
        )
        self.origin_y_inp = pmui.FloatInput(
            label="origin_y",
            value=0,
            start=-1e6,
            end=1e6,
            step=0.1,
            width=125, margin=5,
            align="center",
        )
        self.origin_z_inp = pmui.FloatInput(
            label="origin_z",
            value=0,
            start=-1e6,
            end=1e6,
            step=0.1,
            width=125, margin=5,
            align="center",
        )
        self.size_u_inp = pmui.FloatInput(
            label="size_u",
            value=1,
            start=0,
            end=1e6,
            step=0.1,
            width=125, margin=5,
            align="center",
        )
        self.size_v_inp = pmui.FloatInput(
            label="size_v",
            value=1,
            start=0,
            end=1e6,
            step=0.1,
            width=125, margin=5,
            align="center",
        )
        self.recompute_button = pmui.Button(
            label = "Update plot",
            description="Update plot using the current bounds",
        )
        self.recompute_button.on_click(self.trigger_update)

        def update_origin_x(*args, **kwargs):
            if self._restoring:
                return
            to_update = {"origin_x": self.origin_x_inp.value}
            self.__new_data = {**self.__new_data, **to_update}
            self.range_updated = True
            if pn.state.curdoc is not None:
                pn.state.curdoc.add_next_tick_callback(self.async_update_data)
            elif scivianna.utils._testing:
                self.update_data()

        def update_origin_y(*args, **kwargs):
            if self._restoring:
                return
            to_update = {"origin_y": self.origin_y_inp.value}
            self.__new_data = {**self.__new_data, **to_update}
            self.range_updated = True
            if pn.state.curdoc is not None:
                pn.state.curdoc.add_next_tick_callback(self.async_update_data)
            elif scivianna.utils._testing:
                self.update_data()

        def update_origin_z(*args, **kwargs):
            if self._restoring:
                return
            to_update = {"origin_z": self.origin_z_inp.value}
            self.__new_data = {**self.__new_data, **to_update}
            self.range_updated = True
            if pn.state.curdoc is not None:
                pn.state.curdoc.add_next_tick_callback(self.async_update_data)
            elif scivianna.utils._testing:
                self.update_data()

        def update_size_u(*args, **kwargs):
            if self._restoring:
                return
            to_update = {"size_u": self.size_u_inp.value}
            self.__new_data = {**self.__new_data, **to_update}
            self.range_updated = True
            if pn.state.curdoc is not None:
                pn.state.curdoc.add_next_tick_callback(self.async_update_data)
            elif scivianna.utils._testing:
                self.update_data()

        def update_size_v(*args, **kwargs):
            if self._restoring:
                return
            to_update = {"size_v": self.size_v_inp.value}
            self.__new_data = {**self.__new_data, **to_update}
            self.range_updated = True
            if pn.state.curdoc is not None:
                pn.state.curdoc.add_next_tick_callback(self.async_update_data)
            elif scivianna.utils._testing:
                self.update_data()

        self.origin_x_inp.param.watch(update_origin_x, "value")
        self.origin_y_inp.param.watch(update_origin_y, "value")
        self.origin_z_inp.param.watch(update_origin_z, "value")
        self.size_u_inp.param.watch(update_size_u, "value")
        self.size_v_inp.param.watch(update_size_v, "value")

        #
        #   Vectors widgets
        #
        self.u0_inp = pmui.FloatInput(
            label="u0", value=1, start=-1, end=1, width=125, margin=5
        )
        self.u1_inp = pmui.FloatInput(
            label="u1", value=0, start=-1, end=1, width=125, margin=5
        )
        self.u2_inp = pmui.FloatInput(
            label="u2", value=0, start=-1, end=1, width=125, margin=5
        )
        self.v0_inp = pmui.FloatInput(
            label="v0", value=0, start=-1, end=1, width=125, margin=5
        )
        self.v1_inp = pmui.FloatInput(
            label="v1", value=1, start=-1, end=1, width=125, margin=5
        )
        self.v2_inp = pmui.FloatInput(
            label="v2", value=0, start=-1, end=1, width=125, margin=5
        )

        def xplus_fn(event):
            """Defines the direction vectors to Y+ and Z+

            Parameters
            ----------
            event : Any
                Argument to make the function linkable to a button.
            """
            if self._restoring:
                return
            to_update = {"u0": 0, "u1": 1, "u2": 0, "v0": 0, "v1": 0, "v2": 1}
            self.__new_data = {**self.__new_data, **to_update}
            self.axes_updated = True
            if pn.state.curdoc is not None:
                pn.state.curdoc.add_next_tick_callback(self.async_update_data)
            elif scivianna.utils._testing:
                self.update_data()

        # Attach the CB to the button
        self.xplus = pmui.Button(label="X+", button_type="success", width=50)
        self.xplus.on_click(xplus_fn)

        def yplus_fn(event):
            """Defines the direction vectors to X+ and Z+

            Parameters
            ----------
            event : Any
                Argument to make the function linkable to a button.
            """
            if self._restoring:
                return
            to_update = {"u0": 1, "u1": 0, "u2": 0, "v0": 0, "v1": 0, "v2": 1}
            self.__new_data = {**self.__new_data, **to_update}
            self.axes_updated = True
            if pn.state.curdoc is not None:
                pn.state.curdoc.add_next_tick_callback(self.async_update_data)
            elif scivianna.utils._testing:
                self.update_data()

        # Attach the CB to the button
        self.yplus = pmui.Button(label="Y+", button_type="success", width=50)
        self.yplus.on_click(yplus_fn)

        def zplus_fn(event):
            """Defines the direction vectors to X+ and Y+

            Parameters
            ----------
            event : Any
                Argument to make the function linkable to a button.
            """
            if self._restoring:
                return
            to_update = {"u0": 1, "u1": 0, "u2": 0, "v0": 0, "v1": 1, "v2": 0}
            self.__new_data = {**self.__new_data, **to_update}
            self.axes_updated = True
            if pn.state.curdoc is not None:
                pn.state.curdoc.add_next_tick_callback(self.async_update_data)
            elif scivianna.utils._testing:
                self.update_data()

        # Attach the CB to the button
        self.zplus = pmui.Button(label="Z+", button_type="success", width=50)
        self.zplus.on_click(zplus_fn)

        u = pmui.Column(self.u0_inp, self.u1_inp, self.u2_inp, margin=0)
        v = pmui.Column(self.v0_inp, self.v1_inp, self.v2_inp, margin=0)

        self.axis_buttons = pn.Row(self.xplus, self.yplus, self.zplus, margin=0)

        self.bounds_card = pmui.Card(
            pmui.Typography("Slice center and size"),
            pmui.Column(
                pmui.Row(
                    self.origin_x_inp,
                    self.origin_y_inp,
                    self.origin_z_inp, margin=0
                ),
                pmui.Row(
                    self.size_u_inp,
                    self.size_v_inp, margin=0
                ),
                margin=0
            ),
            title="Slice bounds",
            width=300,
            margin=0,
            collapsed=True,
        )

        self.axes_card = pmui.Card(
            pmui.Column(
            pmui.Typography("2D plot plane vectors"),
            self.axis_buttons,
            pn.Row(u, v, margin=0),
            margin=0),
            title="Axes vectors",
            width=300,
            margin=0,
            collapsed=True,
            outlined=False
        )
        self.update_widgets_visibility()

    @pn.io.hold()
    def toggle_axis_visibility(self, *args, **kwargs):
        """Hides and shows the figure axis

        Parameters
        ----------
        _ : Any
            Button clic event
        """
        if not self.borders_displayed:
            self.plotter.display_borders(True)
            self.borders_displayed = True
        else:
            self.plotter.display_borders(False)
            self.borders_displayed = False

    def make_gui(self,) -> pn.viewable.Viewable:
        """Returns a panel viewable to display in the extension tab.

        Returns
        -------
        pn.viewable.Viewable
            Viewable to display in the extension tab
        """
        return pmui.Column(
            pmui.Typography("Hide/show axis"),
            self.recompute_button,
            self.hide_show_button,
            self.bounds_card,
            self.axes_card,
            margin=0
        )

    def on_range_change(self, origin, size_u, size_v):
        """Called when the range/coordinates change.
        
        Parameters
        ----------
        origin : tuple
            Physical 3D position of the slice center (x, y, z)
        size_u : float
            Size of the slice along the u axis
        size_v : float
            Size of the slice along the v axis
        """
        if origin is None or size_u is None or size_v is None:
            return
            
        origin_tuple = tuple(origin) if not isinstance(origin, tuple) else origin
        
        if (origin_tuple[0] != self.origin_x_inp.value or 
            origin_tuple[1] != self.origin_y_inp.value or 
            origin_tuple[2] != self.origin_z_inp.value or
            size_u != self.size_u_inp.value or
            size_v != self.size_v_inp.value):
            
            self.__new_data["origin_x"] = origin_tuple[0]
            self.__new_data["origin_y"] = origin_tuple[1]
            self.__new_data["origin_z"] = origin_tuple[2]
            self.__new_data["size_u"] = size_u
            self.__new_data["size_v"] = size_v
            
            if pn.state.curdoc is not None:
                pn.state.curdoc.add_next_tick_callback(self.async_update_data)
            elif scivianna.utils._testing:
                self.update_data()

    def on_frame_change(self, u_vector, v_vector):
        u, v = self.get_uv()
        if [*list(u_vector), *list(v_vector)] != [*u.tolist(), *(v.tolist())]:
            if not (all([
                e == 0 for e in u
            ]) or all([
                e == 0 for e in v
            ])):
                self.__new_data["u0"], self.__new_data["u1"], self.__new_data["u2"] = u_vector
                self.__new_data["v0"], self.__new_data["v1"], self.__new_data["v2"] = v_vector

                if pn.state.curdoc is not None:
                    pn.state.curdoc.add_next_tick_callback(self.async_update_data)
                elif scivianna.utils._testing:
                    self.update_data()

    def update_data(self,):
        if self.__new_data != {}:
            if "u0" in self.__new_data:
                self.u0_inp.value = self.__new_data["u0"]
            if "u1" in self.__new_data:
                self.u1_inp.value = self.__new_data["u1"]
            if "u2" in self.__new_data:
                self.u2_inp.value = self.__new_data["u2"]

            if "v0" in self.__new_data:
                self.v0_inp.value = self.__new_data["v0"]
            if "v1" in self.__new_data:
                self.v1_inp.value = self.__new_data["v1"]
            if "v2" in self.__new_data:
                self.v2_inp.value = self.__new_data["v2"]

            if "origin_x" in self.__new_data:
                self.origin_x_inp.value = self.__new_data["origin_x"]
            if "origin_y" in self.__new_data:
                self.origin_y_inp.value = self.__new_data["origin_y"]
            if "origin_z" in self.__new_data:
                self.origin_z_inp.value = self.__new_data["origin_z"]
            if "size_u" in self.__new_data:
                self.size_u_inp.value = self.__new_data["size_u"]
            if "size_v" in self.__new_data:
                self.size_v_inp.value = self.__new_data["size_v"]

            self.__new_data = {}

        self.update_widgets_visibility()
        if self.axes_updated or self.range_updated:
            self.trigger_update()

    async def async_update_data(self,):
        self.update_data()

    def update_widgets_visibility(self, ):
        geom_type: GeometryType = self.slave.get_geometry_type()

        # Definition of U and V vectors
        self.axes_card.visible = geom_type in [GeometryType._3D, GeometryType._3D_INFINITE]

        # Definition of U and V coords
        self.bounds_card.visible = geom_type in [GeometryType._2D, GeometryType._3D]
        
        # Origin Z is only relevant for 3D (for 2D, origin is in u-v plane)
        self.origin_z_inp.visible = geom_type in [GeometryType._3D, GeometryType._3D_INFINITE]
            
    def trigger_update(self, *args, **kwargs):
        if self._restoring:
            return
        u, v = self.get_uv()
        
        # Get origin and size values from widgets
        origin = (
            self.origin_x_inp.value,
            self.origin_y_inp.value,
            self.origin_z_inp.value,
        )
        size_u = self.size_u_inp.value
        size_v = self.size_v_inp.value
        
        self.panel.set_coordinates(
            u,
            v,
            origin=origin,
            size_u=size_u,
            size_v=size_v,
        )

    def get_uv(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Gets the normal direction vectors from the FloatInput objects.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Vectors U, V
        """
        u = np.array([self.u0_inp.value, self.u1_inp.value, self.u2_inp.value])
        v = np.array([self.v0_inp.value, self.v1_inp.value, self.v2_inp.value])

        u = u / np.linalg.norm(u)
        v = v / np.linalg.norm(v)

        return u, v

    def to_json(self) -> dict:
        """Returns a dictionary with the information required to rebuild the extension.

        Returns
        -------
        dict
            Information dictionary
        """
        return {
            "borders_displayed": self.borders_displayed,
            "u_vector": [self.u0_inp.value, self.u1_inp.value, self.u2_inp.value],
            "v_vector": [self.v0_inp.value, self.v1_inp.value, self.v2_inp.value],
            "origin": [self.origin_x_inp.value, self.origin_y_inp.value, self.origin_z_inp.value],
            "size_u": self.size_u_inp.value,
            "size_v": self.size_v_inp.value,
        }

    @classmethod
    def from_json(cls, extension: "Axes", info_dict: dict) -> "Axes":
        """Restores the extension from its information dict.

        Parameters
        ----------
        extension : Axes
            Extension instance to restore
        info_dict : dict
            Dictionary containing extension state information

        Returns
        -------
        Axes
            Restored extension
        """
        extension._restoring = True

        extension.borders_displayed = info_dict.get("borders_displayed", False)
        
        u_vector = info_dict.get("u_vector", [1, 0, 0])
        extension.u0_inp.value = u_vector[0]
        extension.u1_inp.value = u_vector[1]
        extension.u2_inp.value = u_vector[2]

        v_vector = info_dict.get("v_vector", [0, 1, 0])
        extension.v0_inp.value = v_vector[0]
        extension.v1_inp.value = v_vector[1]
        extension.v2_inp.value = v_vector[2]
        
        origin = info_dict.get("origin", [0.0, 0.0, 0.0])
        extension.origin_x_inp.value = origin[0]
        extension.origin_y_inp.value = origin[1]
        extension.origin_z_inp.value = origin[2]
        extension.size_u_inp.value = info_dict.get("size_u", 1.0)
        extension.size_v_inp.value = info_dict.get("size_v", 1.0)
        
        extension._restoring = False
        return extension
