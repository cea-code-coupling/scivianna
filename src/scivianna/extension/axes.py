from typing import Tuple, TYPE_CHECKING
import numpy as np
import panel as pn
import panel_material_ui as pmui
from scivianna.enums import GeometryType
from scivianna.extension.extension import Extension
from scivianna.plotter_2d.generic_plotter import Plotter2D
from scivianna.slave import ComputeSlave

if TYPE_CHECKING:
    from scivianna.panel.visualisation_panel import VisualizationPanel

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
        slave: ComputeSlave,
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
        #   Bounds widgets
        #     
        self.x0_inp = pmui.FloatInput(
            name="u_min",
            value=0,
            start=-1e6,
            end=1e6,
            step=0.1,
            width=125, margin=5,
            align="center",
        )
        self.y0_inp = pmui.FloatInput(
            name="v_min",
            value=0,
            start=-1e6,
            end=1e6,
            step=0.1,
            width=125, margin=5,
            align="center",
        )
        self.x1_inp = pmui.FloatInput(
            name="u_max",
            value=1,
            start=-1e6,
            end=1e6,
            step=0.1,
            width=125, margin=5,
            align="center",
        )
        self.y1_inp = pmui.FloatInput(
            name="v_max",
            value=1,
            start=-1e6,
            end=1e6,
            step=0.1,
            width=125, margin=5,
            align="center",
        )
        self.w_inp = pmui.FloatInput(
            name="w", value=.5, start=-1e6, end=1e6, step=0.1, width=125, margin=5, align="center"
        )
        self.recompute_button = pmui.Button(
            label = "Update plot",
            description="Update plot using the current bounds",
        )
        self.recompute_button.on_click(self.trigger_update)

        self.w_col = pmui.Column(
            pmui.Typography("Coordinate along the normal axis"),
            self.w_inp,
            margin=0
        )
        def update_w(*args, **kwargs):
            to_update = {"w": self.w_inp.value}
            self.__new_data = {**self.__new_data, **to_update}
            self.range_updated = True
            if pn.state.curdoc is not None:
                pn.state.curdoc.add_next_tick_callback(self.async_update_data)
        self.w_inp.param.watch(update_w, "value")

        # 
        #   Vectors widgets
        #     
        self.u0_inp = pmui.FloatInput(
            name="u0", value=1, start=0, end=1, width=125, margin=5
        )
        self.u1_inp = pmui.FloatInput(
            name="u1", value=0, start=0, end=1, width=125, margin=5
        )
        self.u2_inp = pmui.FloatInput(
            name="u2", value=0, start=0, end=1, width=125, margin=5
        )
        self.v0_inp = pmui.FloatInput(
            name="v0", value=0, start=0, end=1, width=125, margin=5
        )
        self.v1_inp = pmui.FloatInput(
            name="v1", value=1, start=0, end=1, width=125, margin=5
        )
        self.v2_inp = pmui.FloatInput(
            name="v2", value=0, start=0, end=1, width=125, margin=5
        )

        def xplus_fn(event):
            """Defines the direction vectors to Y+ and Z+

            Parameters
            ----------
            event : Any
                Argument to make the function linkable to a button.
            """
            to_update = {"u0": 0, "u1": 1, "u2": 0, "v0": 0, "v1": 0, "v2": 1}
            self.__new_data = {**self.__new_data, **to_update}
            self.axes_updated = True
            if pn.state.curdoc is not None:
                pn.state.curdoc.add_next_tick_callback(self.async_update_data)

        # Attach the CB to the button
        self.xplus = pmui.Button(name="X+", button_type="success", width=50)
        self.xplus.on_click(xplus_fn)

        def yplus_fn(event):
            """Defines the direction vectors to X+ and Z+

            Parameters
            ----------
            event : Any
                Argument to make the function linkable to a button.
            """
            to_update = {"u0": 1, "u1": 0, "u2": 0, "v0": 0, "v1": 0, "v2": 1}
            self.__new_data = {**self.__new_data, **to_update}
            self.axes_updated = True
            if pn.state.curdoc is not None:
                pn.state.curdoc.add_next_tick_callback(self.async_update_data)

        # Attach the CB to the button
        self.yplus = pmui.Button(name="Y+", button_type="success", width=50)
        self.yplus.on_click(yplus_fn)

        def zplus_fn(event):
            """Defines the direction vectors to X+ and Y+

            Parameters
            ----------
            event : Any
                Argument to make the function linkable to a button.
            """
            to_update = {"u0": 1, "u1": 0, "u2": 0, "v0": 0, "v1": 1, "v2": 0}
            self.__new_data = {**self.__new_data, **to_update}
            self.axes_updated = True
            if pn.state.curdoc is not None:
                pn.state.curdoc.add_next_tick_callback(self.async_update_data)

        # Attach the CB to the button
        self.zplus = pmui.Button(name="Z+", button_type="success", width=50)
        self.zplus.on_click(zplus_fn)

        u = pmui.Column(self.u0_inp, self.u1_inp, self.u2_inp, margin=0)
        v = pmui.Column(self.v0_inp, self.v1_inp, self.v2_inp, margin=0)

        self.axis_buttons = pn.Row(self.xplus, self.yplus, self.zplus, margin=0)

        self.bounds_card = pmui.Card(
            pmui.Typography("Bounds along the axes"),
            pmui.Column(
                pmui.Row(
                    self.x0_inp,
                    self.x1_inp, margin=0
                ),
                pmui.Row(
                    self.y0_inp,
                    self.y1_inp, margin=0
                ),
                self.recompute_button,
                margin=0
            ),
            title="Axes bounds",
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
            self.hide_show_button,
            self.w_col, 
            self.bounds_card,
            self.axes_card,
            margin=0
        )

    def on_range_change(self, u_bounds, v_bounds, w_value):
        if [
            *u_bounds, 
            *v_bounds, 
            w_value
        ] != [
            self.x0_inp.value,
            self.x1_inp.value,
            self.y0_inp.value,
            self.y1_inp.value,
            self.w_inp.value
        ]:
            self.__new_data["x0"], self.__new_data["x1"] = u_bounds
            self.__new_data["y0"], self.__new_data["y1"] = v_bounds
            self.__new_data["w"] = w_value
            
            if pn.state.curdoc is not None:
                pn.state.curdoc.add_next_tick_callback(self.async_update_data)

    def on_frame_change(self, u_vector, v_vector):
        u, v = self.get_uv()
        if [*u_vector, *v_vector] != [*u.tolist(), *(v.tolist())]:
            self.__new_data["u0"], self.__new_data["u1"], self.__new_data["u2"] = u_vector
            self.__new_data["v0"], self.__new_data["v1"], self.__new_data["v2"] = v_vector

            if pn.state.curdoc is not None:
                pn.state.curdoc.add_next_tick_callback(self.async_update_data)

    async def async_update_data(self,):
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

            if "x0" in self.__new_data:
                self.x0_inp.value = self.__new_data["x0"]
            if "y0" in self.__new_data:
                self.y0_inp.value = self.__new_data["y0"]
            if "x1" in self.__new_data:
                self.x1_inp.value = self.__new_data["x1"]
            if "y1" in self.__new_data:
                self.y1_inp.value = self.__new_data["y1"]

            if "w" in self.__new_data:
                self.w_inp.value = self.__new_data["w"]

            self.__new_data = {}

        self.update_widgets_visibility()
        if self.axes_updated or self.range_updated:
            self.trigger_update()

    def update_widgets_visibility(self, ):
        geom_type: GeometryType = self.slave.get_geometry_type()

        # Definition of U and V vectors
        self.axes_card.visible = geom_type in [GeometryType._3D, GeometryType._3D_INFINITE]

        # Definition of U and V coords
        self.bounds_card.visible = geom_type in [GeometryType._2D, GeometryType._3D]
        
        # Definition of the normal coordinate
        self.w_inp.visible = geom_type in [GeometryType._3D, GeometryType._3D_INFINITE]
            
    def trigger_update(self, *args, **kwargs):
            u, v = self.get_uv()
            self.panel.set_coordinates(
                u,
                v,
                self.x0_inp.value,
                self.x1_inp.value,
                self.y0_inp.value,
                self.y1_inp.value,
                float(self.w_inp.value),
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