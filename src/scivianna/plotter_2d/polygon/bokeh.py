import functools
from typing import IO, Callable, List, Tuple, Union
import bokeh.events
import panel as pn
from scivianna.data.data2d import Data2D
from scivianna.utils.polygonize_tools import PolygonElement
from scivianna.plotter_2d.generic_plotter import Plotter2D

import bokeh
from bokeh.colors import RGB
from bokeh.plotting import figure as Figure
from bokeh.plotting import save, output_file
from bokeh.models import (
    HoverTool,
    ColumnDataSource,
    CustomJSHover,
    LinearColorMapper,
    ColorBar,
    TapTool,
)
# from bokeh.models import CustomJS
from bokeh import events

import numpy as np

from scivianna.constants import XS, YS, CELL_NAMES, COMPO_NAMES, COLORS, EDGE_COLORS, GEOMETRY, EDGE_ALPHA, FILL_ALPHA
from scivianna.utils.color_tools import beautiful_color_maps

import os


class Bokeh2DPolygonPlotter(Plotter2D):
    """2D geometry plotter based on the bokeh python module"""

    def __init__(
        self,
    ):
        """Creates the bokeh Figure and ColumnDataSources"""
        self.source_polygons = ColumnDataSource(
            {
                XS: [],
                YS: [],
                COLORS: [],
                EDGE_COLORS: [],
                CELL_NAMES: [],
                COMPO_NAMES: [],
            }
        )

        self.source_coordinates = ColumnDataSource(
            {
                "u_min": [0],
                "dw": [1],
                "v_min": [0],
                "dh": [1],
                "u0": [1.0],
                "u1": [0.0],
                "u2": [0.0],
                "v0": [0.0],
                "v1": [1.0],
                "v2": [0.0],
                "w0": [0.0],
                "w1": [0.0],
                "w2": [1.0],
                "w": [0],
                # "x":[x],
                # "y":[y],
            }
        )

        self.source_mouse = ColumnDataSource(
            {
                # U-V space coordinates
                "u": [0],
                "v": [0],
                # Geometry space coordinates
                "x": [0],
                "y": [0],
                "z": [0],
                # Polygon index
                "index": [0],
                # Plot size?
                "snap_sx": [0],
                "snap_sy": [0],
                # Pixel coordinates in plot
                "sx": [0],
                "sy": [0],
            }
        )

        code_get_mouse_location = """
            const u0 = full_data.data.u0[0]
            const v0 = full_data.data.v0[0]
            const w0 = full_data.data.w0[0]
            const u1 = full_data.data.u1[0]
            const v1 = full_data.data.v1[0]
            const w1 = full_data.data.w1[0]
            const u2 = full_data.data.u2[0]
            const v2 = full_data.data.v2[0]
            const w2 = full_data.data.w2[0]
            const w = full_data.data.w[0]

            const x = special_vars.x;
            const y = special_vars.y;

            const new_data = Object.assign({}, mouse.data)

            new_data["u"] = [x];
            new_data["v"] = [y];

            new_data["x"] = [x*u0 + y*v0 +w*w0];
            new_data["y"] = [x*u1 + y*v1 +w*w1];
            new_data["z"] = [x*u2 + y*v2 +w*w2];

            new_data["index"] = [special_vars.index]

            new_data["sx"] = [special_vars.sx];
            new_data["sy"] = [special_vars.sy];

            new_data["snap_sx"] = [special_vars.snap_sx];
            new_data["snap_sy"] = [special_vars.snap_sy];

            mouse.data = new_data

            mouse.change.emit();

            return  "(" + (x*u0 + y*v0 +w*w0).toFixed(3) + ", " + (x*u1 + y*v1 +w*w1).toFixed(3)+", "+  (x*u2 + y*v2 +w*w2 ).toFixed(3) + ")"
        """

        # Manifestement, ca ne marche pas avec un nom a plusieurs caracteres pour x ???
        TOOLTIPS = [
            ("Coordinates", "$x{custom}"),
            ("Cell ID", "@cell_names"),
            ("Value", "@compo_names"),
        ]

        self.hover_tool = HoverTool(
            tooltips=TOOLTIPS,
            formatters={
                "$x": CustomJSHover(
                    args=dict(
                        full_data=self.source_coordinates, mouse=self.source_mouse
                    ),
                    code=code_get_mouse_location,
                )
            },
            point_policy="follow_mouse"
        )

        self.figure = Figure(
            name="plot",
            sizing_mode="stretch_both",
            match_aspect=True,
            toolbar_location=None,
        )
        self.figure.xgrid.grid_line_color = None
        self.figure.ygrid.grid_line_color = None

        self.figure.min_border_left = 0
        self.figure.min_border_right = 0
        self.figure.min_border_top = 0
        self.figure.min_border_bottom = 0

        self.figure.xaxis.visible = False
        self.figure.yaxis.visible = False

        self.figure.add_tools(self.hover_tool)

        self.color_mapper = LinearColorMapper(
            palette=self.__get_color_mapper_from_string("BuRd"), low=0.0, high=1.0
        )

        self.figure_color_bar = ColorBar(
            color_mapper=self.color_mapper,
            label_standoff=12,
            border_line_color=None,
            location=(0, 0),
        )

        self.figure.add_layout(self.figure_color_bar, "right")

        # This is a massive hack, too lazy to do it the right way !
        zoom_tool = [
            t
            for t in self.figure.toolbar.tools
            if type(t).__name__.startswith("WheelZoom")
        ][0]
        pan_tool = [
            t
            for t in self.figure.toolbar.tools
            if type(t).__name__.startswith("PanTool")
        ][0]
        self.figure.toolbar.active_scroll = zoom_tool
        self.figure.toolbar.active_drag = pan_tool

    def display_borders(self, display: bool):
        """Display or hides the figure borders and axis

        Parameters
        ----------
        display : bool
            Display if true, hides otherwise
        """
        if display:
            self.figure.toolbar_location = "right"
            self.figure.xaxis.visible = True
            self.figure.yaxis.visible = True
        else:
            self.figure.toolbar_location = None
            self.figure.xaxis.visible = False
            self.figure.yaxis.visible = False

    # @pn.io.hold()
    def update_colorbar(self, display: bool, value_range: Tuple[float, float]):
        """Displays or hide the color bar, if display, updates its range

        Parameters
        ----------
        display : bool
            Display or hides the color bar
        value_range : Tuple[float, float]
            New colormap range
        """
        if (value_range[0] is not None) and any(np.isnan(value_range)):
            self.figure_color_bar.visible = False
            return

        if display:
            self.color_mapper.update(low=value_range[0], high=value_range[1])
            self.figure_color_bar.color_mapper = self.color_mapper
        self.figure_color_bar.visible = display

    def set_color_map(self, color_map_name: str):
        """Sets the colorbar color map name

        Parameters
        ----------
        color_map_name : str
            Color map name
        """
        self.colormap = color_map_name
        self.figure_color_bar.color_mapper = self.color_mapper
        self.figure_color_bar.color_mapper.update(
            palette=self.__get_color_mapper_from_string(color_map_name),
        )

    def plot_2d_frame(
        self,
        data: Data2D,
    ):
        """Adds a new plot to the figure from a set of polygons

        Parameters
        ----------
        data : Data2D
            Data2D object containing the geometry to plot
        """
        xs, ys = self._polygons_to_coords(data.get_polygons())

        self.source_polygons.data = {
            XS: xs,
            YS: ys,
            CELL_NAMES: data.cell_ids,
            COMPO_NAMES: data.cell_values,
            COLORS: np.array(data.cell_colors)[:, :-1].tolist(),
            FILL_ALPHA: (np.array(data.cell_colors)[:, -1]/255).tolist(),
            EDGE_COLORS: np.array(data.cell_edge_colors)[:, :-1].tolist(),
            EDGE_ALPHA: (np.array(data.cell_edge_colors)[:, -1]/255).tolist(),
        }

        self.hovered_glyph = self.figure.multi_polygons(
            xs=XS,
            ys=YS,
            line_width = self.line_width,
            source=self.source_polygons,
            color=COLORS,
            line_color=EDGE_COLORS,
            fill_alpha = FILL_ALPHA,
            line_alpha = EDGE_ALPHA,
            # hover_line_alpha=0.6,
            # hover_fill_alpha=0.6,
        )

    def update_2d_frame(
        self,
        data: Data2D,
    ):
        """Updates plot to the figure

        Parameters
        ----------
        data : Data2D
            Data2D object containing the data to update
        """
        xs, ys = self._polygons_to_coords(data.get_polygons())

        self.source_polygons.update(
            data={
                XS: xs,
                YS: ys,
                CELL_NAMES: data.cell_ids,
                COMPO_NAMES: data.cell_values,
                COLORS: np.array(data.cell_colors).tolist(),
                EDGE_COLORS: np.array(data.cell_edge_colors).tolist(),
                FILL_ALPHA: (np.array(data.cell_colors)[:, -1]/255).tolist(),
                EDGE_ALPHA: (np.array(data.cell_edge_colors)[:, -1]/255).tolist(),
            }
        )

    def update_colors(self, data: Data2D,):
        """Updates the colors of the displayed polygons

        Parameters
        ----------
        data : Data2D
            Data2D object containing the data to update
        """
        data.convert_to_polygons()
        colors = data.cell_colors
        cell_count = len(colors)

        self.source_polygons.patch(
            {
                COMPO_NAMES: [(slice(0, cell_count), data.cell_values)],
                COLORS: [(slice(0, cell_count), np.array(data.cell_colors)[:, :-1].tolist())],
                EDGE_COLORS: [
                    (slice(0, cell_count), np.array(data.cell_edge_colors).tolist())
                ],
                FILL_ALPHA: [
                    (slice(0, cell_count), (np.array(data.cell_colors)[:, -1]/255).tolist())
                ],
                EDGE_ALPHA: [
                    (slice(0, cell_count), (np.array(data.cell_edge_colors)[:, -1]/255).tolist())
                ],
            }
        )

    def _set_callback_on_range_update(self, callback: IO):
        """Sets a callback to update the x and y ranges in the GUI.

        Parameters
        ----------
        callback : IO
            Function that takes x0, x1, y0, y1 as arguments
        """

        def update_range(event: events.RangesUpdate):
            callback(event.x0, event.x1, event.y0, event.y1)

        self.figure.on_event("rangesupdate", update_range)

    def make_panel(self) -> pn.viewable.Viewable:
        """Makes the Holoviz panel viewable displayed in the web app.

        Returns
        -------
        pn.viewable.Viewable
            Displayed viewable
        """
        return pn.pane.Bokeh(
            self.figure,
            name=GEOMETRY,
            sizing_mode="stretch_both",
            margin=0,
        )

    def _disable_interactions(self, disable: bool):
        """Disables de plot interactions for multi panel web-app resizing

        Parameters
        ----------
        disable : bool
            Disable if True, enable if False
        """
        if disable:
            self.figure.toolbar.active_drag = None
        else:
            self.figure.toolbar.active_drag = [
                t
                for t in self.figure.toolbar.tools
                if type(t).__name__.startswith("PanTool")
            ][0]

    def __get_color_mapper_from_string(self, color_map_name: str) -> List[RGB]:
        """Gets a color list from a color map name

        Parameters
        ----------
        color_map_name : str
            Color map name

        Returns
        -------
        List[RGB]
            List of RGB colors
        """
        return [RGB(*c) for c in beautiful_color_maps[color_map_name]]

    def get_resolution(self) -> Tuple[float, float]:
        """Returns the current plot resolution to display. For resolution based codes, it will be replaced by the value present in the gui

        Returns
        -------
        Tuple[float, float]
            Resolution if possible, else (None, None)
        """
        # Try catch here in case the figure does not have inner_width or inner_height set.
        try:
            res_x = self.figure.inner_width
            res_y = self.figure.inner_height

            return res_x, res_y
        except:
            return None, None

    def export(self, file_name: str, title="Bokeh 2D plot"):
        """Exports the plot in a file

        Parameters
        ----------
        file_name : str
            Export file path
        """
        filename, file_extension = os.path.splitext(file_name)
        if not file_extension.lower() == ".html":
            raise ValueError(
                f"This plotter can only export to .html format, requested {file_extension}."
            )

        output_file(filename=file_name, title=title)

        # Moving sizing to auto to prevent over sizing
        self.figure.match_aspect = False
        self.figure.width_policy = "fixed"
        self.figure.height_policy = "fixed"

        save(self.figure)

        self.figure.match_aspect = True
        self.figure.width_policy = "max"
        self.figure.height_policy = "max"

    def _polygons_to_coords(self, polygons: List[PolygonElement]) -> Tuple[List[List[Tuple[List[float]]]], List[List[Tuple[List[float]]]]]:
        xs_dict = [
            [
                {
                    "exterior": p.exterior_polygon.x_coords,
                    "holes": [h.x_coords for h in p.holes],
                }
            ]
            for p in polygons
        ]

        ys_dict = [
            [
                {
                    "exterior": p.exterior_polygon.y_coords,
                    "holes": [h.y_coords for h in p.holes],
                }
            ]
            for p in polygons
        ]

        xs: List[List[Tuple[List[float]]]] = [
            [[p["exterior"], *p["holes"]] for p in mp] for mp in xs_dict
        ]
        ys: List[List[Tuple[List[float]]]] = [
            [[p["exterior"], *p["holes"]] for p in mp] for mp in ys_dict
        ]

        return xs, ys

    def send_event(self, callback):
        # If the mouse is hovered while a range update triggered update is done, the self.source_polygons.data length is updated faster than the data coming from the mouse.
        #   The value of self.source_mouse.data["index"][0] will be greater than the polygon length. In this case, the callback is not called.
        if "index" in self.source_mouse.data and CELL_NAMES in self.source_polygons.data and \
            int(self.source_mouse.data["index"][0]) < len(self.source_polygons.data[CELL_NAMES]):
            callback(
                screen_location=(
                    self.source_mouse.data["sx"][0],
                    self.source_mouse.data["sy"][0]
                ),
                space_location=(
                    self.source_mouse.data["x"][0], 
                    self.source_mouse.data["y"][0], 
                    self.source_mouse.data["z"][0]
                ), 
                cell_id=self.source_polygons.data[CELL_NAMES][int(self.source_mouse.data["index"][0])]
            )


    def provide_on_mouse_move_callback(self, callback:Callable):
        """Stores a function to call everytime the user moves the mouse on the plot. 
        Functions arguments are location, cell_id.

        Parameters
        ----------
        callback : Callable
            Function to call.
        """
        super().provide_on_mouse_move_callback(callback)

        self.figure.on_event(bokeh.events.MouseMove, functools.partial(self.send_event, callback))

    def provide_on_clic_callback(self, callback:Callable):
        """Stores a function to call everytime the user clics on the plot. 
        Functions arguments are location, cell_id.

        Parameters
        ----------
        callback : Callable
            Function to call.
        """
        super().provide_on_clic_callback(callback)

        self.figure.on_event(bokeh.events.Tap, functools.partial(self.send_event, callback))
        # self.figure.add_tools(TapTool())

    def set_axes(self, u:Tuple[float, float, float], v:Tuple[float, float, float], w:float):
        """Stores the u v axes of the current plot

        Parameters
        ----------
        u : Tuple[float, float, float]
            Horizontal axis direction vector
        v : Tuple[float, float, float]
            Vertical axis direction vector
        w : float
            Normal vector coordinate
        """
        w_vector = np.cross(np.array(u), np.array(v))
        
        new_data = self.source_coordinates.data.copy()
        new_data["u0"] = [u[0]]
        new_data["u1"] = [u[1]]
        new_data["u2"] = [u[2]]
        new_data["v0"] = [v[0]]
        new_data["v1"] = [v[1]]
        new_data["v2"] = [v[2]]
        new_data["w0"] = [w_vector[0]]
        new_data["w1"] = [w_vector[1]]
        new_data["w2"] = [w_vector[2]]
        new_data["w"]  = [w]

        self.source_coordinates.update(data = new_data)
