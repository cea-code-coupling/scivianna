
from typing import Dict, List, Tuple
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.models.renderers import GlyphRenderer
from bokeh.models.annotations import Legend, LegendItem
from bokeh.palettes import Viridis11 as palette
from bokeh.models import HoverTool, LinearScale, LogScale

import numpy as np
import pandas as pd
import panel as pn

from scivianna.plotter_1d.generic_plotter import Plotter1D


class BokehPlotter1D(Plotter1D):
    """Unfinished 1D plotter to get the coupling working"""

    def __init__(
        self,
    ):
        self.source_data_dict:Dict[str, ColumnDataSource] = {}
        """Dictionnary of x, y ColumnDataSource containing the data to plot"""
        self.line_dict:Dict[str, GlyphRenderer]= {}
        """Dictionnary of plot GlyphRenderers"""

        self.fig = figure(
            name="plot",
            sizing_mode="stretch_both",
        )

        # Reset ranges when plots are hidden
        self.fig.x_range.only_visible = True
        self.fig.y_range.only_visible = True

        """Bokeh figure"""
        self.hover = HoverTool(
            tooltips="$name: (@x, @y)"
        )
        """Tool defining the hovered label"""
        self.fig.add_tools(self.hover)

        self.fig_pane = pn.pane.Bokeh(
            self.fig,
            name="Plot",
            sizing_mode="stretch_both",
            margin=0,
            styles={"border": "2px solid lightgray"},
        )
        """panel Bokeh pane"""
        self.visible:List[str] = []
        """List of visible plots"""

    def plot(
        self,
        name:str,
        serie: pd.Series
    ):
        """Adds a new plot to the figure from a set of polygons

        Parameters
        ----------
        name : str
            Plot name
        serie : pd.Series
            Sata to plot
        """
        print(f"Plotting {name}")
        if len(serie.values) == 2 and list(serie.values) == ["min", "max"]:
            serie = pd.Series(list(self.get_y_bounds()), index=serie.index, name=serie.name)
        if len(serie.index) == 2 and list(serie.index) == ["min", "max"]:
            serie = pd.Series(serie.values, index=list(self.get_x_bounds()), name=serie.name)

        self.source_data_dict[name] = ColumnDataSource(
                                            {
                                                "x": serie.index.tolist(),
                                                "y": serie.values.tolist(),
                                            }
                                        )
        self.line_dict[name] = self.fig.line(x="x",
                        y="y",
                        line_width=2,
                        source=self.source_data_dict[name],
                        legend_label=str(name),
                        name=str(name).replace(" ", "_"))



    def update_plot(
        self,
        name:str,
        serie: pd.Series
    ):
        """Updates plot to the figure

        Parameters
        ----------
        name : str
            Plot name
        serie : pd.Series
            Sata to plot
        """
        # If on one of the axes, the value is ["min", "max"], a line will be plotted along the displayed range
        if len(serie.values) == 2 and list(serie.values) == ["min", "max"]:
            serie = pd.Series(list(self.get_y_bounds()), index=serie.index, name=serie.name)
        if len(serie.index) == 2 and list(serie.index) == ["min", "max"]:
            serie = pd.Series(serie.values, index=list(self.get_x_bounds()), name=serie.name)

        if name in self.source_data_dict:
            self.source_data_dict[name].update(data={
                "x":serie.index.tolist(),
                "y":serie.values.tolist(),
            })
        else:
            self.plot(name, serie)

    @pn.io.hold()
    def set_visible(
        self,
        names:List[str],
    ):
        """Updates the visible plots in the figure

        Parameters
        ----------
        names : List[str]
            List of displayed plots
        """
        for glyph_name in self.line_dict:
            self.line_dict[glyph_name].visible = glyph_name in names
            if glyph_name in names:
                self.line_dict[glyph_name].glyph.line_color = palette[
                    (0 if len(names)== 1 else int((names.index(glyph_name)/(len(names)-1))*(len(palette)-1)))
                ]

        l:Legend
        li:LegendItem
        for l in self.fig.legend:
            for li in l.items:
                if li.label.value in names:
                    li.visible = True
                else:
                    li.visible = False

        self.hover.update(renderers = [self.line_dict[glyph_name]
                                        for glyph_name in names
                                        if glyph_name in self.line_dict])
        self.visible = names

    def _disable_interactions(self, val: bool):
        """Enable/disable the plot interactions

        Parameters
        ----------
        val : bool
            enable or disable the plot
        """
        pass

    def make_panel(self,) -> pn.viewable.Viewable:
        """Returns the panel to display in the layout

        Returns
        -------
        pn.viewable.Viewable
            Panel to display
        """
        return self.fig_pane

    def get_y_bounds(self,) -> Tuple[float, float]:
        """Returns the bounds of the displayed data along the Y axis

        Returns
        -------
        Tuple[float, float]
            Displayed data Y bounds
        """
        min_val = np.nan
        max_val = np.nan

        y_mins = [np.nanmin(self.source_data_dict[name].data["y"])
                    for name in self.visible
                    if (name in self.source_data_dict) and not (
                            isinstance(self.source_data_dict[name].data["y"][0], str)
                            or np.count_nonzero(~np.isnan(self.source_data_dict[name].data["y"])) == 0
                        )]
        y_maxs = [np.nanmax(self.source_data_dict[name].data["y"])
                    for name in self.visible
                    if (name in self.source_data_dict) and not (
                            isinstance(self.source_data_dict[name].data["y"][0], str)
                            or np.count_nonzero(~np.isnan(self.source_data_dict[name].data["y"])) == 0
                        )]

        if len(y_mins) != 0:
            min_val = np.nanmin(y_mins)
            max_val = np.nanmax(y_maxs)

        return min_val, max_val

    def get_x_bounds(self,) -> Tuple[float, float]:
        """Returns the bounds of the displayed data along the X axis

        Returns
        -------
        Tuple[float, float]
            Displayed data X bounds
        """
        min_val = np.nan
        max_val = np.nan

        x_mins = [np.nanmin(self.source_data_dict[name].data["x"])
                    for name in self.visible
                    if (name in self.source_data_dict) and not (
                            isinstance(self.source_data_dict[name].data["x"][0], str)
                            or np.count_nonzero(~np.isnan(self.source_data_dict[name].data["x"])) == 0
                        )]
        x_maxs = [np.nanmax(self.source_data_dict[name].data["x"])
                    for name in self.visible
                    if (name in self.source_data_dict) and not (
                            isinstance(self.source_data_dict[name].data["x"][0], str)
                            or np.count_nonzero(~np.isnan(self.source_data_dict[name].data["x"])) == 0
                        )]

        if len(x_mins) != 0:
            min_val = np.nanmin(x_mins)
            max_val = np.nanmax(x_maxs)

        return min_val, max_val

    def set_x_scale(self, scale: str):
        """Sets the X axis scale to either log or lin

        Parameters
        ----------
        scale : str
            Scale to set
        """
        self.fig.x_scale = LogScale() if scale == "log" else LinearScale()

    def set_y_scale(self, scale: str):
        """Sets the Y axis scale to either log or lin

        Parameters
        ----------
        scale : str
            Scale to set
        """
        self.fig.y_scale = LogScale() if scale == "log" else LinearScale()
