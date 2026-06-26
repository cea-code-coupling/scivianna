from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import panel as pn

from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    GlyphRenderer
)
from bokeh.palettes import Viridis11 as palette
from bokeh.plotting import figure

from scivianna.plotter_1d.generic_plotter import Plotter1D


class BokehPlotter1D(Plotter1D):
    """1D Bokeh plotter."""

    def __init__(self):

        self.source_data_dict: Dict[str, ColumnDataSource] = {}
        """Dictionnary of x, y ColumnDataSource containing the data to plot"""

        self.line_dict: Dict[str, GlyphRenderer] = {}
        """Dictionnary of plot GlyphRenderers"""
        self.visible: List[str] = []
        """List of visible plots"""

        self.x_scale_type = "linear"
        self.y_scale_type = "linear"

        self._create_figure()

        self.fig_pane = pn.pane.Bokeh(
            self.fig,
            name="Plot",
            sizing_mode="stretch_both",
            margin=0,
            styles={"border": "2px solid lightgray"},
        )

    # -------------------------------------------------------------------------
    # Figure construction
    # -------------------------------------------------------------------------

    def _create_figure(self):

        self.fig = figure(
            name="plot",
            sizing_mode="stretch_both",
            x_axis_type=self.x_scale_type,
            y_axis_type=self.y_scale_type,
        )

        # Reset ranges when plots are hidden
        self.fig.x_range.only_visible = True
        self.fig.y_range.only_visible = True

        self.hover = HoverTool(
            tooltips="$name: (@x, @y)"
        )
        """Tool defining the hovered label"""
        self.fig.add_tools(self.hover)

    def _add_renderer(self, name: str):

        renderer = self.fig.line(
            x="x",
            y="y",
            source=self.source_data_dict[name],
            line_width=2,
            legend_label=name,
            name=name.replace(" ", "_"),
        )

        self.line_dict[name] = renderer

    def _rebuild_figure(self):
        """Recreate the Bokeh figure while preserving data."""

        self._create_figure()

        self.line_dict = {}

        for name in self.source_data_dict:
            self._add_renderer(name)

        self.set_visible(self.visible)

        self.fig_pane.object = self.fig

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------

    def _normalize_series(self, serie: pd.Series) -> pd.Series:

        if len(serie.values) == 2 and list(serie.values) == ["min", "max"]:
            serie = pd.Series(
                list(self.get_y_bounds()),
                index=serie.index,
                name=serie.name,
            )

        if len(serie.index) == 2 and list(serie.index) == ["min", "max"]:
            serie = pd.Series(
                serie.values,
                index=list(self.get_x_bounds()),
                name=serie.name,
            )

        return serie

    def plot(self, name: str, serie: pd.Series):
        """Adds a new plot to the figure from a set of polygons

        Parameters
        ----------
        name : str
            Plot name
        serie : pd.Series
            Sata to plot
        """

        serie = self._normalize_series(serie)

        self.source_data_dict[name] = ColumnDataSource(
            dict(
                x=serie.index.tolist(),
                y=serie.values.tolist(),
            )
        )

        self._add_renderer(name)

    def update_plot(self, name: str, serie: pd.Series):
        """Updates plot to the figure

        Parameters
        ----------
        name : str
            Plot name
        serie : pd.Series
            Sata to plot
        """

        serie = self._normalize_series(serie)

        if name not in self.source_data_dict:
            self.plot(name, serie)
            return

        self.source_data_dict[name].data = dict(
            x=serie.index.tolist(),
            y=serie.values.tolist(),
        )

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
        for glyph_name, renderer in self.line_dict.items():

            renderer.visible = glyph_name in names

            if glyph_name in names:
                renderer.glyph.line_color = palette[
                    0 if len(names) == 1 else int(
                        names.index(glyph_name)
                        * (len(palette) - 1)
                        / (len(names) - 1)
                    )
                ]

        for legend in self.fig.legend:
            for item in legend.items:
                item.visible = item.label.value in names

        self.hover.renderers = [
            self.line_dict[name]
            for name in names
            if name in self.line_dict
        ]

        self.visible = names

    def make_panel(self):
        """Returns the panel to display in the layout

        Returns
        -------
        pn.viewable.Viewable
            Panel to display
        """
        return self.fig_pane

    def get_y_bounds(self) -> Tuple[float, float]:
        """Returns the bounds of the displayed data along the Y axis

        Returns
        -------
        Tuple[float, float]
            Displayed data Y bounds
        """

        mins = []
        maxs = []

        for name in self.visible:

            if name not in self.source_data_dict:
                continue

            y = np.asarray(self.source_data_dict[name].data["y"])

            if len(y) == 0:
                continue

            if isinstance(y[0], str):
                continue

            if np.count_nonzero(~np.isnan(y)) == 0:
                continue

            mins.append(np.nanmin(y))
            maxs.append(np.nanmax(y))

        if not mins:
            return np.nan, np.nan

        return np.nanmin(mins), np.nanmax(maxs)

    def get_x_bounds(self) -> Tuple[float, float]:
        """Returns the bounds of the displayed data along the X axis

        Returns
        -------
        Tuple[float, float]
            Displayed data X bounds
        """

        mins = []
        maxs = []

        for name in self.visible:

            if name not in self.source_data_dict:
                continue

            x = np.asarray(self.source_data_dict[name].data["x"])

            if len(x) == 0:
                continue

            if isinstance(x[0], str):
                continue

            if np.count_nonzero(~np.isnan(x)) == 0:
                continue

            mins.append(np.nanmin(x))
            maxs.append(np.nanmax(x))

        if not mins:
            return np.nan, np.nan

        return np.nanmin(mins), np.nanmax(maxs)

    def set_x_scale(self, scale: str):
        """Sets the X axis scale to either log or lin

        Parameters
        ----------
        scale : str
            Scale to set
        """

        scale = "log" if scale == "log" else "linear"

        if scale == self.x_scale_type:
            return

        self.x_scale_type = scale
        self._rebuild_figure()

    def set_y_scale(self, scale: str):
        """Sets the Y axis scale to either log or lin

        Parameters
        ----------
        scale : str
            Scale to set
        """

        scale = "log" if scale == "log" else "linear"

        if scale == self.y_scale_type:
            return

        self.y_scale_type = scale
        self._rebuild_figure()


    def _disable_interactions(self, val: bool):
        """Enable/disable the plot interactions

        Parameters
        ----------
        val : bool
            enable or disable the plot
        """
        pass
