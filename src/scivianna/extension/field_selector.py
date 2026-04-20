
from typing import Any, Dict, TYPE_CHECKING
import numpy as np
import panel as pn
import panel_material_ui as pmui
import time

from scivianna.constants import OUTSIDE
from scivianna.data.data2d import Data2D
from scivianna.enums import VisualizationMode
from scivianna.extension.extension import Extension
from scivianna.plotter_2d.generic_plotter import Plotter2D
from scivianna.slave import ComputeSlave
from scivianna.utils.color_tools import beautiful_color_maps, get_edges_colors, interpolate_cmap_at_values

if TYPE_CHECKING:
    from scivianna.panel.visualisation_panel import VisualizationPanel

profile_time = False


def set_colors_list(
    data: Data2D,
    slave: ComputeSlave,
    coloring_label: str,
    color_map: str,
    center_colormap_on_zero: bool,
    options: Dict[str, Any],
):
    """Sets in a Data2D the list of colors for a field per polygon.

    Parameters
    ----------
    data : Data2D
        Geometry data
    slave : ComputeSlave
        Slave to which request values
    coloring_label : str
        Field to color
    color_map : str
        Colormap in which select colors
    center_colormap_on_zero : bool
        Center the color map on zero
    options : Dict[str, Any]
        Plot extra options

    Raises
    ------
    NotImplementedError
        The field visualisation mode is not implemented.
    """
    if profile_time:
        start_time = time.time()

    coloring_mode = slave.get_label_coloring_mode(coloring_label)

    cell_values = data.cell_values

    if profile_time:
        print(f"get color list prepare time {time.time() - start_time}")
        start_time = time.time()

    if coloring_mode == VisualizationMode.FROM_STRING:
        """
        A random color is given for each string value.
        """
        sorted_values = np.sort(np.unique(list(cell_values)))
        map_to = np.array([hash(c) % 255 for c in sorted_values]) / 255

        value_list = np.array(cell_values)

        _, inv = np.unique(value_list, return_inverse=True)

        cell_colors = interpolate_cmap_at_values(
            color_map, map_to[inv].astype(float)
        )

        if OUTSIDE in data.cell_ids:
            for index_ in np.where(data.cell_ids == OUTSIDE):
                cell_colors[index_] = (255, 255, 255, 0)

    elif coloring_mode == VisualizationMode.FROM_VALUE:
        """
        The color is got from a color map set in the range (-max, max)
        """
        normalized_cell_values = np.array(cell_values).astype(float)
        no_nan_values = normalized_cell_values[~np.isnan(normalized_cell_values)]

        if profile_time:
            print(f"extracting no nan {time.time() - start_time}")
            start_time = time.time()

        if center_colormap_on_zero:
            if (
                len(no_nan_values) == 0 or max(abs(no_nan_values.min()), no_nan_values.max()) == 0.0
            ):
                minmax = 1.0
            else:
                minmax = max(abs(no_nan_values.min()), no_nan_values.max())

            normalized_cell_values = (normalized_cell_values + minmax) / (2 * minmax)
        else:
            if (
                len(no_nan_values) == 0 or max(abs(no_nan_values.min()), no_nan_values.max()) == 0.0
            ):
                minmax = 1.0
                min_val = 0.0
            elif no_nan_values.min() == no_nan_values.max():
                minmax = 1.0
                min_val = no_nan_values.min()
            else:
                minmax = no_nan_values.max() - no_nan_values.min()
                min_val = no_nan_values.min()

            normalized_cell_values = (normalized_cell_values - min_val) / minmax

        if profile_time:
            print(f"Rescaling data {time.time() - start_time}")
            start_time = time.time()

        cell_colors = interpolate_cmap_at_values(
            color_map, normalized_cell_values
        )

        if profile_time:
            print(f"Extracting colors {time.time() - start_time}")
            start_time = time.time()

        # Changing the main color from black to gray in case of Nan
        for c in range(len(cell_colors)):
            if cell_colors[c, 3] == 0.0:
                cell_colors[c] = (200, 200, 200, 0)

        if profile_time:
            print(f"Fixing nans {time.time() - start_time}")
            start_time = time.time()

    elif coloring_mode == VisualizationMode.NONE:
        """
        No color, mesh displayed only
        """
        cell_colors = np.array([(200, 200, 200, 0)] * (len(data.cell_ids)))
    else:
        raise NotImplementedError(
            f"Visualization mode {coloring_mode} not implemented."
        )

    data.cell_colors = cell_colors

    edge_colors = get_edges_colors(cell_colors)

    if not isinstance(cell_values[0], str):
        edge_colors[:, 3] = np.where(np.isnan(np.array(cell_values)), 255, edge_colors[:, 3])

    data.cell_edge_colors = edge_colors


class FieldSelector(Extension):
    """ Extension used to select the displayed field and edit its colors.
    """

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
        super().__init__(
            "Color map",
            "palette",
            slave,
            plotter,
            panel,
        )

        self.description = """
The color map extension lets you decide which field is being displayed on the cells, and what colorbar is used.

If a color bar is used, you can decide to center it on zero.
"""

        fields_list = self.slave.get_labels()
        self.field_color_selector = pmui.Select(
            name="Color field",
            options=fields_list,
            value=fields_list[0],
            width=280
        )

        self.field_color_selector.param.watch(self.trigger_field_change, "value")
        self.panel.param.watch(self.receive_colormap_change, "colormap")

        self.color_map_selector = pn.widgets.ColorMap(
            options=beautiful_color_maps,
            swatch_width=60,
            width_policy='max'
        )

        self.color_map_selector.width = self.color_map_selector.height
        self.center_colormap_on_zero_tick = pn.widgets.Checkbox(
            name="Center color map on zero.", value=False,
            visible=slave.get_label_coloring_mode(self.field_color_selector.value) == VisualizationMode.FROM_VALUE,
        )

        self.color_map_selector.value_name = "BuRd"
        self.color_map_selector.value = beautiful_color_maps["BuRd"]

        self.color_map_selector.param.watch(self.trigger_colormap_change, "value")
        self.center_colormap_on_zero_tick.param.watch(self.trigger_update, "value")

    def trigger_field_change(self, *args, **kwargs):
        """Trigger a field change in the visualization panel
        """
        self.center_colormap_on_zero_tick.visible = self.slave.get_label_coloring_mode(self.field_color_selector.value) == VisualizationMode.FROM_VALUE
        self.panel.set_field(self.field_color_selector.value)

    def receive_colormap_change(self, *args, **kwargs):
        """Receive a field change from the visualization panel
        """
        if self.panel.colormap != self.color_map_selector.value_name:
            self.color_map_selector.value_name = self.panel.colormap

    def trigger_colormap_change(self, *args, **kwargs):
        """Trigger a field change in the visualization panel
        """
        self.panel.set_colormap(self.color_map_selector.value_name)
        self.panel.recompute()

    def trigger_update(self, *args, **kwargs):
        """Trigger a color map change in the visualization panel
        """
        self.panel.recompute()

    @pn.io.hold()
    def on_file_load(self, file_path: str, file_key: str):
        """Function called when the user requests a change of field on the GUI

        Parameters
        ----------
        file_path : str
            Path of the loaded file
        file_key : str
            Key associated to the loaded file
        """
        self.field_color_selector.options = list(
            self.slave.get_labels()
        )
        self.field_color_selector.value = self.field_color_selector.options[0]

    def on_updated_data(self, data: Data2D):
        """Function called when the displayed data is being updated. Extension can edit the data on its way to the plotter.

        Parameters
        ----------
        data : Data2D
            Data to display
        """
        set_colors_list(
            data,
            self.slave,
            self.field_color_selector.value,
            self.color_map_selector.value_name,
            self.center_colormap_on_zero_tick.value,
            {},
        )

    def make_gui(self,) -> pn.viewable.Viewable:
        """Returns a panel viewable to display in the extension tab.

        Returns
        -------
        pn.viewable.Viewable
            Viewable to display in the extension tab
        """
        return pn.Column(
            self.field_color_selector,
            self.color_map_selector,
            self.center_colormap_on_zero_tick,
        )

    def on_field_change(self, field_name: str):
        """Function called when the user requests a displayed field change

        Parameters
        ----------
        field_name : str
            Name of the new displayed field
        """
        self.field_color_selector.value = field_name
