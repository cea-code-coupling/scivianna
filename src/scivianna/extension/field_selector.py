
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
from scivianna.utils.color_tools import beautiful_color_maps, get_edges_colors, interpolate_cmap_at_values

if TYPE_CHECKING:
    from scivianna.panel.visualisation_panel import VisualizationPanel
    from scivianna.slave import ComputeSlave

profile_time = False


def set_colors_list(
    data: Data2D,
    slave: "ComputeSlave",
    coloring_label: str,
    color_map: str,
    center_colormap_on_zero: bool,
    options: Dict[str, Any],
    min_value: float = None,
    max_value: float = None,
    offset: int = 20
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
    min_value : float
        Force scale minimum value
    max_value : float
        Force scale maximum value
    offset : int
        To how many decrease the edge color

    Raises
    ------
    NotImplementedError
        The field visualisation mode is not implemented.
    """
    if profile_time:
        start_time = time.time()

    coloring_mode = slave.get_label_coloring_mode(coloring_label)

    cell_values = data.cell_values

    min_val = None
    minmax = None

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

            if min_value is not None:
                min_val = min_value
            if max_value is not None:
                minmax = max_value - min_val

            if minmax == 0:
                minmax = 1

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

    edge_colors = get_edges_colors(cell_colors, offset=offset)

    if len(cell_values) > 0:
        if not isinstance(cell_values[0], str):
            edge_colors[:, 3] = np.where(np.isnan(np.array(cell_values)), 255, edge_colors[:, 3])
    else:
        data.cell_edge_colors = data.cell_colors

    data.cell_edge_colors = edge_colors

    return min_val, minmax


class FieldSelector(Extension):
    """ Extension used to select the displayed field and edit its colors.
    """

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
            label="Color field",
            options=fields_list,
            value=fields_list[0],
            width=280,
            searchable = True
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
        
        self.min_value = pmui.FloatInput(label="Min value", width = 220, disabled=True)
        self.max_value = pmui.FloatInput(label="Max value", width = 220, disabled=True)

        self.min_activated = pmui.Checkbox(value=False, description="Force minimum value.")
        self.max_activated = pmui.Checkbox(value=False, description="Force maximum value.")
        
        self.force_range_column = pmui.Column(
            pmui.Divider(),
            pmui.Typography("### Force value range"),
            pmui.Row(
                self.min_value, self.min_activated
            ),
            pmui.Row(
                self.max_value, self.max_activated
            ),
        )

        self.update_range_visibility()

        self.min_activated.param.watch(self.enable_disable_bounds, "value")
        self.max_activated.param.watch(self.enable_disable_bounds, "value")

        self.min_value.param.watch(self.trigger_update, "value")
        self.max_value.param.watch(self.trigger_update, "value")
        
        self.edge_offset = pmui.IntInput(
            label="Edge color offset", 
            value = -20, 
            start=-255, 
            end=255, 
            step=10, 
            description="Offset added to the edges color, if 255, everything will be white, if -255, everything will be black.",
            width=280,
        )
        self.edge_offset_column = pmui.Column(
            pmui.Divider(),
            pmui.Typography("### Edge color offset"),
            self.edge_offset
        )
        self.edge_offset.param.watch(self.trigger_update, "value")


    def update_range_visibility(self, *args, **kwargs):
        """Updates range visibility based on the field type
        """
        print(f"Setting visible for field : {self.field_color_selector.value}")
        self.force_range_column.visible = self.slave.get_label_coloring_mode(self.field_color_selector.value) == VisualizationMode.FROM_VALUE

    def enable_disable_bounds(self, *args, **kwargs):
        """Enable/disable min max widgets
        """
        self.min_value.disabled = not self.min_activated.value
        self.max_value.disabled = not self.max_activated.value

        self.trigger_update()

    def trigger_field_change(self, *args, **kwargs):
        """Trigger a field change in the visualization panel
        """
        if self._restoring:
            return
        self.center_colormap_on_zero_tick.visible = self.slave.get_label_coloring_mode(self.field_color_selector.value) == VisualizationMode.FROM_VALUE
        self.panel.set_field(self.field_color_selector.value)

    def receive_colormap_change(self, *args, **kwargs):
        """Receive a field change from the visualization panel
        """
        if self._restoring:
            return
        if self.panel.colormap != self.color_map_selector.value_name:
            self.color_map_selector.value_name = self.panel.colormap

    def trigger_colormap_change(self, *args, **kwargs):
        """Trigger a field change in the visualization panel
        """
        if self._restoring:
            return
        self.panel.set_colormap(self.color_map_selector.value_name)
        self.panel.recompute()

    def trigger_update(self, *args, **kwargs):
        """Trigger a color map change in the visualization panel
        """
        if self._restoring:
            return
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
        min_val, minmax = set_colors_list(
            data,
            self.slave,
            self.field_color_selector.value,
            self.color_map_selector.value_name,
            False if any([self.min_activated.value, self.max_activated.value]) else self.center_colormap_on_zero_tick.value,
            {},
            min_value=self.min_value.value if self.min_activated.value else None,
            max_value=self.max_value.value if self.max_activated.value else None,
            offset = self.edge_offset.value
        )
        if min_val is not None and minmax is not None:
            self.panel.plotter.update_colorbar(True, value_range=(min_val, min_val + minmax))

            self._restoring = True
            if not self.min_activated.value:
                self.min_value.value = min_val
            if not self.max_activated.value:
                self.max_value.value = min_val + minmax
            self._restoring = False

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
            self.force_range_column,
            self.edge_offset_column
        )

    def on_field_change(self, field_name: str):
        """Function called when the user requests a displayed field change

        Parameters
        ----------
        field_name : str
            Name of the new displayed field
        """
        self.field_color_selector.value = field_name
        self.update_range_visibility()

    def to_json(self) -> dict:
        """Returns a dictionary with the information required to rebuild the extension.

        Returns
        -------
        dict
            Information dictionary
        """
        return {
            "field": self.field_color_selector.value,
            "colormap": self.color_map_selector.value_name,
            "center_colormap_on_zero": self.center_colormap_on_zero_tick.value,
            "min_value": self.min_value.value,
            "max_value": self.max_value.value,
            "min_activated": self.min_activated.value,
            "max_activated": self.max_activated.value,
            "edge_offset": self.edge_offset.value,
        }

    @classmethod
    def from_json(cls, extension: "FieldSelector", info_dict: dict) -> "FieldSelector":
        """Restores the extension from its information dict.

        Parameters
        ----------
        extension : FieldSelector
            Extension instance to restore
        info_dict : dict
            Dictionary containing extension state information

        Returns
        -------
        FieldSelector
            Restored extension
        """
        extension._restoring = True

        try:
            if info_dict.get("field") is not None:
                extension.field_color_selector.value = info_dict["field"]

            extension.color_map_selector.value_name = info_dict.get("colormap", "BuRd")
            extension.center_colormap_on_zero_tick.value = info_dict.get("center_colormap_on_zero", False)

            # Restore force range bounds
            if "min_value" in info_dict:
                extension.min_value.value = info_dict["min_value"]
            if "max_value" in info_dict:
                extension.max_value.value = info_dict["max_value"]
            if "min_activated" in info_dict:
                extension.min_activated.value = info_dict["min_activated"]
            if "max_activated" in info_dict:
                extension.max_activated.value = info_dict["max_activated"]

            # Restore edge offset
            if "edge_offset" in info_dict:
                extension.edge_offset.value = info_dict["edge_offset"]

            # Update UI state for bounds enabled/disabled
            extension.enable_disable_bounds()

            # Update range visibility based on current field
            extension.update_range_visibility()
        finally:
            extension._restoring = False

        return extension

    def on_coupling_update(self):
        """Function called at the end of a coupling time step
        """
        labels = self.slave.get_labels()
        if set(labels) != set(self.field_color_selector.options):
            self.field_color_selector.options = list(
                self.slave.get_labels()
            )