from typing import Callable, Dict, List, Tuple, Type, Union
import panel as pn
import pandas as pd

from scivianna.extension.extension import Extension
from scivianna.panel.visualisation_panel import VisualizationPanel
from scivianna.slave import ComputeSlave
from scivianna.extension.file_loader import FileLoader
from scivianna.plotter_dataframe.dataframe_plotter import DataframePlotter


default_extensions = [FileLoader]

class PanelDataFrame(VisualizationPanel):
    """DataFrame visualisation panel associated to a DataFrameInterface code."""

    cell_id: str = None
    """Cell identifier passed to get_dataframe"""
    origin: Tuple[float, float, float] = None
    """Origin position passed to get_dataframe"""

    def __init__(
        self,
        slave: ComputeSlave,
        name: str = "",
        extensions: List[Extension] = default_extensions,
    ):
        """Visualization panel constructor

        Parameters
        ----------
        slave : ComputeSlave
            ComputeSlave object to which request the plots.
        name : str
            Name of the panel.
        extensions : List[Extension]
            List of extensions loaded with the visualizer.
        """
        self.plotter = DataframePlotter()

        self.slave = slave
        self.slave.allow_errors = True
        self.panel_name = name
        self.copy_index = 1

        self._restoring = False

        super().__init__(slave, name, extensions.copy())

        self.periodic_recompute_added = False
        self.marked_to_recompute = False

        # Load initial data
        self.recompute()

    #
    #   VisualizationPanel API overrides (no-ops or simple implementations)
    #
    def recompute(self, *args, **kwargs):
        """Recomputes the DataFrame by fetching data from the interface."""
        try:
            # Build options dynamically from extensions
            options = {key: value for opt in [
                e.provide_options() for e in self.extensions
            ] for key, value in opt.items()}

            df = self.slave.get_dataframe(
                cell_id=self.cell_id,
                origin=self.origin,
                options=options,
            )
            if df is not None:
                self.plotter.update_data(df)
        except Exception as e:
            pn.state.notifications.error(
                f"Error building the dataframe, got {e}"
            )

    def provide_field_change_callback(self, callback: Callable):
        """Stores a function to call everytime the displayed field is changed."""
        self.field_change_callback = callback

    def recompute_at(self, position: Tuple[float, float, float], cell_id: str):
        """Not applicable for DataFrame panel."""
        self.cell_id = cell_id
        self.origin = position
        self.recompute()

    def set_field(self, field_name: str):
        """Not applicable for DataFrame panel."""
        pass

    def to_json(self) -> Dict:
        """Returns a dictionary with the information required to rebuild the visualization panel."""
        # Convert DataFrame to dict for serialization if present
        dataframe_dict = None
        df = self.plotter.get_data()
        if df is not None:
            dataframe_dict = df.to_dict(orient="list")

        return {
            "name": self.panel_name,
            "cell_id": self.cell_id,
            "origin": self.origin,
            "dataframe": dataframe_dict,
        }

    @classmethod
    def from_json(
        cls,
        info_dict: Dict,
        slave: ComputeSlave,
        extensions: Union[List[Extension], List[Tuple[Type[Extension], dict]]] = []
    ) -> "PanelDataFrame":
        """Restores the visualization panel from its information dict."""
        panel = PanelDataFrame(
            slave,
            info_dict.get("name", ""),
            extensions,
        )
        panel._restoring = True
        panel.panel_name = info_dict.get("name", "")
        panel.cell_id = info_dict.get("cell_id")
        panel.origin = info_dict.get("origin")

        # Restore dataframe if present
        dataframe_dict = info_dict.get("dataframe")
        if dataframe_dict is not None:
            panel.plotter.update_data(pd.DataFrame(dataframe_dict))
        else:
            # Trigger a recompute to fetch fresh data
            panel.recompute()

        panel._restoring = False
        return panel

    @pn.io.hold()
    def duplicate(self, keep_name: bool = False) -> "PanelDataFrame":
        """Get a copy of the panel. A panel of the same type is generated, but a new slave process is created."""
        panel = PanelDataFrame(
            self.slave.duplicate(),
            self.get_new_name(),
            extensions=[e for e in self.extension_classes]
        )
        # Restore panel state
        panel.cell_id = self.cell_id
        panel.origin = self.origin
        return panel

    def trigger_on_file_load(self, file_path: str, file_label: str):
        for extension in self.extensions:
            extension.on_file_load(file_path, file_label)
