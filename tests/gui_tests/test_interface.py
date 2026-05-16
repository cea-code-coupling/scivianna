from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np
import multiprocessing as mp
import panel as pn
import panel_material_ui as pmui
from bokeh.plotting import curdoc

import scivianna
from scivianna.panel.panel_2d import Panel2D
from scivianna.slave import ComputeSlave
from scivianna.plotter_2d.generic_plotter import Plotter2D

from scivianna.extension.extension import Extension
import scivianna.icon
from scivianna.data.data2d import Data2D
from scivianna.interface.generic_interface import Geometry2DPolygon, IcocoInterface
from scivianna.utils.polygonize_tools import PolygonElement, PolygonCoords
from scivianna.enums import GeometryType, VisualizationMode

from scivianna.constants import MESH, MATERIAL, GEOMETRY, CSV
from scivianna.constants import XS, YS, CELL_NAMES, COMPO_NAMES, COLORS, EDGE_COLORS, GEOMETRY, EDGE_ALPHA, FILL_ALPHA

with open(Path(scivianna.icon.__file__).parent / "salome.svg", "r") as f:
    icon_svg = f.read()


@dataclass
class FieldChangeEvent:
    """Event recorded when a field change callback is triggered."""
    field_name: str


@dataclass
class RangeChangeEvent:
    """Event recorded when a range change callback is triggered."""
    u_bounds: Tuple[float, float]
    v_bounds: Tuple[float, float]
    w_value: float


@dataclass
class FrameChangeEvent:
    """Event recorded when a frame change callback is triggered."""
    u_vector: Tuple[float, float, float]
    v_vector: Tuple[float, float, float]


class TestExtension(Extension):
    """Extension to load files and send them to the slave.

    This extension overrides all Extension callback methods to track
    when they are called by the Panel2D system. Tracking data is stored
    in instance variables for test verification.
    """

    def __init__(
        self,
        slave: "ComputeSlave",
        plotter: "Plotter2D",
        panel: "Panel2D"
    ):
        """Constructor of the extension, saves the slave and the panel

        Parameters
        ----------
        slave : ComputeSlave
            Slave computing the displayed data
        plotter : Plotter2D
            Figure plotter
        panel : Panel2D
            Panel to which the extension is attached
        """
        super().__init__(
            "TestExtension",
            icon_svg,
            slave,
            plotter,
            panel,
        )

        self.description = """
This extension allows defining the medcoupling field display parameters.
"""

        self.iconsize = "1.0em"

        # Tracking variables for callback invocations
        self._field_change_history: List[str] = []
        self._range_change_history: List[RangeChangeEvent] = []
        self._frame_change_history: List[FrameChangeEvent] = []
        self._updated_data_history: List[Data2D] = []

        self._on_field_change_called: bool = False
        self._on_range_change_called: bool = False
        self._on_frame_change_called: bool = False
        self._on_updated_data_called: bool = False

    def make_gui(self,) -> pn.viewable.Viewable:
        """Returns a panel viewable to display in the extension tab.

        Returns
        -------
        pn.viewable.Viewable
            Viewable to display in the extension tab
        """
        return pmui.Column(
            pmui.Typography("Test extension"),
            margin=0
        )

    def on_field_change(self, field_name: str):
        """Override to track field change events."""
        self._field_change_history.append(field_name)
        self._on_field_change_called = True

    def on_range_change(
        self,
        u_bounds: Tuple[float, float],
        v_bounds: Tuple[float, float],
        w_value: float,
    ):
        """Override to track range change events."""
        self._range_change_history.append(RangeChangeEvent(
            u_bounds=u_bounds, v_bounds=v_bounds, w_value=w_value
        ))
        self._on_range_change_called = True

    def on_frame_change(
        self,
        u_vector: Tuple[float, float, float],
        v_vector: Tuple[float, float, float],
    ):
        """Override to track frame change events."""
        self._frame_change_history.append(FrameChangeEvent(
            u_vector=u_vector, v_vector=v_vector
        ))
        self._on_frame_change_called = True

    def on_updated_data(self, data: Data2D):
        """Override to track updated data events."""
        self._updated_data_history.append(data)
        self._on_updated_data_called = True


class TestInterface(Geometry2DPolygon):
    geometry_type: GeometryType = GeometryType._3D_INFINITE
    extensions = [TestExtension]

    def __init__(self):
        self.data: Data2D = None
        self.file_path: Dict[str, str] = {}
        self.current_field = None

    def read_file(self, file_path: str, file_label: str):
        """Read a file and store its content in the interface

        Parameters
        ----------
        file_path : str
            File to read
        file_label : str
            Label to define the file type
        """
        self.file_path[file_label] = file_path

    def compute_2D_data(
        self,
        u: Tuple[float, float, float],
        v: Tuple[float, float, float],
        u_min: float,
        u_max: float,
        v_min: float,
        v_max: float,
        w_value: float,
        q_tasks: mp.Queue,
        options: Dict[str, Any],
    ) -> Tuple[Data2D, bool]:
        """Returns a list of polygons that defines the geometry in a given frame

        Parameters
        ----------
        u : Tuple[float, float, float]
            Horizontal coordinate director vector
        v : Tuple[float, float, float]
            Vertical coordinate director vector
        u_min : float
            Lower bound value along the u axis
        u_max : float
            Upper bound value along the u axis
        v_min : float
            Lower bound value along the v axis
        v_max : float
            Upper bound value along the v axis
        w_value : float
            Value along the u ^ v axis
        q_tasks : mp.Queue
            Queue from which get orders from the master.
        options : Dict[str, Any]
            Additional options for frame computation.

        Returns
        -------
        Data2D
            Geometry to display
        bool
            Were the polygons updated compared to the past call
        """
        last_frame_key = (*u, *v, w_value)
        if (self.data is not None) and (
            self.last_computed_frame == last_frame_key
        ):
            print("Skipping polygon computation.")
            return self.data, False

        v_offset = v[1] + 10*w_value
        polygons = [
            PolygonElement(
                exterior_polygon = PolygonCoords(
                    x_coords = [i, i, i+1, i+1],
                    y_coords = [0+v_offset, 1+v_offset, 1+v_offset, 0+v_offset]
                ),
                holes = [],
                cell_id = i
            )
            for i in range(3)
        ]

        print("Offset", v_offset)

        self.last_computed_frame = last_frame_key
        self.data = Data2D.from_polygon_list(polygons)
        return self.data, True

    def get_labels(
        self,
    ) -> List[str]:
        """Returns a list of fields names displayable with this interface

        Returns
        -------
        List[str]
            List of fields names
        """
        labels = [MESH,  MATERIAL, "VALUE"]
        return labels

    def get_value_dict(
        self, value_label: str, cells: List[Union[int, str]], options: Dict[str, Any]
    ) -> Dict[Union[int, str], str]:
        """Returns a cell name - field value map for a given field name

        Parameters
        ----------
        value_label : str
            Field name to get values from
        cells : List[Union[int,str]]
            List of cells names
        options : Dict[str, Any]
            Additional options for frame computation.

        Returns
        -------
        Dict[Union[int,str], str]
            Field value for each requested cell names
        """
        self.current_field = value_label
        if value_label == MESH:
            return {v: np.nan for v in cells}
        if value_label == MATERIAL:
            return {v: str(v) for v in cells}
        if value_label == "VALUE":
            return {v: float(v) for v in cells}

        raise NotImplementedError(
            f"The field {value_label} is not implemented, fields available : {self.get_labels()}"
        )

    def get_label_coloring_mode(self, label: str) -> VisualizationMode:
        """Returns wheter the given field is colored based on a string value or a float.

        Parameters
        ----------
        label : str
            Field to color name

        Returns
        -------
        VisualizationMode
            Coloring mode
        """
        if label == MESH:
            return VisualizationMode.NONE
        if label == MATERIAL:
            return VisualizationMode.FROM_STRING

        return VisualizationMode.FROM_VALUE

    def get_file_input_list(self) -> List[Tuple[str, str]]:
        """Returns a list of file label and its description for the GUI

        Returns
        -------
        List[Tuple[str, str]]
            List of (file label, description)
        """
        return [(GEOMETRY, "MED file."), (CSV, "CSV result file.")]
    
    def get_info(self,):
        return self.data, self.last_computed_frame, self.file_path, self.current_field

def make_panel_2d():
    slave = ComputeSlave(TestInterface)
    panel = Panel2D(slave)

    return panel, {
        e.__class__: e for e in panel.extensions
    }

def get_polygons(panel: Panel2D):
    return [
        panel.plotter.source_polygons.data[XS],
        panel.plotter.source_polygons.data[YS],
    ]

def get_colors(panel: Panel2D):
    return panel.plotter.source_polygons.data[COLORS]

def get_edge_colors(panel: Panel2D):
    return panel.plotter.source_polygons.data[EDGE_COLORS]

def get_cell_ids(panel: Panel2D):
    return panel.plotter.source_polygons.data[CELL_NAMES]

def get_cell_values(panel: Panel2D):
    return panel.plotter.source_polygons.data[COMPO_NAMES]

def test_build_panel():
    make_panel_2d()
    assert True