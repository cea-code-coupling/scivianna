from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, TYPE_CHECKING
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

with open(Path(scivianna.icon.__file__).parent / "salome.svg", "r") as f:
    icon_svg = f.read()


class TestExtension(Extension):
    """Extension to load files and send them to the slave."""

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
            "MEDCoupling",
            icon_svg,
            slave,
            plotter,
            panel,
        )

        self.description = """
This extension allows defining the medcoupling field display parameters.
"""

        self.iconsize = "1.0em"

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
        if (self.data is not None) and (
            self.last_computed_frame == [*u, *v, w_value]
        ):
            print("Skipping polygon computation.")
            return self.data, False

        polygons = [
            PolygonElement(
                exterior_polygon = PolygonCoords(
                    x_coords = [i, i, i+1, i+1],
                    y_coords = [0, 1, 1, 0]
                ),
                holes = [],
                cell_id = i
            )
            for i in range(3)
        ]

        self.last_computed_frame = [*u, *v, w_value]
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

def test_build_panel():
    make_panel_2d()
    assert True