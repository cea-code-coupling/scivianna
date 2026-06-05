
import os
from pathlib import Path
import pickle
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import multiprocessing as mp

from scivianna.data.data2d import Data2D
from scivianna.interface.generic_interface import Geometry2DPolygon
from scivianna.utils.polygonize_tools import PolygonElement
from scivianna.enums import GeometryType, VisualizationMode
from scivianna.utils.structured_mesh import CarthesianStructuredMesh, StructuredMesh

from scivianna.constants import GEOMETRY, MESH


class StructuredMeshInterface(Geometry2DPolygon):
    """ StructuredMesh generic interface. This interface is not usable as such as it can't load a file.
    To use it, the developper must implement another interface inheriting from it implementing the read_file function.
    """

    data: Data2D
    """Data computed at the previous iteration"""

    mesh: StructuredMesh
    """Mesh read from the .med file."""

    fields: Dict[str, np.ndarray]
    """Dictionnary containing the list of per cell value for each read field."""

    geometry_type = GeometryType._3D_INFINITE

    def __init__(self, ):
        """StructuredMesh interface constructor."""
        self.data: Dict[str, Data2D] = {}
        """Dictionary with caller as key storing past computed data for each caller"""
        self.last_computed_frame = {}
        """Dictionary with caller as key storing parameters of the last computed frame"""

    def read_file(self, file_path: str, file_label: str):
        """Read a file and store its content in the interface

        Parameters
        ----------
        file_path : str
            File to read
        file_label : str
            Label to define the file type
        """
        raise NotImplementedError()

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
        caller: str = "API",
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
        if (caller in self.last_computed_frame) and (
            self.last_computed_frame.get(caller) == [*u, *v, w_value]
        ) and (caller in self.data):
            print("Skipping polygon computation.")
            return self.data[caller], False

        self.last_computed_frame[caller] = [*u, *v, w_value]

        u = np.array(u)
        v = np.array(v)
        vec = np.cross(u, v)
        origin = u_min * u + v_min * v + w_value * vec

        self.data[caller] = Data2D.from_polygon_list(self.mesh.compute_2D_slice(origin, u, v))

        return self.data[caller], True

    def get_labels(
        self,
    ) -> List[str]:
        """Returns a list of fields names displayable with this interface

        Returns
        -------
        List[str]
            List of fields names
        """
        labels = [MESH] + list(self.mesh.grids.keys())
        return labels

    def get_value_dict(
        self, value_label: str, cells: List[Union[int, str]], options: Dict[str, Any], caller: str = "API"
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
        if value_label == MESH:
            return {v: np.nan for v in cells}

        return dict(zip(cells, self.mesh.get_cells_values(value_label, cells)))

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

        return VisualizationMode.FROM_VALUE

    def get_file_input_list(self) -> List[Tuple[str, str]]:
        """Returns a list of file label and its description for the GUI

        Returns
        -------
        List[Tuple[str, str]]
            List of (file label, description)
        """
        return []

    def save(self, file_path: Path, include_files: bool):
        """Pickle saves the slave content to a file, allows slave state reload.

        Two modes are available:
            -   If **include_files** is at True, all loaded data are saved, the pickled file can be loaded on its own to recover last session.
            -   If **include_files** is at False, only the computed data are loaded, enabling faster first computation allowing a smaller pickle file size.

        Parameters
        ----------
        file_path : Path
            File in which save the file
        include_files : bool
            Included loaded file
        """
        os.makedirs(Path(file_path).parent, exist_ok=True)

        with open(file_path, "wb") as f:
            data = self.last_computed_frame, self.mesh, self.fields, self.data

            pickle.dump(data, f)

    def load(self, file_path: Path, include_files: bool):
        """Pickle loads the slave content to a file, allows slave state reload

        Two modes are available:
            -   If **include_files** is at True, all loaded data are saved, the pickled file can be loaded on its own to recover last session.
            -   If **include_files** is at False, only the computed data are loaded, enabling faster first computation allowing a smaller pickle file size.

        Parameters
        ----------
        file_path : Path
            File from which load the slave
        include_files : bool
            Included loaded file
        """
        if not os.path.isfile(file_path):
            raise ValueError(f"Provided path {file_path} does not exist")

        with open(file_path, "rb") as f:
            data = pickle.load(f)

            self.last_computed_frame, self.mesh, self.fields, self.data = data


if __name__ == "__main__":
    from scivianna.slave import ComputeSlave
    from scivianna.panel.visualisation_panel import VisualizationPanel
    from scivianna.notebook_tools import _show_panel

    class MyMeshInterface(StructuredMeshInterface):
        def read_file(self, file_path: str, file_label: str):
            """Read a file and store its content in the interface

            Parameters
            ----------
            file_path : str
                File to read
            file_label : str
                Label to define the file type
            """
            size = 40
            self.mesh = CarthesianStructuredMesh(
                np.linspace(0, 4, size),
                np.linspace(0, 4, size),
                np.linspace(0, 4, size),
            )
            self.mesh.set_values("id", np.arange(size * size * size).reshape(size, size, size))

    slave = ComputeSlave(MyMeshInterface)
    slave.read_file(
        None,
        None,
    )

    _show_panel(VisualizationPanel(slave, name="StructuredMesh visualizer"))
