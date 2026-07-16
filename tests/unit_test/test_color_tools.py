from typing import Any, Dict, List, Tuple, Union
import numpy as np
import multiprocessing as mp
import pytest

from scivianna.data.data2d import Data2D
from scivianna.interface.generic_interface import Geometry2DPolygon
from scivianna.extension.field_selector import set_colors_list
from scivianna.utils.polygonize_tools import PolygonCoords, PolygonElement
from scivianna.enums import VisualizationMode
from scivianna.constants import MESH

from scivianna.utils.color_tools import interpolate_cmap_at_values, get_edges_colors


class ColorTestInterface(Geometry2DPolygon):
    def __init__(
        self,
    ):
        """Interface built to test color tools."""
        pass

    def read_file(self, file_path: str, file_label: str):
        """Read a file and store its content in the interface

        Parameters
        ----------
        file_path : str
            File to read
        file_label : str
            Label to define the file type
        """
        pass

    def compute_2D_data(
        self,
        u: Tuple[float, float, float],
        v: Tuple[float, float, float],
        origin: Tuple[float, float, float],
        size_u: float,
        size_v: float,
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
        origin : Tuple[float, float, float]
            Physical 3D position of the slice corner
        size_u : float
            Size of the slice along the u axis
        size_v : float
            Size of the slice along the v axis
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
        return (
            Data2D.from_polygon_list(
                [
                    PolygonElement(
                        exterior_polygon=PolygonCoords(x_coords=[], y_coords=[]),
                        holes=[],
                        cell_id=i,
                    )
                    for i in range(5)
                ]
            ),
            True,
        )

    def get_labels(
        self,
    ) -> List[str]:
        """Returns a list of fields names displayable with this interface

        Returns
        -------
        List[str]
            List of fields names
        """
        return [MESH, "str", "float"]

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
        elif value_label == "str":
            return {v: str(v) for v in cells}
        elif value_label == "float":
            return {v: float(v) if v < 2 else np.nan for v in cells}
        else:
            raise ValueError(
                f"Value label {value_label} not implemented. Keys available : MESH, str, and float."
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
        elif label == "str":
            return VisualizationMode.FROM_STRING
        elif label == "float":
            return VisualizationMode.FROM_VALUE
        else:
            raise ValueError(
                f"Label {label} not implemented. Keys available : MESH, str, and float."
            )

    def get_file_input_list(self) -> List[Tuple[str, str]]:
        """Returns a list of file label and its description for the GUI

        Returns
        -------
        List[Tuple[str, str]]
            List of (file label, description)
        """
        return []


@pytest.mark.default
def test_interpolate_cmap_gray():
    colors = interpolate_cmap_at_values("gray", [0, 0.5, 1.0])

    white = np.array((255, 255, 255, 255))
    black = np.array((0, 0, 0, 255))

    np.testing.assert_equal(
        colors,
        [
            black,
            (0.5 * (black + white + np.array((1.0, 1.0, 1.0, 0.0)))).astype(int),
            white,
        ],
    )


@pytest.mark.default
def test_interpolate_cmap_blues():
    colors = interpolate_cmap_at_values("Blues", [0, 0.4, 1.0])

    value_0 = np.array((247, 251, 255, 255))
    value_05 = np.array((147, 196, 222, 255))
    value_1 = np.array((8, 48, 107, 255))

    np.testing.assert_equal(colors, [value_0, value_05, value_1])


@pytest.mark.default
def test_color_list_none():
    interface = ColorTestInterface()
    data2D, _ = interface.compute_2D_data(u=None, v=None, origin=(0.0, 0.0, 0.0), size_u=1.0, size_v=1.0, q_tasks=None, options={}, caller="Test")
    set_colors_list(data2D, interface, MESH, "gray", False, {})

    np.testing.assert_equal(data2D.cell_colors, [(200, 200, 200, 0)] * 5)
    np.testing.assert_equal(data2D.cell_edge_colors, [(180, 180, 180, 255)] * 5)


@pytest.mark.default
def test_color_list_str():
    interface = ColorTestInterface()
    data2D, _ = interface.compute_2D_data(u=None, v=None, origin=(0.0, 0.0, 0.0), size_u=1.0, size_v=1.0, q_tasks=None, options={}, caller="Test")
    set_colors_list(data2D, interface, "str", "gray", False, {})

    np.testing.assert_equal(
        data2D.cell_edge_colors, get_edges_colors(np.array(data2D.cell_colors))
    )


@pytest.mark.default
def test_color_list_float():
    interface = ColorTestInterface()
    data2D, _ = interface.compute_2D_data(u=None, v=None, origin=(0.0, 0.0, 0.0), size_u=1.0, size_v=1.0, q_tasks=None, options={}, caller="Test")

    dict_value_per_cell = interface.get_value_dict("float", data2D.cell_ids, {})

    data2D.cell_values = [dict_value_per_cell[v] for v in data2D.cell_ids]

    set_colors_list(data2D, interface, "float", "gray", False, {})

    white = np.array((255, 255, 255, 255))
    black = np.array((0, 0, 0, 255))
    transparent = np.array((200, 200, 200, 0))

    np.testing.assert_equal(
        data2D.cell_colors, [black, white, transparent, transparent, transparent]
    )
    np.testing.assert_equal(
        data2D.cell_edge_colors,
        [
            (0, 0, 0, 255),
            (235, 235, 235, 255),
            [180, 180, 180, 255],
            [180, 180, 180, 255],
            [180, 180, 180, 255],
        ],
    )
