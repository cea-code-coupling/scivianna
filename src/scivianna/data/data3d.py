from typing import List, Tuple, Union
import numpy as np
from scivianna.utils.polygonize_tools import PolygonCell
from scivianna.enums import DataType
from scivianna.data.data_container import DataContainer


class Data3D(DataContainer):
    """Data class containing the 3D geometry data"""

    data_type: DataType
    """Whether the data are provided from a polygon list or a grid"""

    cells: List[PolygonCell]
    """List of polygons defining the geometry"""

    grid: np.ndarray
    """3D grid defining the geometry"""
    x_values: np.ndarray
    """Coordinates of the grid points on the X axis"""
    y_values: np.ndarray
    """Coordinates of the grid points on the Y axis"""
    z_values: np.ndarray
    """Coordinates of the grid points on the Z axis"""

    cell_ids: List[Union[int, str]]
    """List of contained cell ids"""
    cell_values: List[Union[float, str]]
    """List of contained cell values"""
    cell_colors: List[Tuple[int, int, int]]
    """List of contained cell colors"""
    cell_edge_colors: List[Tuple[int, int, int]]
    """List of contained cell edge colors"""

    simplify: bool
    """Simplify the polygons when converting from grid to polygon list"""

    def __init__(self):
        """Empty constructor of the Data3D class."""
        self.data_type = None
        self.cells = []
        self.grid = np.array([])
        self.u_values = np.array([])
        self.v_values = np.array([])
        self.cell_ids = []
        self.cell_values = []
        self.cell_colors = []
        self.cell_edge_colors = []
        self.simplify = None

    @classmethod
    def from_polygon_list(cls, cells: List[PolygonCell]):
        """Build a Data3D object from a list of PolygonCell

        Parameters
        ----------
        cells : List[PolygonCell]
            Polygons contained in the Data3D

        Returns
        -------
        Data3D
            Requested Data3D
        """
        data_ = Data3D()
        data_.cells = cells

        data_.cell_ids = [p.cell_id for p in cells]
        data_.cell_values = [np.nan] * len(cells)

        data_.cell_colors = np.zeros((len(cells), 4)) + 255
        data_.cell_edge_colors = np.zeros((len(cells), 4)) + 50

        data_.data_type = DataType.POLYGONS

        return data_

    @classmethod
    def from_grid(
        cls,
        grid: np.ndarray,
        u_values: np.ndarray,
        v_values: np.ndarray,
        simplify: bool = False,
    ):
        """Build a Data3D object from a list of PolygonCell

        Parameters
        ----------
        grid : np.ndarray
            Numpy 3D array defining the 3D geometry
        u_values : np.ndarray
            Coordinates of the grid points on the horizontal axis
        v_values : np.ndarray
            Coordinates of the grid points on the vertical axis
        simplify : bool
            Simplify the polygons when converted to polygon list

        Returns
        -------
        Data3D
            Requested Data3D
        """
        assert (
            len(grid.shape) == 3
        ), f"Provided grid must be of dimension 3, found shape {grid.shape}"
        raise NotImplementedError()

    def convert_to_polygons(
        self,
    ):
        """Convert the geometry to polygons"""
        if self.data_type == DataType.POLYGONS:
            pass
        else:
            raise NotImplementedError()

    def get_polygons(
        self,
    ) -> List[PolygonCell]:
        """Returns the polygon list of the geometry. If defined as grid, the grid is rasterized and self is converted to polygon data.

        Returns
        -------
        List[PolygonCell]
            Polygon list
        """
        if self.data_type == DataType.POLYGONS:
            return self.cells
        else:
            self.convert_to_polygons()

            return self.cells

    def get_grid(
        self,
    ) -> np.ndarray:
        """Returns the grid associated to the current geometry

        Returns
        -------
        np.ndarray
            Geometry as a 3D grid

        Raises
        ------
        NotImplementedError
            Grid extraction from polygon list not implemented yet.
        """
        if self.data_type == DataType.POLYGONS:
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def copy(
        self,
    ) -> "Data3D":
        """Returns a copy of self

        Returns
        -------
        Data3D
            Identical copy of self
        """
        data3D = Data3D()
        data3D.data_type = self.data_type
        data3D.polygons = self.cells.copy()
        data3D.grid = self.grid.copy()
        data3D.u_values = self.u_values.copy()
        data3D.v_values = self.v_values.copy()
        data3D.cell_ids = np.array(self.cell_ids).tolist()
        data3D.cell_values = np.array(self.cell_values).tolist()
        data3D.cell_colors = np.array(self.cell_colors).tolist()
        data3D.cell_edge_colors = np.array(self.cell_edge_colors).tolist()
        data3D.simplify = self.simplify

        return data3D

    def check_valid(
        self,
    ):
        """Checks if this Data3D is valid, raises an AssertionError otherwise"""
        assert len(self.cell_ids) == len(
            self.cell_colors
        ), "The Data3D object must have the same number of cell id and colors"
        assert len(self.cell_values) == len(
            self.cell_colors
        ), "The Data3D object must have the same number of cell values and colors"
        assert len(self.cell_values) == len(
            self.cell_edge_colors
        ), "The Data3D object must have the same number of cell values and edge colors"
        if self.data_type == DataType.POLYGONS:
            assert len(self.cell_values) == len(
                self.cells
            ), "The Data3D object must have the same number of cell values and polygons"

        if any(isinstance(item, str) for item in self.cell_values):
            assert all(
                isinstance(item, str) for item in self.cell_values
            ), "If any of the values is a string, they all must be"
