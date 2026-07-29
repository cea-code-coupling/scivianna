"""
2D data container for Scivianna.

This module defines the Data2D class, which stores 2D geometry data in either
polygon or grid format, along with associated field values and colors.

Data2D supports:
- Polygon-based representation (unstructured meshes)
- Grid-based representation (structured/rasterized data)
- Automatic conversion between formats
- Arithmetic operations (add, subtract, multiply, divide)
- Color mapping and visualization properties
"""

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from scivianna.data.data_container import DataContainer
from scivianna.enums import DataType
from scivianna.utils.polygonize_tools import PolygonElement, numpy_2D_array_to_polygons


class Data2D(DataContainer):
    """
    Container for 2D geometry data with field values and colors.

    Data2D is the primary data structure for 2D visualization in Scivianna.
    It can represent geometry either as a list of polygons (for unstructured
    meshes) or as a 2D grid (for structured/rasterized data).

    The class supports arithmetic operations between Data2D objects or between
    a Data2D and a scalar, enabling on-the-fly data processing.

    Attributes
    ----------
    data_type : DataType
        Whether data is stored as POLYGONS or GRID
    polygons : List[PolygonElement]
        List of polygons (only used if data_type == POLYGONS)
    grid : np.ndarray
        2D array of cell IDs (only used if data_type == GRID)
    u_values : np.ndarray
        Grid point coordinates along horizontal axis
    v_values : np.ndarray
        Grid point coordinates along vertical axis
    cell_ids : np.ndarray
        Array of unique cell identifiers
    cell_values : np.ndarray
        Array of field values (float or string) for each cell
    cell_colors : np.ndarray
        Array of RGBA colors (0-255) for each cell
    cell_edge_colors : np.ndarray
        Array of RGBA edge colors (0-255) for each cell
    simplify : bool
        Whether to simplify polygons when converting from grid
    """

    data_type: Optional[DataType]
    """Whether the data is stored as POLYGONS or GRID."""

    polygons: List[PolygonElement]
    """List of polygons defining the geometry (used when data_type == POLYGONS)."""

    grid: np.ndarray
    """2D grid of cell IDs (used when data_type == GRID)."""

    u_values: np.ndarray
    """Grid point coordinates along the horizontal axis."""

    v_values: np.ndarray
    """Grid point coordinates along the vertical axis."""

    cell_ids: np.ndarray
    """Numpy array of unique cell identifiers (strings or integers)."""

    cell_values: np.ndarray
    """Numpy array of field values (float or string) for each cell."""

    cell_colors: np.ndarray
    """Numpy array of RGBA colors (shape: N x 4, values 0-255) for each cell."""

    cell_edge_colors: np.ndarray
    """Numpy array of RGBA edge colors (shape: N x 4, values 0-255) for each cell."""

    simplify: Optional[bool]
    """Whether to simplify polygons when converting from grid to polygon list."""

    def __init__(self) -> None:
        """
        Initialize an empty Data2D object.

        This is the default constructor. Use the class methods `from_polygon_list`
        or `from_grid` to create properly initialized Data2D objects.

        Example
        -------
        >>> data = Data2D()  # Empty, not recommended
        >>> data = Data2D.from_polygon_list(polygons)  # Recommended
        >>> data = Data2D.from_grid(grid, u_values, v_values)  # Recommended

        See Also
        --------
        from_polygon_list : Create Data2D from polygon list
        from_grid : Create Data2D from 2D grid
        """
        self.data_type: Optional[DataType] = None
        self.polygons: List[PolygonElement] = []
        self.grid: np.ndarray = np.array([])
        self.u_values: np.ndarray = np.array([])
        self.v_values: np.ndarray = np.array([])
        self.cell_ids: Optional[np.ndarray] = None
        self.cell_values: Optional[np.ndarray] = None
        self.cell_colors: Optional[np.ndarray] = None
        self.cell_edge_colors: Optional[np.ndarray] = None
        self.simplify: Optional[bool] = None

    @classmethod
    def from_polygon_list(cls, polygon_list: List[PolygonElement]) -> "Data2D":
        """
        Create a Data2D object from a list of polygons.

        This factory method initializes a Data2D with polygon-based geometry,
        setting up default colors and empty value arrays. The cell IDs are
        extracted from the polygon cell_id attributes.

        Parameters
        ----------
        polygon_list : List[PolygonElement]
            List of PolygonElement objects defining the geometry.
            Each polygon should have a valid cell_id attribute.

        Returns
        -------
        Data2D
            New Data2D object with data_type set to POLYGONS.
            Initial cell_values are set to NaN, cell_colors to white (255),
            and edge_colors to dark gray (50).

        Example
        -------
        >>> from scivianna.utils.polygonize_tools import PolygonElement, PolygonCoords
        >>>
        >>> # Create a simple square polygon
        >>> exterior = PolygonCoords(
        ...     x_coords=[0, 1, 1, 0],
        ...     y_coords=[0, 0, 1, 1]
        ... )
        >>> polygon = PolygonElement(exterior_polygon=exterior, holes=[], cell_id="cell_1")
        >>>
        >>> # Create Data2D from polygon list
        >>> data = Data2D.from_polygon_list([polygon])
        >>> print(f"Data type: {data.data_type}")  # DataType.POLYGONS
        >>> print(f"Number of cells: {len(data.cell_ids)}")  # 1

        See Also
        --------
        from_grid : Create Data2D from 2D grid
        PolygonElement : Polygon data structure
        """
        data_ = Data2D()
        data_.polygons = polygon_list

        data_.cell_ids = np.array([p.cell_id for p in polygon_list])
        data_.cell_values = np.array([np.nan] * len(polygon_list))

        data_.cell_colors = np.zeros((len(polygon_list), 4)) + 255
        data_.cell_edge_colors = np.zeros((len(polygon_list), 4)) + 50

        data_.data_type = DataType.POLYGONS

        return data_

    @classmethod
    def from_grid(
        cls, grid: np.ndarray, u_values: np.ndarray, v_values: np.ndarray, simplify: bool = False
    ):
        """Build a Data2D object from a list of PolygonElement

        Parameters
        ----------
        grid : np.ndarray
            Numpy 2D array defining the 2D geometry
        u_values : np.ndarray
            Coordinates of the grid points on the horizontal axis
        v_values : np.ndarray
            Coordinates of the grid points on the vertical axis
        simplify : bool
            Simplify the polygons when converted to polygon list

        Returns
        -------
        Data2D
            Requested Data2D
        """
        assert (
            len(grid.shape) == 2
        ), f"Provided grid must be of dimension 2, found shape {grid.shape}"
        data_ = Data2D()
        data_.grid = grid
        data_.u_values = np.array(u_values)
        data_.v_values = np.array(v_values)

        data_.cell_ids = np.unique(grid.flatten())
        data_.cell_values = np.array([np.nan] * len(data_.cell_ids))

        data_.cell_colors = np.zeros((len(data_.cell_ids), 4)) + 1
        data_.cell_edge_colors = np.zeros((len(data_.cell_ids), 4)) + 1

        data_.simplify = simplify
        data_.data_type = DataType.GRID

        return data_

    def convert_to_polygons(
        self,
    ):
        """Convert the geometry to polygons"""
        if self.data_type == DataType.POLYGONS:
            pass
        else:
            self.polygons = numpy_2D_array_to_polygons(
                self.u_values, self.v_values, self.grid, self.simplify
            )

            # The polygons count will become different than the number of cell values, so we update and change the data_type
            id_to_value = dict(zip(self.cell_ids, self.cell_values))
            id_to_color = dict(zip(self.cell_ids, self.cell_colors))
            id_to_edge_color = dict(zip(self.cell_ids, self.cell_edge_colors))

            self.cell_ids = np.array([p.cell_id for p in self.polygons])
            self.cell_values = np.array([id_to_value[e] for e in self.cell_ids])
            self.cell_colors = np.array([id_to_color[e] for e in self.cell_ids])
            self.cell_edge_colors = np.array([id_to_edge_color[e] for e in self.cell_ids])

            self.data_type = DataType.POLYGONS

    def get_polygons(
        self,
    ) -> List[PolygonElement]:
        """Returns the polygon list of the geometry. If defined as grid, the grid is rasterized and self is converted to polygon data.

        Returns
        -------
        List[PolygonElement]
            Polygon list
        """
        if self.data_type == DataType.POLYGONS:
            return self.polygons
        else:
            self.convert_to_polygons()

            return self.polygons

    def get_grid(
        self,
    ) -> np.ndarray:
        """Returns the grid associated to the current geometry

        Returns
        -------
        np.ndarray
            Geometry as a 2D grid

        Raises
        ------
        NotImplementedError
            Grid extraction from polygon list not implemented yet.
        """
        if self.data_type == DataType.POLYGONS:
            raise NotImplementedError()
        else:
            return self.grid

    def copy(
        self,
    ) -> "Data2D":
        """Returns a copy of self

        Returns
        -------
        Data2D
            Identical copy of self
        """
        data2D = Data2D()
        data2D.data_type = self.data_type
        data2D.polygons = self.polygons.copy()
        data2D.grid = self.grid.copy()
        data2D.u_values = self.u_values.copy()
        data2D.v_values = self.v_values.copy()
        data2D.cell_ids = np.array(self.cell_ids)
        data2D.cell_values = np.array(self.cell_values)
        data2D.cell_colors = np.array(self.cell_colors)
        data2D.cell_edge_colors = np.array(self.cell_edge_colors)
        data2D.simplify = self.simplify

        return data2D

    def check_valid(
        self,
    ):
        """Checks if this Data2D is valid, raises an AssertionError otherwise"""
        assert len(self.cell_ids) == len(
            self.cell_colors
        ), "The Data2D object must have the same number of cell id and colors"
        assert len(self.cell_values) == len(
            self.cell_colors
        ), "The Data2D object must have the same number of cell values and colors"
        assert len(self.cell_values) == len(
            self.cell_edge_colors
        ), "The Data2D object must have the same number of cell values and edge colors"
        if self.data_type == DataType.POLYGONS:
            assert len(self.cell_values) == len(
                self.polygons
            ), "The Data2D object must have the same number of cell values and polygons"

        if any(isinstance(item, str) for item in self.cell_values):
            assert all(
                isinstance(item, str) for item in self.cell_values
            ), "If any of the values is a string, they all must be"

    @staticmethod
    def _reorder_data2d_to_match(
        data2d: "Data2D", target_cell_ids: List[Union[int, str]]
    ) -> "Data2D":
        """Reorder a Data2D object to match the order of target cell_ids.

        For cell_ids that exist in data2d but not in target_cell_ids, they are excluded.
        For cell_ids that exist in target_cell_ids but not in data2d, default values (np.nan for values,
        (0, 0, 0) for colors) are used.

        Parameters
        ----------
        data2d : Data2D
            The Data2D object to reorder
        target_cell_ids : List[Union[int, str]]
            The target cell_ids order to match

        Returns
        -------
        Data2D
            A new Data2D object with cell_ids reordered to match target_cell_ids
        """
        # Create a mapping from cell_id to index in the original data2d
        id_to_idx = {cid: idx for idx, cid in enumerate(data2d.cell_ids)}

        new_cell_ids = []
        new_cell_values = []
        new_cell_colors = []
        new_cell_edge_colors = []

        default_color = np.array([0, 0, 0, 0])
        default_edge_color = np.array([0, 0, 0, 0])

        for cid in target_cell_ids:
            if cid in id_to_idx:
                idx = id_to_idx[cid]
                new_cell_ids.append(cid)
                new_cell_values.append(data2d.cell_values[idx])
                new_cell_colors.append(data2d.cell_colors[idx])
                new_cell_edge_colors.append(data2d.cell_edge_colors[idx])
            else:
                # Cell not in original data2d, use default values
                new_cell_ids.append(cid)
                new_cell_values.append(np.nan)
                new_cell_colors.append(default_color.copy())
                new_cell_edge_colors.append(default_edge_color.copy())

        result = data2d.copy()
        result.cell_ids = np.array(new_cell_ids)
        result.cell_values = np.array(new_cell_values)
        result.cell_colors = np.array(new_cell_colors)
        result.cell_edge_colors = np.array(new_cell_edge_colors)

        return result

    def _binary_operation(self, other: Union["Data2D", float], op: Callable) -> "Data2D":
        """Perform a binary operation between this Data2D and another Data2D or a float.

        Parameters
        ----------
        other : Data2D or float
            The other operand
        op : Callable
            The operation to perform

        Returns
        -------
        Data2D
            A new Data2D object with the operation applied
        """
        if isinstance(other, (int, float)):
            # Operation with a scalar - preserve colors from self
            result = self.copy()
            result.cell_values = np.array([op(float(v), other) for v in self.cell_values])
            return result

        elif isinstance(other, Data2D):
            # Operation with another Data2D
            # Find the union of cell_ids from both objects, maintaining order
            seen = set()
            union_cell_ids = []
            self_ids = (
                self.cell_ids if isinstance(self.cell_ids, (list, np.ndarray)) else [self.cell_ids]
            )
            other_ids = (
                other.cell_ids
                if isinstance(other.cell_ids, (list, np.ndarray))
                else [other.cell_ids]
            )
            for cid in list(self_ids) + list(other_ids):
                if cid not in seen:
                    seen.add(cid)
                    union_cell_ids.append(cid)

            # Reorder both to match the union
            self_reordered = self._reorder_data2d_to_match(self, union_cell_ids)
            other_reordered = self._reorder_data2d_to_match(other, union_cell_ids)

            # Create result
            result = self.copy()
            result.cell_ids = np.array(union_cell_ids)

            # Apply the operation element-wise
            new_values = []
            for v1, v2 in zip(self_reordered.cell_values, other_reordered.cell_values):
                try:
                    new_values.append(op(float(v1), float(v2)))
                except (TypeError, ZeroDivisionError, ValueError):
                    new_values.append(np.nan)

            result.cell_values = np.array(new_values)
            # Set default colors: white for cells (255, 255, 255, 255), gray for edges (200, 200, 200, 255)
            n = len(result.cell_ids)
            cell_color_rgba = np.array([255, 255, 255, 255])
            edge_color_rgba = np.array([200, 200, 200, 255])
            result.cell_colors = np.tile(cell_color_rgba.reshape(1, 4), (n, 1))
            result.cell_edge_colors = np.tile(edge_color_rgba.reshape(1, 4), (n, 1))

            return result

        else:
            raise TypeError(
                f"Unsupported operand type(s) for operation: '{type(self).__name__}' and '{type(other).__name__}'"
            )

    def __add__(self, other: Union["Data2D", float]) -> "Data2D":
        """Add two Data2D objects or a Data2D and a float.

        Parameters
        ----------
        other : Data2D or float
            The other operand

        Returns
        -------
        Data2D
            A new Data2D object with the addition applied
        """
        return self._binary_operation(other, lambda a, b: a + b)

    def __sub__(self, other: Union["Data2D", float]) -> "Data2D":
        """Subtract two Data2D objects or a float from a Data2D.

        Parameters
        ----------
        other : Data2D or float
            The other operand

        Returns
        -------
        Data2D
            A new Data2D object with the subtraction applied
        """
        return self._binary_operation(other, lambda a, b: a - b)

    def __mul__(self, other: Union["Data2D", float]) -> "Data2D":
        """Multiply two Data2D objects or a Data2D by a float.

        Parameters
        ----------
        other : Data2D or float
            The other operand

        Returns
        -------
        Data2D
            A new Data2D object with the multiplication applied
        """
        return self._binary_operation(other, lambda a, b: a * b)

    def __truediv__(self, other: Union["Data2D", float]) -> "Data2D":
        """Divide two Data2D objects or a Data2D by a float.

        Parameters
        ----------
        other : Data2D or float
            The other operand

        Returns
        -------
        Data2D
            A new Data2D object with the division applied
        """
        return self._binary_operation(other, lambda a, b: a / b if b != 0 else np.nan)

    def __radd__(self, other: float) -> "Data2D":
        """Add a float to a Data2D (reverse addition).

        Parameters
        ----------
        other : float
            The float operand

        Returns
        -------
        Data2D
            A new Data2D object with the addition applied
        """
        return self.__add__(other)

    def __rsub__(self, other: float) -> "Data2D":
        """Subtract a Data2D from a float (reverse subtraction).

        Parameters
        ----------
        other : float
            The float operand

        Returns
        -------
        Data2D
            A new Data2D object with the subtraction applied
        """
        return self._binary_operation(other, lambda a, b: b - a)

    def __rmul__(self, other: float) -> "Data2D":
        """Multiply a float by a Data2D (reverse multiplication).

        Parameters
        ----------
        other : float
            The float operand

        Returns
        -------
        Data2D
            A new Data2D object with the multiplication applied
        """
        return self.__mul__(other)

    def __rtruediv__(self, other: float) -> "Data2D":
        """Divide a float by a Data2D (reverse division).

        Parameters
        ----------
        other : float
            The float operand

        Returns
        -------
        Data2D
            A new Data2D object with the division applied
        """
        return self._binary_operation(other, lambda a, b: b / a if a != 0 else np.nan)

    def __iadd__(self, other: Union["Data2D", float]) -> "Data2D":
        """Add in-place (+=).

        Parameters
        ----------
        other : Data2D or float
            The other operand

        Returns
        -------
        Data2D
            self with the addition applied
        """
        result = self.__add__(other)
        self.cell_values = result.cell_values
        return self

    def __isub__(self, other: Union["Data2D", float]) -> "Data2D":
        """Subtract in-place (-=).

        Parameters
        ----------
        other : Data2D or float
            The other operand

        Returns
        -------
        Data2D
            self with the subtraction applied
        """
        result = self.__sub__(other)
        self.cell_values = result.cell_values
        return self

    def __imul__(self, other: Union["Data2D", float]) -> "Data2D":
        """Multiply in-place (*=).

        Parameters
        ----------
        other : Data2D or float
            The other operand

        Returns
        -------
        Data2D
            self with the multiplication applied
        """
        result = self.__mul__(other)
        self.cell_values = result.cell_values
        return self

    def __itruediv__(self, other: Union["Data2D", float]) -> "Data2D":
        """Divide in-place (/=).

        Parameters
        ----------
        other : Data2D or float
            The other operand

        Returns
        -------
        Data2D
            self with the division applied
        """
        result = self.__truediv__(other)
        self.cell_values = result.cell_values
        return self
