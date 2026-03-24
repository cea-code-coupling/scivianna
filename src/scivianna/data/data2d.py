from typing import Callable, Dict, List, Any, Tuple, Union
import numpy as np
from scivianna.utils.polygonize_tools import (
    PolygonCoords,
    PolygonElement,
    numpy_2D_array_to_polygons,
)
from scivianna.enums import DataType
from scivianna.data.data_container import DataContainer


class Data2D(DataContainer):
    """Data class containing the 2D geometry data"""

    data_type: DataType
    """Whether the data are provided from a polygon list or a grid"""

    polygons: List[PolygonElement]
    """List of polygons defining the geometry"""

    grid: np.ndarray
    """2D grid defining the geometry"""
    u_values: np.ndarray
    """Coordinates of the grid points on the horizontal axis"""
    v_values: np.ndarray
    """Coordinates of the grid points on the vertical axis"""

    cell_ids: np.ndarray[Union[int, str]]
    """List of contained cell ids"""
    cell_values: np.ndarray[Union[float, str]]
    """List of contained cell values"""
    cell_colors: np.ndarray[Tuple[int, int, int]]
    """List of contained cell colors"""
    cell_edge_colors: np.ndarray[Tuple[int, int, int]]
    """List of contained cell edge colors"""

    simplify: bool
    """Simplify the polygons when converting from grid to polygon list"""

    def __init__(self):
        """Empty constructor of the Data2D class."""
        self.data_type = None
        self.polygons = []
        self.grid = np.array([])
        self.u_values = np.array([])
        self.v_values = np.array([])
        self.cell_ids = np.array([])
        self.cell_values = np.array([])
        self.cell_colors = np.array([])
        self.cell_edge_colors = np.array([])
        self.simplify = None

    @classmethod
    def from_polygon_list(cls, polygon_list: List[PolygonElement]):
        """Build a Data2D object from a list of PolygonElement

        Parameters
        ----------
        polygon_list : List[PolygonElement]
            Polygons contained in the Data2D

        Returns
        -------
        Data2D
            Requested Data2D
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
        cls,
        grid: np.ndarray,
        u_values: np.ndarray,
        v_values: np.ndarray,
        simplify: bool = False,
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
        grid = np.array(grid)
        assert (
            len(grid.shape) == 2
        ), f"Provided grid must be of dimension 2, found shape {grid.shape}"
        data_ = Data2D()
        data_.grid = grid
        data_.u_values = u_values
        data_.v_values = v_values

        data_.cell_ids = np.unique(grid.flatten())
        data_.cell_values = [np.nan] * len(data_.cell_ids)

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

            self.cell_ids = [p.cell_id for p in self.polygons]
            self.cell_values = [id_to_value[e] for e in self.cell_ids]
            self.cell_colors = [id_to_color[e] for e in self.cell_ids]
            self.cell_edge_colors = [id_to_edge_color[e] for e in self.cell_ids]

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
        data_2d = Data2D()
        data_2d.data_type = self.data_type
        data_2d.polygons = self.polygons.copy()
        data_2d.grid = self.grid.copy()
        data_2d.u_values = self.u_values.copy()
        data_2d.v_values = self.v_values.copy()
        data_2d.cell_ids = self.cell_ids.copy()
        data_2d.cell_values = self.cell_values.copy()
        data_2d.cell_colors = self.cell_colors.copy()
        data_2d.cell_edge_colors = self.cell_edge_colors.copy()
        data_2d.simplify = self.simplify

        return data_2d

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

    def __init_operation__(
        self, other: "Data2D"
    ) -> Tuple["Data2D", np.ndarray, np.ndarray, np.ndarray]:
        """
        Initializes the operation between two Data2D objects, checking the validity of the operation and preparing the output Data2D and the indexes for the operation.
        -   If the data types are different, an AssertionError is raised.
        -   If the values are not numbers, a ValueError is raised.
        -   The order in the first Data2D is kept.

        Parameters
        ----------
        other : Data2D
            The other Data2D to operate with self

        Returns
        -------
        Tuple["Data2D", np.ndarray, np.ndarray, np.ndarray]
            The output Data2D, the result indexes, the mask and the stored values
        """
        assert (
            self.data_type == other.data_type
        ), "The two Data2D must have the same data type to be operated together"
        if any(isinstance(item, str) for item in self.cell_values + other.cell_values):
            raise ValueError("Cannot operate two Data2D with string values")

        output = self.copy()

        index = np.argsort(output.cell_ids)

        output.cell_ids = np.array(output.cell_ids)[index]
        output.cell_colors = np.array(output.cell_colors)[index]
        output.cell_values = np.array(output.cell_values)[index]
        output.cell_edge_colors = np.array(output.cell_edge_colors)[index]

        x = np.array(output.cell_ids)
        sorted_index = np.searchsorted(x, other.cell_ids)

        yindex = np.take(index, sorted_index, mode="clip")

        print("Sorting index", sorted_index)

        print("self before sort", self.cell_ids, "self after sort", output.cell_ids, "self sorted like other", x[yindex], "other ", other.cell_ids, "try ", other.cell_ids[yindex])
        mask = output.cell_ids != other.cell_ids[yindex]

        result = np.ma.array(yindex, mask=mask, hard_mask=True).compressed()

        stored_values = np.array(output.cell_values)[result]
        output.cell_values.fill(np.nan)

        return output, result, mask, stored_values

    def __add__(self, other: Union[float, "Data2D"]) -> "Data2D":
        """Adds a Data2D or a float to a Data2D together: adds the cell values and resets the colors.

        -   If the data types are different, an AssertionError is raised.
        -   If the values are not numbers, a ValueError is raised.
        -   If there is a mismatch in the cell ids, np.nan will be set as value for the missing cell ids in the resulting Data2D.
        -   The order in the first Data2D is kept.

        Parameters
        ----------
        other : Union[float, "Data2D"]
            The other Data2D to add to self or a float to add to the cell values of self

        Returns
        -------
        Data2D
            The resulting Data2D after addition
        """
        if isinstance(other, Data2D):
            output, result, mask, stored_values = self.__init_operation__(other)
            output.cell_values[result] = (
                stored_values + np.array(other.cell_values)[~mask]
            )
        else:
            if any(isinstance(item, str) for item in self.cell_values):
                raise ValueError("Cannot add a float to a Data2D with string values")
            output = self.copy()
            output.cell_values = np.array(output.cell_values) + other

        return output

    def __radd__(self, other: Union[float, "Data2D"]) -> "Data2D":
        """Adds a Data2D or a float to a Data2D together: adds the cell values and resets the colors.

        -   If the data types are different, an AssertionError is raised.
        -   If the values are not numbers, a ValueError is raised.
        -   If there is a mismatch in the cell ids, np.nan will be set as value for the missing cell ids in the resulting Data2D.
        -   The order in the first Data2D is kept.

        Parameters
        ----------
        other : Union[float, "Data2D"]
            The other Data2D to add to self or a float to add to the cell values of self

        Returns
        -------
        Data2D
            The resulting Data2D after addition
        """
        return self.__add__(other)

    def __sub__(self, other: Union[float, "Data2D"]) -> "Data2D":
        """Subtracts a Data2D or a float from a Data2D together: subtracts the cell values and resets the colors.

        -   If the data types are different, an AssertionError is raised.
        -   If the values are not numbers, a ValueError is raised.
        -   If there is a mismatch in the cell ids, np.nan will be set as value for the missing cell ids in the resulting Data2D.
        -   The order in the first Data2D is kept.

        Parameters
        ----------
        other : Union[float, "Data2D"]
            The other Data2D to subtract from self or a float to subtract from the cell values of self

        Returns
        -------
        Data2D
            The resulting Data2D after subtraction
        """
        if isinstance(other, Data2D):
            output, result, mask, stored_values = self.__init_operation__(other)
            output.cell_values[result] = (
                stored_values - np.array(other.cell_values)[~mask]
            )
        else:
            if any(isinstance(item, str) for item in self.cell_values):
                raise ValueError(
                    "Cannot subtract a float from a Data2D with string values"
                )
            output = self.copy()
            output.cell_values = np.array(output.cell_values) - other

        return output

    def __rsub__(self, other: Union[float, "Data2D"]) -> "Data2D":
        """Subtracts a Data2D or a float from a Data2D together: subtracts the cell values and resets the colors.

        -   If the data types are different, an AssertionError is raised.
        -   If the values are not numbers, a ValueError is raised.
        -   If there is a mismatch in the cell ids, np.nan will be set as value for the missing cell ids in the resulting Data2D.
        -   The order in the first Data2D is kept.

        Parameters
        ----------
        other : Union[float, "Data2D"]
            The other Data2D to subtract from self or a float to subtract from the cell values of self

        Returns
        -------
        Data2D
            The resulting Data2D after subtraction
        """
        data_2d = self.__sub__(other)
        data_2d.cell_values *= -1
        return data_2d

    def __mul__(self, other: Union[float, "Data2D"]) -> "Data2D":
        """Multiplies a Data2D or a float with a Data2D together: multiplies the cell values and resets the colors.

        -   If the data types are different, an AssertionError is raised.
        -   If the values are not numbers, a ValueError is raised.
        -   If there is a mismatch in the cell ids, np.nan will be set as value for the missing cell ids in the resulting Data2D.
        -   The order in the first Data2D is kept.

        Parameters
        ----------
        other : Union[float, "Data2D"]
            The other Data2D to multiply with self or a float to multiply with the cell values of self

        Returns
        -------
        Data2D
            The resulting Data2D after multiplication
        """
        if isinstance(other, Data2D):
            output, result, mask, stored_values = self.__init_operation__(other)
            output.cell_values[result] = (
                stored_values * np.array(other.cell_values)[~mask]
            )
        else:
            if any(isinstance(item, str) for item in self.cell_values):
                raise ValueError(
                    "Cannot multiply a float with a Data2D with string values"
                )
            output = self.copy()
            output.cell_values = np.array(output.cell_values) * other

        return output

    def __rmul__(self, other: Union[float, "Data2D"]) -> "Data2D":
        """Multiplies a Data2D or a float with a Data2D together: multiplies the cell values and resets the colors.

        -   If the data types are different, an AssertionError is raised.
        -   If the values are not numbers, a ValueError is raised.
        -   If there is a mismatch in the cell ids, np.nan will be set as value for the missing cell ids in the resulting Data2D.
        -   The order in the first Data2D is kept.

        Parameters
        ----------
        other : Union[float, "Data2D"]
            The other Data2D to multiply with self or a float to multiply with the cell values of self

        Returns
        -------
        Data2D
            The resulting Data2D after multiplication
        """
        return self.__mul__(other)

    def __truediv__(self, other: Union[int, float, "Data2D"]) -> "Data2D":
        """Divides a Data2D or a float with a Data2D together: divides the cell values and resets the colors.

        -   If the data types are different, an AssertionError is raised.
        -   If the values are not numbers, a ValueError is raised.
        -   If there is a mismatch in the cell ids, np.nan will be set as value for the missing cell ids in the resulting Data2D.
        -   The order in the first Data2D is kept.

        Parameters
        ----------
        other : Union[float, "Data2D"]
            The other Data2D to divide with self or a float to divide with the cell values of self

        Returns
        -------
        Data2D
            The resulting Data2D after division
        """
        if isinstance(other, Data2D):
            output, result, mask, stored_values = self.__init_operation__(other)
            output.cell_values[result] = (
                stored_values / np.array(other.cell_values)[~mask]
            )
        else:
            if any(isinstance(item, str) for item in self.cell_values):
                raise ValueError(
                    "Cannot divide a float with a Data2D with string values"
                )
            output = self.copy()
            output.cell_values = np.array(output.cell_values) / other

        return output

    def __rtruediv__(self, other: Union[int, float, "Data2D"]) -> "Data2D":
        """Divides a Data2D or a float with a Data2D together: divides the cell values and resets the colors.

        -   If the data types are different, an AssertionError is raised.
        -   If the values are not numbers, a ValueError is raised.
        -   If there is a mismatch in the cell ids, np.nan will be set as value for the missing cell ids in the resulting Data2D.
        -   The order in the first Data2D is kept.

        Parameters
        ----------
        other : Union[float, "Data2D"]
            The other Data2D to divide with self or a float to divide with the cell values of self

        Returns
        -------
        Data2D
            The resulting Data2D after division
        """
        data_2d = self / other
        data_2d.cell_values = 1/data_2d.cell_values
        return data_2d


if __name__ == "__main__":
    data_1 = Data2D.from_polygon_list(
        [
            PolygonElement(
                cell_id=i,
                exterior_polygon=PolygonCoords(
                    x_coords=[0, 1, 1, 0], y_coords=[0, 0, 1, 1]
                ),
                holes=[],
            )
            for i in range(5)
        ]
    )
    data_1.cell_values = [i * 1.0 for i in range(5)]

    data_2 = data_1.copy()
    data_2.cell_ids = [i * 3 for i in range(-2, 5)]
    data_2.cell_values = [i * 3.0 for i in range(-2, 5)]

    print((data_1 / data_2).cell_values)

    print((2 * data_2).cell_values)
