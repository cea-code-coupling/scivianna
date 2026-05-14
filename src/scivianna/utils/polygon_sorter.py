
from typing import List, Union

import numpy as np
from scivianna.data.data2d import Data2D, DataType


class PolygonSorter:
    """Object used to sort a list of polygons per field
    """
    def __init__(self):
        """Object used to sort a list of polygons per field
        """
        self.sort_indexes = None

    def sort_from_value(self,
            data:Data2D
        ):
        """Sorts a set of polygons per field

        Parameters
        ----------
        data : Data2D
            Data2D object containing the geometry properties
        """
        values_list:List[Union[int, str]] = data.cell_values

        assert len(data.cell_ids) == len(data.cell_colors), "The Data2D object must have the same number of cell id and colors"
        assert len(data.cell_ids) == len(data.cell_edge_colors), "The Data2D object must have the same number of cell id and edge colors"
        assert len(data.cell_values) == len(data.cell_colors), "The Data2D object must have the same number of cell values and colors"
        if data.data_type == DataType.POLYGONS:
            assert len(data.cell_values) == len(data.polygons), "The Data2D object must have the same number of cell values and polygons"

        if any(isinstance(item, str) for item in values_list):
            assert all(isinstance(item, str) for item in values_list), "If any of the values is a string, they all must be"

        # Sorting the polygons per color in order to prevent overlaping edges of different colors
        # Check if both sort and sort_indexes is None in case a slave is used for different panels.
        self.sort_indexes = np.argsort(values_list)
        
        self.sort_list(data)

    def sort_list(self, data:Data2D):
        """Sort the value and color list of a Data2D object in the same order as the past sort_polygon_list order. 
        If no sort_polygon_list was called, it is called instead.

        Parameters
        ----------
        arr : List[Any]
            List to sort
        """
        if self.sort_indexes is None:
            self.sort_from_value(data)

        assert len(data.cell_values) == len(self.sort_indexes), f"Given cell values list has a different length from the sorted indexes, respectively found {len(data.cell_values)} and {len(self.sort_indexes)}."
        assert len(data.cell_colors) == len(self.sort_indexes), f"Given cell colors list has a different length from the sorted indexes, respectively found {len(data.cell_colors)} and {len(self.sort_indexes)}."
        assert len(data.cell_edge_colors) == len(self.sort_indexes), f"Given cell edge colors list has a different length from the sorted indexes, respectively found {len(data.cell_edge_colors)} and {len(self.sort_indexes)}."
        assert len(data.cell_ids) == len(self.sort_indexes), f"Given cell ID list has a different length from the sorted indexes, respectively found {len(data.cell_ids)} and {len(self.sort_indexes)}."
        
        data.cell_ids = np.array(data.cell_ids)[self.sort_indexes]
        data.cell_colors = np.array(data.cell_colors)[self.sort_indexes]
        data.cell_values = np.array(data.cell_values)[self.sort_indexes]
        data.cell_edge_colors = np.array(data.cell_edge_colors)[self.sort_indexes]
        
        if data.data_type == DataType.POLYGONS:
            data.polygons = [data.polygons[i] for i in self.sort_indexes]

    def reset_indexes(self, *args, **kwargs):
        """Clears the saved sort indexes.
        """
        self.sort_indexes = None