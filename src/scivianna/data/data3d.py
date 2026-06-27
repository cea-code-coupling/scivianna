from typing import List, Tuple, Union
import numpy as np
from scivianna.data.data_container import DataContainer

import vtk


class Data3D(DataContainer):
    """Data class containing the 3D geometry data""" 
    polydata: vtk.vtkPolyData
    """VTK polydata defining the geometry""" 
    cell_ids: List[Union[int, str]]
    """List of contained cell ids""" 
    cell_values: List[Union[float, str]]
    """List of contained cell values""" 
    cell_colors: List[Tuple[int, int, int]]
    """List of contained cell colors""" 
    cell_edge_colors: List[Tuple[int, int, int]]
    """List of contained cell edge colors""" 
    
    def __init__(self):
        """Empty constructor of the Data3D class.""" 
        self.polydata = None 
        self.cell_ids = [] 
        self.cell_values = [] 
        self.cell_colors = [] 
        self.cell_edge_colors = [] 
   
    @classmethod 
    def from_polydata(cls, polydata: vtk.vtkPolyData):
        """Build a Data3D object from a vtk.vtkPolyData 
        
        Parameters 
        ---------- 
        polydata : vtk.vtkPolyData 
            VTK polydata defining the geometry 
            
        Returns 
        ------- 
        Data3D 
            Requested Data3D
        """
        data_ = Data3D()
        data_.polydata = polydata
        data_.cell_ids = [p.cell_id for p in polydata]
        data_.cell_values = [np.nan] * len(polydata)
        data_.cell_colors = np.zeros((len(polydata), 4)) + 255
        data_.cell_edge_colors = np.zeros((len(polydata), 4)) + 50
        return data_
    
    def copy(self) -> "Data3D":
        """Returns a deep copy of self."""
        data3D = Data3D()

        poly = vtk.vtkPolyData()
        poly.DeepCopy(self.polydata)
        data3D.polydata = poly

        data3D.cell_ids = list(self.cell_ids)
        data3D.cell_values = list(self.cell_values)
        data3D.cell_colors = np.array(self.cell_colors, copy=True).tolist()
        data3D.cell_edge_colors = np.array(self.cell_edge_colors, copy=True).tolist()

        return data3D

    def check_valid( self, ):
        """Checks if this Data3D is valid, raises an AssertionError otherwise"""
        assert len(self.cell_ids) == len( self.cell_colors ), "The Data3D object must have the same number of cell id and colors"
        assert len(self.cell_values) == len( self.cell_colors ), "The Data3D object must have the same number of cell values and colors"
        assert len(self.cell_values) == len( self.cell_edge_colors ), "The Data3D object must have the same number of cell values and edge colors"
        assert len(self.cell_values) == self.polydata.GetPolys().GetNumberOfCells(), "The Data3D object must have the same number of cell values and polygons"

        if any(isinstance(item, str) for item in self.cell_values):
            assert all( isinstance(item, str) for item in self.cell_values ), "If any of the values is a string, they all must be strings"

    def update_cell_data(self):
        """Updates the cell data stored in the polydata."""

        self.check_valid()

        # IDs
        self.polydata.cell_data["cell_id"] = np.asarray(self.cell_ids)

        # Values
        self.polydata.cell_data["cell_value"] = np.asarray(self.cell_values)

        # RGB colors (drop alpha if present and normalize to [0, 1])
        colors = np.asarray(self.cell_colors, dtype=float)

        if colors.shape[1] == 4:
            colors = colors[:, :3]

        self.polydata.cell_data["rgb"] = colors / 255.0

        # Edge colors (optional)
        edge_colors = np.asarray(self.cell_edge_colors, dtype=float)

        if edge_colors.shape[1] == 4:
            edge_colors = edge_colors[:, :3]

        self.polydata.cell_data["edge_rgb"] = edge_colors / 255.0
