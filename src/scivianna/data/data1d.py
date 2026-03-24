from typing import Callable, Dict, List, Any, Literal, Tuple, Union
import numpy as np
import pandas as pd

from scivianna.enums import DataType
from scivianna.data.data_container import DataContainer

class Data1D(DataContainer):
    """Data class containing the 2D geometry data"""

    line_ids:np.ndarray[Union[int, str]]
    """List of contained line ids"""
    line_values:np.ndarray[pd.Series]
    """List of contained line values"""
    line_colors:np.ndarray[Tuple[int, int, int]]
    """List of contained line colors"""
    line_styles:np.ndarray[str]
    """List of contained line edge colors"""
    
    def __init__(self):
        """ Empty constructor of the Data1D class.
        """
        self.line_ids = np.array([])
        self.line_values = np.array([])
        self.line_colors = np.array([])
        self.line_styles = np.array([])

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        """Build a Data1D object from a list of PolygonElement

        Parameters
        ----------
        df : pd.DataFrame
            Pandas dataframe to build the Data1D from

        Returns
        -------
        Data1D
            Requested Data1D
        """
        data_ = Data1D()
        data_.line_ids = list(df.columns)
        data_.line_values = [df[key] for key in df.columns]

        data_.line_colors = [None for _ in len(df.columns)]
        data_.line_styles = [None for _ in len(df.columns)]
        
        return data_

    @classmethod
    def from_serie_dict(cls, series: Dict[str, pd.Series]):
        """Build a Data1D object from a list of PolygonElement

        Parameters
        ----------
        series : Dict[str, pd.Series]
            Dict of pandas Series to build the Data1D with

        Returns
        -------
        Data1D
            Requested Data1D
        """
        assert isinstance(series, dict), f"Provided series must be a dictionnary, found {type(series)}"
        data_ = Data1D()

        data_.line_ids = list(series.keys())
        data_.line_values = list(series.values())

        data_.line_colors = [None for _ in len(data_.line_ids)]
        data_.line_styles = [None for _ in len(data_.line_ids)]

        return data_
    
    def copy(self,) -> "Data1D":
        """Returns a copy of self

        Returns
        -------
        Data1D
            Identical copy of self
        """
        data1D = Data1D()
        data1D.line_ids = self.line_ids.copy()
        data1D.line_values = self.line_values.copy()
        data1D.line_colors = self.line_colors.copy()
        data1D.line_styles = self.line_styles.copy()

        return data1D

    def check_valid(self,):
        """Checks if this Data1D is valid, raises an AssertionError otherwise
        """
        assert len(self.line_ids) == len(self.line_colors), "The Data1D object must have the same number of line id and colors"
        assert len(self.line_values) == len(self.line_colors), "The Data1D object must have the same number of line values and colors"
        assert len(self.line_values) == len(self.line_styles), "The Data1D object must have the same number of line values and styles"
        
