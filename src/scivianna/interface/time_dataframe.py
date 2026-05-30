import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Tuple, Union

from scivianna.enums import UpdatePolicy
from scivianna.interface.generic_interface import Value1DAtLocation, CouplingInterface


class TimeDataFrame(Value1DAtLocation, CouplingInterface):
    def __init__(self, ):
        """Interface hosting a dataframe that is filled along a coupling

        """
        self.df = pd.DataFrame()
        self.time = 0.
        self.update_policy = UpdatePolicy.APPEND_DATA

    def get_labels(self) -> List[str]:
        """Returns the fields names providable.

        Returns
        -------
        List[str]
            Fields names
        """
        return self.df.columns.tolist()

    def get_1D_value(
        self,
        position: Tuple[float, float, float],
        cell_index: str,
        material_name: str,
        field: str,
    ) -> Union[pd.Series, List[pd.Series]]:
        """Provides the 1D value of a field from either the (x, y, z) position, the cell index, or the material name.

        Parameters
        ----------
        position : Tuple[float, float, float]
            Position at which the value is requested
        cell_index : str
            Index of the requested cell
        material_name : str
            Name of the requested material
        field : str
            Requested field name

        Returns
        -------
        Union[pd.Series, List[pd.Series]]
            Field value
        """
        if field in self.df.columns:
            return self.df[field]
        else:
            raise ValueError(f"Field {field} not found, dataframe contains {self.df.columns.tolist()}")
    
    def set_time(self, time:float):
        """This non-Icoco function allows setting the current time in an interface to associate to the received value.

        Parameters
        ----------
        time : float
            Current time
        """
        self.time = time

        if not time in self.df.index:
            self.df = pd.concat([
                self.df,
                pd.DataFrame({
                    col:[np.nan] for col in self.df.columns
                }, index = [self.time])
            ])

    def append_data(self, name: str, value: float):
        """Replaces the interface data by the current value

        Parameters
        ----------
        key : str
            Data associated key
        data : Any
            New value
        """
        if name not in self.df.columns:
            self.df[name] = np.nan
            
        self.df.loc[len(df), name] = val
