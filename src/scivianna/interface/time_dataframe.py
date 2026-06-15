import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Union

from scivianna.enums import UpdatePolicy
from scivianna.interface.generic_interface import Value1DAtLocation, CouplingInterface


class TimeDataFrame(Value1DAtLocation, CouplingInterface):
    def __init__(self, ):
        """Interface hosting a dataframe that is filled along a coupling
        """
        self.df = pd.DataFrame()
        self.time = -1
        self.update_policy = UpdatePolicy.APPEND_DATA

    def get_labels(self) -> List[str]:
        """Returns the fields names providable.

        Returns
        -------
        List[str]
            Fields names
        """
        labels = self.df.columns.tolist()
        if self.time != -1:
            # We are in a coupling, time exists
            labels += ["Time"]
        return labels

    def get_1D_value(
        self,
        position: Tuple[float, float, float],
        cell_index: str,
        material_name: str,
        field: str,
        options: Dict[str, Any] = None,
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
        options : Dict[str, Any], optional
            Additional options for 1D value computation.

        Returns
        -------
        Union[pd.Series, List[pd.Series]]
            Field value
        """
        if field == "Time":
            if "time" in options:
                return pd.Series(["min", "max"], index=[options["time"], options["time"]]).rename(
                    "Time"
                )
            else:
                return pd.Series([]).rename(
                    "Time"
                )

        if field in self.df.columns:
            return self.df[field]
        
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

    def append_data(self, key: str, data: Any):
        """Stores the data and associates it to the current time.

        Parameters
        ----------
        key : str
            Data associated key
        data : Any
            New value
        """
        if not key in self.df.columns:
            self.df.loc[:,key] = pd.Series([np.nan]*len(self.df), index=self.df.index)

        self.df.loc[self.time, key] = data

    def update_data(self, key: str, data: Any):
        """Replaces the interface data by the current value

        Parameters
        ----------
        key : str
            Data associated key
        data : Any
            New value
        """
        self.append_data(key=key, data=data)

    def update_mesh(self, key: str, data: Any):
        """Replaces the interface data and mesh by the current value

        Parameters
        ----------
        key : str
            Data associated key
        data : Any
            New value
        """
        # For TimeDataFrame, mesh is not applicable, treating same as update_data
        self.append_data(key=key, data=data)

    def append_mesh(self, key: str, data: Any):
        """Stores the data and mesh and associate them to the current time.

        Parameters
        ----------
        key : str
            Data associated key
        data : Any
            New value
        """
        # For TimeDataFrame, mesh is not applicable, treating same as append_data
        self.append_data(key=key, data=data)

    def get_template(self, name: str):
        """Returns the template for the C3PO getOutputxxxFieldTemplate functions

        Parameters
        ----------
        name : str
            Field name
        """
        # Return a pandas Series as template for TimeDataFrame
        return pd.Series(dtype=float)

    def set_template(self, name: str, template: Any):
        """Sets the template returned by C3PO getOutputxxxFieldTemplate functions

        Parameters
        ----------
        name : str
            Field name
        template : Any
            Object to set as template
        """
        # Templates are not used with TimeDataFrame, pass silently
        pass

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
            data = self.df, self.time

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

            self.df, self.time = data

