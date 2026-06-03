import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple, Union
from scivianna.interface.generic_interface import Value1DAtLocation, IcocoInterface


class TimeDataFrame(Value1DAtLocation, IcocoInterface):
    def __init__(self, ):
        """Interface hosting a dataframe that is filled along a coupling

        """
        self.df = pd.DataFrame()
        self.time = 0.

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
    
    def setTime(self, time:float):
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

    def setInputDoubleValue(self, name: str, val: float) -> None:
        """(Optional) Provide the code with a scalar double data.

        See Problem documentation for more details on the time semantic of a scalar value.

        Parameters
        ----------
        name : str
            name of the scalar value that is given to the code.
        val : float
            value passed to the code.

        Raises
        ------
        WrongArgument
            exception if the scalar name ('name' parameter) is invalid.
        WrongContext
            exception if called before initialize() or after terminate().
        """
        if not name in self.df.columns:
            self.df.loc[:,name] = pd.Series([np.nan]*len(self.df), index=self.df.index)

        self.df.loc[self.time, name] = val

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

