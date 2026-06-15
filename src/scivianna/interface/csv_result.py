import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Union
from scivianna.enums import UpdatePolicy
from scivianna.interface.generic_interface import ValueAtLocation, CouplingInterface



class CSVInterface(ValueAtLocation, CouplingInterface):
    def __init__(self, csv_file_path: str):
        """CSV file interface to get results from.

        Parameters
        ----------
        csv_file_path : str
            CSV file input path

        Raises
        ------
        ValueError
            File not found
        ValueError
            Cell column not found in the file columns
        """
        path = Path(csv_file_path)
        self.basename = path.name.replace(".csv", "")

        if not os.path.isfile(path):
            raise ValueError(f"Provided path does not exist : {csv_file_path}.")

        self.df = pd.read_csv(path)

        if "cell" not in self.df.columns:
            raise ValueError(
                f"cell column was not found in the csv columns. Found: {self.df.columns}."
            )

        # Store dataframes at each time step
        self.dfs: Dict[float, pd.DataFrame] = {}
        self.time = 0.
        self.update_policy = UpdatePolicy.APPEND_DATA

    def get_value(
        self,
        position: Tuple[float, float, float],
        cell_index: str,
        material_name: str,
        field: str,
        options: Dict[str, Any] = None,
    ):
        """Provides the result value of a field from either the (x, y, z) position or the cell index.

        Parameters
        ----------
        position : Tuple[float, float, float]
            Position at which the value is requested
        cell_index : str
            Index of the requested cell
        material_name : str
            Name of the requested material (ignored for CSV interface)
        field : str
            Requested field name
        options : Dict[str, Any], optional
            Additional options for value computation.

        Returns
        -------
        Union[str, float]
            Field value
        """
        if field not in self.df.columns:
            raise ValueError(
                f"Field {field} not found in dataframe columns, found : {self.df.columns}."
            )

        look_column = self.df["cell"]
        line_index = look_column[look_column == cell_index].index[0]
        return self.df.loc[line_index, field]

    def get_values(
        self,
        positions: List[Tuple[float, float, float]],
        cell_indexes: List[str],
        material_names: List[str],
        field: str,
        options: Dict[str, Any] = None,
    ) -> List[Union[str, float]]:
        """Provides the result values at different positions from either the (x, y, z) positions or the cell indexes.

        Parameters
        ----------
        positions : List[Tuple[float, float, float]]
            List of position at which the value is requested
        cell_indexes : List[str]
            Indexes of the requested cells
        material_names : List[str]
            Names of the requested materials (ignored for CSV interface)
        field : str
            Requested field name
        options : Dict[str, Any], optional
            Additional options for value computation.

        Returns
        -------
        List[Union[str, float]]
            Field values
        """
        if field not in self.df.columns:
            raise ValueError(
                f"Field {field} not found in dataframe columns, found : {self.df.columns}."
            )

        new_df = self.df.copy()
        new_df["cell"] = new_df["cell"].astype(str)
        new_df = new_df.set_index("cell")

        result = []
        for idx in cell_indexes:
            # Check for np.inf or float('inf')
            if isinstance(idx, float) and np.isinf(idx):
                result.append(np.nan)
            else:
                str_idx = str(idx)
                if str_idx in new_df.index:
                    result.append(new_df[field][str_idx])
                else:
                    result.append(np.nan)

        return result

    def get_fields(self) -> List[str]:
        """Returns the fields names providable.

        Returns
        -------
        List[str]
            Fields names
        """
        return [
            c
            for c in self.df.columns
            if c != "cell"
        ]

    def get_labels(self) -> List[str]:
        """Returns the fields names providable.

        Returns
        -------
        List[str]
            Fields names
        """
        return self.df.columns.tolist()

    def set_time(self, time: float):
        """This non-Icoco function allows setting the current time in an interface to associate to the received value.

        Parameters
        ----------
        time : float
            Current time
        """
        self.time = time

        # If no dataframe exists for this time, create one from the initial structure
        if time not in self.dfs:
            self.dfs[time] = self.df.copy()
        self.df = self.dfs[time]

    def append_data(self, key: str, data: Any):
        """Stores the data and associates it to the current time.
        Replaces the whole dataframe at the current time step.

        Parameters
        ----------
        key : str
            Data associated key
        data : Any
            New value
        """
        # Store the current dataframe at the current time
        self.dfs[self.time] = self.df.copy()

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
        self.append_data(key=key, data=data)

    def append_mesh(self, key: str, data: Any):
        """Stores the data and mesh and associate them to the current time.
        Replaces the whole dataframe at the current time step.

        Parameters
        ----------
        key : str
            Data associated key
        data : Any
            New value
        """
        self.append_data(key=key, data=data)

    def get_template(self, name: str):
        """Returns the template for the C3PO getOutputxxxFieldTemplate functions

        Parameters
        ----------
        name : str
            Field name
        """
        # Return a pandas Series as template for CSVInterface
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
        # Templates are not used with CSVInterface, pass silently
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
            data = self.dfs, self.basename, self.time

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

            self.dfs, self.basename, self.time = data
            
            # Restore current dataframe from the last time step
            if self.dfs:
                self.df = self.dfs.get(self.time, list(self.dfs.values())[-1])