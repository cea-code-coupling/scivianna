import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Tuple, Union
from scivianna.interface.generic_interface import ValueAtLocation


class CSVInterface(ValueAtLocation):
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
                f"Cell column was not found in the csv columns. Found: {self.df.columns}."
            )

    def get_value(
        self,
        position: Tuple[float, float, float],
        cell_index: str,
        material_name: str,
        field: str,
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
            self.basename + "_" + c
            for c in self.df.columns
            if c != "cell"
        ]