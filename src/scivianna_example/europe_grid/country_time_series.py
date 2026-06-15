import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from typing import Any, Dict, List, Tuple, Union
from scivianna.interface.generic_interface import ValueAtLocation, Value1DAtLocation
from scivianna.enums import VisualizationMode


class CountryTimeSeriesInterface(ValueAtLocation, Value1DAtLocation):
    def __init__(self, ):
        """CSV file interface to get results from.
        """
        pass

    def read_file(self, file_path:str, file_label:str):
        """Read CSV file to get results from.

        Parameters
        ----------
        file_path : str
            CSV file input path

        """
        path = Path(file_path)
        self.df = pd.read_csv(path)
        
        self.country_codes = []
        self.fields = []
        for col in self.df.columns:
            if not (col.startswith("Unnamed") or col=="Time"):
                if "load" in col:
                    country_code = col[:2].upper()
                    self.country_codes.append(country_code)
                
                field_name = col[3:]
                self.fields.append(field_name)
                
        self.country_codes = list(set(self.country_codes))
        self.fields = list(set(self.fields))

    def get_value(
        self,
        position: Tuple[float, float, float],
        cell_index: str,
        material_name: str,
        field: str,
        options: Dict[str, Any] = None,
    ):
        """Provides the result value of a field from either the (x, y, z) position, the cell index, or the material name.

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
            Additional options for value computation.

        Returns
        -------
        List[Union[str, float]]
            Field value
        """
        if cell_index in self.country_codes:
            return 1.
        else:
            return 0.

    def get_values(
        self,
        positions: List[Tuple[float, float, float]],
        cell_indexes: List[str],
        material_names: List[str],
        field: str,
        options: Dict[str, Any] = None,
    ) -> List[Union[str, float]]:
        """Provides the result values at different positions from either the (x, y, z) positions, the cell indexes, or the material names.

        Parameters
        ----------
        positions : List[Tuple[float, float, float]]
            List of position at which the value is requested
        cell_indexes : List[str]
            Indexes of the requested cells
        material_names : List[str]
            Names of the requested materials
        field : str
            Requested field name
        options : Dict[str, Any], optional
            Additional options for value computation.

        Returns
        -------
        List[Union[str, float]]
            Field values
        """
        output = []

        for vol_id in cell_indexes:
            if vol_id in self.country_codes:
                val = 0.

                for column in self.df.columns:
                    if column.startswith(vol_id.lower()) and column.endswith(field):
                        val += self.df[column].sum()
                output.append(val)
            else:
                output.append(np.nan)
        return output

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
        output = None

        if cell_index in self.country_codes:
            for column in self.df.columns:
                if column.startswith(cell_index.lower()) and column.endswith(field):
                    if output is None:
                        output = self.df[column].copy()
                    else:
                        output += self.df[column]

        if output is None:         
            output = self.df["Time"].copy()*0.
            output.replace(0., np.nan)

        output.rename(f"{cell_index}_{field}")

        return pd.Series(output)

    def get_fields(self) -> List[str]:
        """Returns the fields names providable.

        Returns
        -------
        List[str]
            Fields names
        """
        return self.fields
    
    def get_labels(self) -> List[str]:
        """Returns the fields names providable.

        Returns
        -------
        List[str]
            Fields names
        """
        return self.get_fields()

    def get_1D_fields(self) -> List[str]:
        """Returns the fields names providable.

        Returns
        -------
        List[str]
            Fields names
        """
        return self.fields

    def get_label_coloring_mode(self, label: str) -> VisualizationMode:
        """Returns the coloring mode of the field.

        Parameters
        ----------
        label : str
            Field name

        Returns
        -------
        VisualizationMode
            Coloring mode
        """
        return VisualizationMode.FLOAT

    def get_file_input_list(self) -> List[Tuple[str, str]]:
        """Returns a list of file label and its description for the GUI.

        Returns
        -------
        List[Tuple[str, str]]
            List of (file label, description)
        """
        return [("csv", "CSV data file")] if hasattr(self, 'df') else []

    def save(self, file_path: Path, include_files: bool):
        """Pickle saves the slave content to a file.

        Parameters
        ----------
        file_path : Path
            File in which save the file
        include_files : bool
            Not used for this interface (CSV data is already in df)
        """
        state = {
            'df': self.df,
            'country_codes': self.country_codes,
            'fields': self.fields,
        }
        with open(file_path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, file_path: Path, include_files: bool):
        """Pickle loads the slave content from a file.

        Parameters
        ----------
        file_path : Path
            File from which load the slave
        include_files : bool
            Not used for this interface (CSV data is already in df)
        """
        with open(file_path, 'rb') as f:
            state = pickle.load(f)
        self.df = state['df']
        self.country_codes = state['country_codes']
        self.fields = state['fields']
