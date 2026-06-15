import multiprocessing as mp
from pathlib import Path
import pandas as pd
from typing import Any, List, Tuple, Dict, Union

from scivianna.data.data2d import Data2D
from scivianna.enums import VisualizationMode, GeometryType, DataType

from typing import TYPE_CHECKING

#   TYPE_CHECKING : Allows fake import of modules pylance work without importing them
if TYPE_CHECKING:
    import medcoupling

from scivianna.constants import MESH, MATERIAL


class GenericInterface:
    """ Generic interface class that implement basic functions. This class mutualises functions that are shared between its child classes.
    """

    extensions = []
    """Extensions associated to this interface."""

    def read_file(self, file_path: str, file_label: str) -> None:
        """Read a file and store its content in the interface

        Parameters
        ----------
        file_path : str
            File to read
        file_label : str
            Label to define the file type

        Raises
        ------
        NotImplementedError
            Function to override in the code interfaces
        """
        raise NotImplementedError()

    def get_labels(self) -> List[str]:
        """Returns a list of fields names displayable with this interface

        Returns
        -------
        List[str]
            List of fields names
        """
        return [MESH, MATERIAL]

    def get_label_coloring_mode(self, label: str) -> VisualizationMode:
        """Returns wheter the given field is colored based on a string value or a float.

        Parameters
        ----------
        label : str
            Field to color name

        Returns
        -------
        VisualizationMode
            Coloring mode

        Raises
        ------
        NotImplementedError
            Function to override in the code interfaces
        """
        raise NotImplementedError()

    def get_file_input_list(self) -> List[Tuple[str, str]]:
        """Returns a list of file label and its description for the GUI

        Returns
        -------
        List[Tuple[str, str]]
            List of (file label, description)

        Raises
        ------
        NotImplementedError
            The function was not implemented in the code interface.
        """
        raise NotImplementedError()

    @classmethod
    def serialize(self, obj: Any, key: str) -> Any:
        """This function receives an object that is about to be transmitted at the given key.
        -   If the object can be passed through a python multiprocessing Queue, it can be returned.
        -   If the object can't, it is serialized, and the code returns the file path.

        The read function will then be able to expect the returned object at the given key.

        By default, this class returns the obj, if the cobject can't be passed, overwrite this function.

        Parameters
        ----------
        obj : Any
            Object that is sent to the generic interface
        key : str
            Key associated to the object

        Returns
        -------
        Any
            Object transmissible through a multiprocessing Queue associated to the given object.
        """
        return obj

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
        raise NotImplementedError()

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
        raise NotImplementedError()

    @classmethod
    def get_slave(cls):
        from scivianna.slave import ComputeSlave
        return ComputeSlave(cls)


class Geometry2D(GenericInterface):
    """ Interface parent class for classes that can compute geometry 2D slices.
    """
    geometry_type: GeometryType
    """Enum telling if the geometry is 2D or 3D (Displays the axis card and the w coordinate in the GUI)."""
    data_type: DataType
    """Enum saying if the data are returned in a 2D grid or a polygon list"""
    rasterized: bool = False
    """Boolean telling if the geometry is made by rasterizing a 2D grid (displays the line count in the GUI)."""

    def compute_2D_data(
        self,
        u: Tuple[float, float, float],
        v: Tuple[float, float, float],
        u_min: float,
        u_max: float,
        v_min: float,
        v_max: float,
        w_value: float,
        q_tasks: mp.Queue,
        options: Dict[str, Any],
        caller: str = "API",
    ) -> Tuple[Data2D, bool]:
        """Returns a list of polygons that defines the geometry in a given frame

        Parameters
        ----------
        u : Tuple[float, float, float]
            Horizontal coordinate director vector
        v : Tuple[float, float, float]
            Vertical coordinate director vector
        u_min : float
            Lower bound value along the u axis
        u_max : float
            Upper bound value along the u axis
        v_min : float
            Lower bound value along the v axis
        v_max : float
            Upper bound value along the v axis=
        w_value : float
            Value along the u ^ v axis
        q_tasks : mp.Queue
            Queue from which get orders from the master.
        options : Dict[str, Any]
            Additional options for frame computation.
        caller : str
            Identifier of the caller requesting the computation (default: "API")

        Returns
        -------
        Data2D
            Geometry to display
        bool
            Were the polygons updated compared to the past call

        Raises
        ------
        NotImplementedError
            Function to override in the code interfaces
        """
        raise NotImplementedError()

    def get_value_dict(
        self, value_label: str, cells: List[Union[int, str]], options: Dict[str, Any], caller: str = "API"
    ) -> Dict[Union[int, str], str]:
        """Returns a cell name - field value map for a given field name

        Parameters
        ----------
        value_label : str
            Field name to get values from
        cells : List[Union[int,str]]
            List of cells names
        options : Dict[str, Any]
            Additional options for frame computation.
        caller : str
            Identifier of the caller requesting the computation (default: "API")

        Returns
        -------
        Dict[Union[int,str], str]
            Field value for each requested cell names

        Raises
        ------
        NotImplementedError
            Function to override in the code interfaces
        """
        raise NotImplementedError()


class Geometry2DPolygon(Geometry2D):
    """ Interface parent class for classes that can compute geometry 2D slices and provide a list of polygons.
    """


class Geometry2DGrid(Geometry2D):
    """ Interface parent class for classes that can compute geometry 2D slices and provide a numpy array.
    """
    rasterized: bool = True
    """Boolean telling if the geometry is made by rasterizing a 2D grid (displays the line count in the GUI)."""


class ValueAtLocation(GenericInterface):
    """ Interface parent class to implement a function to get values at a specific location.
    """
    def get_value(
        self,
        position: Tuple[float, float, float],
        cell_index: str,
        material_name: str,
        field: str,
        options: Dict[str, Any] = None,
    ) -> Union[str, float]:
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
        Union[str, float]
            Field value
        """
        raise NotImplementedError()

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
        raise NotImplementedError()


class Value1DAtLocation(GenericInterface):
    """ Interface parent class to implement a function to get 1D data at a specific location.
    """
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
        raise NotImplementedError()


class OverLine(GenericInterface):
    """ Interface parent class to implement a function to compute a field value along a 1D line.
    """
    def compute_1D_line_data(
        self,
        pos: Tuple[float, float, float],
        u: Tuple[float, float, float],
        d: float,
        q_tasks: mp.Queue,
        options: Dict[str, Any],
    ) -> pd.DataFrame:
        """Returns a list of polygons that defines the geometry in a given frame

        Parameters
        ----------
        pos : Tuple[float, float, float]
            1D data line start location
        u : Tuple[float, float, float]
            Data line direction vector
        d : float
            Distance to travel by the 1D line
        q_tasks : mp.Queue
            Queue from which get orders from the master.
        options : Dict[str, Any]
            Additional options for frame computation.

        Returns
        -------
        pd.DataFrame
            Pandas dataframe containing the data

        Raises
        ------
        NotImplementedError
            Function to override in the code interfaces
        """
        raise NotImplementedError()


class CouplingInterface(GenericInterface):
    """ Interface parent class to implement the C3PO functions required for a code coupling visualization.
    """
    def set_time(self, time: float):
        """This non-Icoco function allows setting the current time in an interface to associate to the received value.

        Parameters
        ----------
        time : float
            Current time
        """
        raise NotImplementedError

    def update_data(self, key: str, data: Any):
        """Replaces the interface data by the current value

        Parameters
        ----------
        key : str
            Data associated key
        data : Any
            New value
        """
        raise NotImplementedError()

    def append_data(self, key: str, data: Any):
        """Stores the data and associates it to the current time.

        Parameters
        ----------
        key : str
            Data associated key
        data : Any
            New value
        """
        raise NotImplementedError()

    def update_mesh(self, key: str, data: Any):
        """Replaces the interface data and mesh by the current value

        Parameters
        ----------
        key : str
            Data associated key
        data : Any
            New value
        """
        raise NotImplementedError()

    def append_mesh(self, key: str, data: Any):
        """Stores the data and mesh and associate them to the current time.

        Parameters
        ----------
        key : str
            Data associated key
        data : Any
            New value
        """
        raise NotImplementedError()

    def get_template(self, name: str):
        """Returns the template for the C3PO getOutputxxxFieldTemplate functions

        Parameters
        ----------
        name : str
            Field name
        """
        raise NotImplementedError(f"get_template function not implemented for class {__class__.__name__}")

    def set_template(self, name: str, template: Any):
        """Sets the template returned by C3PO getOutputxxxFieldTemplate functions

        Parameters
        ----------
        name : str
            Field name
        template : Any
            Object to set as template
        """
        pass
        
