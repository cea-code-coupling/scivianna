import atexit
import panel as pn
from pathlib import Path
import os
import queue
import multiprocessing as mp
import dill
import traceback
from threading import Lock

import pandas as pd
from typing import Any, List, Dict, Tuple, Type, Union

from scivianna.data.data2d import Data2D
from scivianna.data.data3d import Data3D

from scivianna.interface.generic_interface import (
    GenericInterface,
    Geometry2D,
    CouplingInterface,
    Geometry3D,
    OverLine,
    ValueAtLocation,
    Value1DAtLocation
)
from scivianna.enums import GeometryType, VisualizationMode

from typing import TYPE_CHECKING

#   TYPE_CHECKING : Allows fake import of modules pylance work without importing them
if TYPE_CHECKING:
    import medcoupling

profile_time = bool(os.environ["VIZ_PROFILE"]) if "VIZ_PROFILE" in os.environ else 0

pn.extension(notifications=True)


class SlaveCommand:
    """Class defining the available commands that are forwarded to the compute slaves"""

    #   GenericInterface functions
    READ_FILE = "read_file"
    """Reads an input file"""
    GET_LABELS = "get_labels"
    """Returns the list of displayable fields"""
    GET_LABEL_COLORING_MODE = "get_label_coloring_mode"
    """Returns the coloring mode of a field"""
    GET_FILE_INPUT_LIST = "get_file_input_list"
    """Returns the list of read input files"""
    SAVE = "save"
    """Pickle saves the slave state to a file"""
    LOAD = "load"
    """Pickle loads the slave state from a file"""

    #   Geometry2D functions
    COMPUTE_2D_DATA = "compute_2d_data"
    """Compute a 2D slice of the geometry"""
    GET_VALUE_DICT = "get_value_dict"
    """Returns the values of a field at cells"""
    GET_GEOMETRY_TYPE = "get_geometry_type"
    """Returns the geometry type"""

    #   Geometry3D functions
    COMPUTE_3D_DATA = "compute_3d_data"
    """Compute a 3D slice of the geometry"""
    GET_3D_VALUE_DICT = "get_3d_value_dict"
    """Returns the values of a field at cells"""

    #   ValueAtLocation functions
    GET_VALUE = "get_value"
    """Returns the value at a location/cell"""
    GET_VALUES = "get_values"
    """Returns the value at a set of locations/cells"""

    #   Value1DAtLocation functions
    GET_1D_VALUE = "get_1D_value"
    """Returns the 1Dvalue at a location/cell"""

    #   OverLine functions
    COMPUTE_1D_LINE_DATA = "compute_1d_line_data"
    """Compute a 1D result along a line"""

    #   CouplingInterface functions
    UPDATE_DATA = "update_data"
    """Updates interface data"""
    APPEND_DATA = "append_data"
    """Appends interface data"""
    UPDATE_MESH = "update_mesh"
    """Updates interface mesh"""
    APPEND_MESH = "append_mesh"
    """Appends interface mesh"""
    GET_TEMPLATE = "get_template"
    """Gets field template"""
    SET_TEMPLATE = "set_template"
    """Sets field template"""
    SET_TIME = "set_time"
    """Sets the current time"""
    GET_UPDATE_POLICY = "get_update_policy"
    """Gets the update policy attribute"""
    SET_UPDATE_POLICY = "set_update_policy"
    """Sets the update policy attribute"""

    CUSTOM = "custom"
    """Custom call to transfer to the interface to ease extension development"""


def worker(
    q_tasks: mp.Queue,
    q_returns: mp.Queue,
    q_errors: mp.Queue,
    code_interface: Type[GenericInterface],
):
    """Creates a worker that will forward the panel requests to the GenericInterface on another process

    Parameters
    ----------
    q_tasks : mp.Queue
        Queue containing the tasks
    q_returns : mp.Queue
        Queue to return the results
    code_interface : Type[GenericInterface]
        GenericInterface to instanciate.
    """
    code_: GenericInterface = code_interface()

    while True:
        try:
            task, data = q_tasks.get(timeout=0.1)  # Block for up to 100ms

            #   GenericInterface functions
            if task == SlaveCommand.READ_FILE:
                file_path, file_label = data
                code_.read_file(file_path=file_path, file_label=file_label)
                q_returns.put("OK")

            elif task == SlaveCommand.GET_LABELS:
                labels = code_.get_labels()
                q_returns.put(labels)

            elif task == SlaveCommand.GET_LABEL_COLORING_MODE:
                field_name = data
                set_return = code_.get_label_coloring_mode(label=field_name)
                q_returns.put(set_return)

            elif task == SlaveCommand.GET_FILE_INPUT_LIST:
                input_list = code_.get_file_input_list()
                q_returns.put(input_list)

            elif task == SlaveCommand.SAVE:
                file_path, include_files = data
                code_.save(file_path=file_path, include_files=include_files)
                q_returns.put("OK")

            elif task == SlaveCommand.LOAD:
                file_path, include_files = data
                code_.load(file_path=file_path, include_files=include_files)
                q_returns.put("OK")

            #
            #   Geometry2D functions
            elif task == SlaveCommand.COMPUTE_2D_DATA:
                (
                    u,
                    v,
                    origin,
                    size_u,
                    size_v,
                    q_tasks_,
                    coloring_label,
                    options,
                    caller,
                ) = data

                if not isinstance(code_, Geometry2D):
                    raise TypeError(
                        f"The requested panel is not associated to an Geometry2D, found class {type(code_)}."
                    )
                data: Data2D
                data, polygons_updated = code_.compute_2D_data(
                    u=u,
                    v=v,
                    origin=origin,
                    size_u=size_u,
                    size_v=size_v,
                    q_tasks=q_tasks_,
                    options=options,
                    caller=caller,
                )

                dict_value_per_cell = code_.get_value_dict(
                    value_label=coloring_label,
                    cells=data.cell_ids,
                    options=options,
                    caller=caller,
                )

                data.cell_values = [dict_value_per_cell[v] for v in data.cell_ids]

                q_returns.put(
                    [
                        data,
                        polygons_updated,
                    ]
                )

            elif task == SlaveCommand.GET_VALUE_DICT:
                if not isinstance(code_, Geometry2D):
                    raise TypeError(
                        f"The requested panel is not associated to an Geometry2D, found class {type(code_)}."
                    )
                value_label, cells, options, caller = data
                set_return = code_.get_value_dict(
                    value_label=value_label,
                    cells=cells,
                    options=options,
                    caller=caller,
                )
                q_returns.put(set_return)

            elif task == SlaveCommand.GET_GEOMETRY_TYPE:
                q_returns.put(code_.geometry_type)

            #   Geometry3D functions
            elif task == SlaveCommand.COMPUTE_3D_DATA:
                coloring_label, options = data
                if not isinstance(code_, Geometry3D):
                    raise TypeError(
                        f"The requested panel is not associated to an Geometry3D, found class {type(code_)}."
                    )
                data: Data3D
                data, polygons_updated = code_.compute_3D_data(
                    options=options,
                )

                dict_value_per_cell = code_.get_3d_value_dict(
                    value_label=coloring_label,
                    cells=data.cell_ids,
                    options=options,
                    caller=options.get("caller", "API"),
                )

                data.cell_values = [dict_value_per_cell[v] for v in data.cell_ids]

                q_returns.put(
                    [
                        data,
                        polygons_updated,
                    ]
                )
            elif task == SlaveCommand.GET_3D_VALUE_DICT:
                if not isinstance(code_, Geometry3D):
                    raise TypeError(
                        f"The requested panel is not associated to an Geometry3D, found class {type(code_)}."
                    )
                value_label, cells, options, caller = data
                set_return = code_.get_3d_value_dict(
                    value_label=value_label,
                    cells=cells,
                    options=options,
                    caller=caller,
                )
                q_returns.put(set_return)


            #   ValueAtLocation functions
            elif task == SlaveCommand.GET_VALUE:
                if not isinstance(code_, ValueAtLocation):
                    raise TypeError(
                        f"The requested panel is not associated to an ValueAtLocation, found class {type(code_)}."
                    )
                position, cell_index, material_name, field, options = data
                set_return = code_.get_value(
                    position=position,
                    cell_index=cell_index,
                    material_name=material_name,
                    field=field,
                    options=options,
                )
                q_returns.put(set_return)

            elif task == SlaveCommand.GET_VALUES:
                if not isinstance(code_, ValueAtLocation):
                    raise TypeError(
                        f"The requested panel is not associated to an ValueAtLocation, found class {type(code_)}."
                    )
                positions, cell_indexes, material_names, field, options = data
                set_return = code_.get_values(
                    positions=positions,
                    cell_indexes=cell_indexes,
                    material_names=material_names,
                    field=field,
                    options=options,
                )
                q_returns.put(set_return)

            #
            #   Value1DAtLocation functions
            elif task == SlaveCommand.GET_1D_VALUE:
                if not isinstance(code_, Value1DAtLocation):
                    raise TypeError(
                        f"The requested panel is not associated to an Value1DAtLocation, found class {type(code_)}."
                    )
                position, cell_index, material_name, field, options = data
                input_list = code_.get_1D_value(
                    position=position,
                    cell_index=cell_index,
                    material_name=material_name,
                    field=field,
                    options=options,
                )
                q_returns.put(input_list)

            #
            #   OverLine functions
            elif task == SlaveCommand.COMPUTE_1D_LINE_DATA:
                if not isinstance(code_, OverLine):
                    raise TypeError(
                        f"The requested panel is not associated to an OverLine, found class {type(code_)}."
                    )
                pos, u, d, q_tasks_, options = data
                input_list = code_.compute_1D_line_data(
                    pos=pos,
                    u=u,
                    d=d,
                    q_tasks_=q_tasks_,
                    options=options,
                )
                q_returns.put(input_list)

            #   CouplingInterface functions
            elif task == SlaveCommand.UPDATE_DATA:
                key, data_value = data
                if not isinstance(code_, CouplingInterface):
                    raise TypeError(
                        f"The requested panel is not associated to an CouplingInterface, found class {type(code_)}."
                    )
                set_return = code_.update_data(key=key, data=data_value)
                q_returns.put(set_return)

            elif task == SlaveCommand.APPEND_DATA:
                key, data_value = data
                if not isinstance(code_, CouplingInterface):
                    raise TypeError(
                        f"The requested panel is not associated to an CouplingInterface, found class {type(code_)}."
                    )
                set_return = code_.append_data(key=key, data=data_value)
                q_returns.put(set_return)

            elif task == SlaveCommand.UPDATE_MESH:
                key, data_value = data
                if not isinstance(code_, CouplingInterface):
                    raise TypeError(
                        f"The requested panel is not associated to an CouplingInterface, found class {type(code_)}."
                    )
                set_return = code_.update_mesh(key=key, data=data_value)
                q_returns.put(set_return)

            elif task == SlaveCommand.APPEND_MESH:
                key, data_value = data
                if not isinstance(code_, CouplingInterface):
                    raise TypeError(
                        f"The requested panel is not associated to an CouplingInterface, found class {type(code_)}."
                    )
                set_return = code_.append_mesh(key=key, data=data_value)
                q_returns.put(set_return)

            elif task == SlaveCommand.GET_TEMPLATE:
                name = data
                if not isinstance(code_, CouplingInterface):
                    raise TypeError(
                        f"The requested panel is not associated to an CouplingInterface, found class {type(code_)}."
                    )
                set_return = code_.get_template(name=name)
                q_returns.put(set_return)

            elif task == SlaveCommand.SET_TEMPLATE:
                name, template = data
                if not isinstance(code_, CouplingInterface):
                    raise TypeError(
                        f"The requested panel is not associated to an CouplingInterface, found class {type(code_)}."
                    )
                set_return = code_.set_template(name=name, template=template)
                q_returns.put(set_return)

            elif task == SlaveCommand.SET_TIME:
                time_ = data[0]
                if not isinstance(code_, CouplingInterface):
                    raise TypeError(
                        f"The requested panel is not associated to an CouplingInterface, found class {type(code_)}."
                    )
                set_return = code_.set_time(time=time_)
                q_returns.put(set_return)

            elif task == SlaveCommand.GET_UPDATE_POLICY:
                if not isinstance(code_, CouplingInterface):
                    raise TypeError(
                        f"The requested panel is not associated to an CouplingInterface, found class {type(code_)}."
                    )
                set_return = code_.update_policy
                q_returns.put(set_return)

            elif task == SlaveCommand.SET_UPDATE_POLICY:
                if not isinstance(code_, CouplingInterface):
                    raise TypeError(
                        f"The requested panel is not associated to an CouplingInterface, found class {type(code_)}."
                    )
                code_.update_policy = data
                q_returns.put("OK")

            elif task == SlaveCommand.CUSTOM:
                function_name, arguments = data
                q_returns.put(code_.__getattribute__(function_name)(**arguments))

        except queue.Empty:
            continue

        except Exception as e:
            traceback.print_exc()
            q_errors.put(e)


class ComputeSlave:
    """Class that creates a subprocess to interface with the code."""


    def __init__(self, code_interface: Type[GenericInterface], allow_errors: bool = False):
        """ComputeSlave constructor

        Parameters
        ----------
        code_interface : Type[GenericInterface]
            Class of the GenericInterface
        allow_errors : bool
            If True, a notification is sent when an error is reached, if False, the error is raised
        """
        self.p: mp.Process = None
        """ Subprocess hosting the worker
        """
        self.q_tasks: mp.Queue = None
        """ Queue in which the tasks are pushed
        """
        self.q_returns: mp.Queue = None
        """ Queue to get the results
        """
        self.code_interface: Type[GenericInterface] = code_interface
        """ Code interface class
        """
        self.file_read: List[Tuple[str, str]] = []
        """ List of file read and their associated key.
        """
        self.allow_errors = allow_errors
        self._lock = Lock()

        self.running = False
        self.reset()

    def reset(
        self,
    ):
        """Kills the worker and create a new one."""
        print("RESETING SLAVE.")
        if self.p is not None:
            self.p.kill()
            self.p.join()

        self.q_tasks = mp.Queue()
        self.q_returns = mp.Queue()
        self.q_errors = mp.Queue()
        self.p = mp.Process(
            target=worker,
            args=(self.q_tasks, self.q_returns, self.q_errors, self.code_interface)
        )
        self.p.start()
        self.running = True
        self.ongoing_request = False

        def terminate_process():
            self.terminate()

        atexit.register(terminate_process)

    #
    #   GenericInterface functions
    def read_file(self, file_path: str, file_label: str):
        """Forwards to the worker a file path to read and its associated label

        Parameters
        ----------
        file_path : str
            File to read
        file_label : str
            File label
        """
        if isinstance(file_path, str) or isinstance(file_path, Path):
            print(f"Reading file {file_path} as {file_label}")
        else:
            print(f"Reading object of type {type(file_path)} as {file_label}")

        file_path = self.code_interface.serialize(file_path, file_label)

        unpicklables = dill.detect.baditems(file_path)

        if len(unpicklables) > 0:
            self.running = False
            raise TypeError(f"Found unpicklable item to send to the interface : {unpicklables[0]}.\nPlease redefine the {self.code_interface.__name__} serialize function to handle this error.")

        self.file_read.append((file_path, file_label))

        return self.__get_function((SlaveCommand.READ_FILE, [file_path, file_label]))

    def __get_function(self, argument: Tuple[SlaveCommand, Any]):
        """Sends a function call to the process, and forward its return.

        Parameters
        ----------
        argument : Tuple[SlaveCommand, Any]
            Command and arguments to send to the slave
        """
        with self._lock:
            self.ongoing_request = True
            self.q_tasks.put(argument)
            value = self.get_result_or_error()
            self.ongoing_request = False
            return value

    def get_labels(
        self,
    ) -> List[str]:
        """Get from the interface the list of displayable labels (fields list)

        Returns
        -------
        List[str]
            List of labels
        """
        return self.__get_function([SlaveCommand.GET_LABELS, None])

    def get_label_coloring_mode(self, field_name: str) -> VisualizationMode:
        """Returns the coloring mode of the plot

        Parameters
        ----------
        field_name : str
            Name of the displayed field

        Returns
        -------
        VisualizationMode
            Coloring mode
        """
        return self.__get_function([SlaveCommand.GET_LABEL_COLORING_MODE, field_name])

    def get_file_input_list(
        self,
    ) -> List[Tuple[str, str]]:
        """Get from the interface the list of files labels and their description

        Returns
        -------
        List[Tuple[str, str]]
            List of (file label, description)
        """
        return self.__get_function([SlaveCommand.GET_FILE_INPUT_LIST, None])

    #   Geometry2D functions
    def compute_2D_data(
        self,
        u: Tuple[float, float, float],
        v: Tuple[float, float, float],
        origin: Tuple[float, float, float],
        size_u: float,
        size_v: float,
        q_tasks: mp.Queue,
        coloring_label: str,
        options: Dict[str, Any],
        caller: str = "API",
    ) -> Tuple[
        Data2D, bool
    ]:
        """Get the geometry from the interface

        Parameters
        ----------
        u : Tuple[float, float, float]
            Horizontal coordinate director vector
        v : Tuple[float, float, float]
            Vertical coordinate director vector
        origin : Tuple[float, float, float]
            Physical 3D position of the slice center
        size_u : float
            Size of the slice along the u axis
        size_v : float
            Size of the slice along the v axis
        q_tasks : mp.Queue
            Queue from which get orders from the master.
        coloring_label : str
            Field label to display
        options : Dict[str, Any]
            Additional options for frame computation.
        caller : str
            Identifier of the caller requesting the computation (default: "API")

        Returns
        -------
        Tuple[Data2D, bool]
            Data2D object containing the geometry, whether the polygons were updated
        """
        return self.__get_function(
            [
                SlaveCommand.COMPUTE_2D_DATA,
                [
                    u,
                    v,
                    origin,
                    size_u,
                    size_v,
                    q_tasks,
                    coloring_label,
                    options,
                    caller,
                ],
            ]
        )

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
        """
        return self.__get_function([SlaveCommand.GET_VALUE_DICT, [value_label, cells, options, caller]])


    def get_geometry_type(self,) -> GeometryType:
        """Returns the interface geometry type

        Returns
        -------
        GeometryType
            Interface geometry type
        """
        return self.__get_function([SlaveCommand.GET_GEOMETRY_TYPE, []])

    #   Geometry3D functions
    def compute_3D_data(
        self,
        coloring_label: str,
        options: Dict[str, Any]
    ) -> Tuple[Data3D, bool]:
        """Returns a list of polygons that defines the geometry in a given frame

        Parameters
        ----------
        options : Dict[str, Any]
            Additional options for frame computation.

        Returns
        -------
        Data3D
            Geometry to display
        bool
            Were the polygons updated compared to the past call
        """
        return self.__get_function(
            [
                SlaveCommand.COMPUTE_3D_DATA,
                [
                    coloring_label,
                    options,
                ],
            ]
        )

    def get_3d_value_dict(
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
        """
        return self.__get_function(
            [
                SlaveCommand.GET_3D_VALUE_DICT,
                [
                    value_label,
                    cells,
                    options,
                    caller,
                ],
            ]
        )

    #   ValueAtLocation functions
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
        return self.__get_function(
            [
                SlaveCommand.GET_VALUE,
                [
                    position,
                    cell_index,
                    material_name,
                    field,
                    options,
                ],
            ]
        )

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
        return self.__get_function(
            [
                SlaveCommand.GET_VALUES,
                [
                    positions,
                    cell_indexes,
                    material_names,
                    field,
                    options,
                ],
            ]
        )

    #
    #   Value1DAtLocation functions
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
        return self.__get_function(
            [
                SlaveCommand.GET_1D_VALUE,
                [
                    position,
                    cell_index,
                    material_name,
                    field,
                    options,
                ],
            ]
        )

    #   OverLine functions
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
        """
        return self.__get_function(
            [
                SlaveCommand.COMPUTE_1D_LINE_DATA,
                [
                    pos,
                    u,
                    d,
                    q_tasks,
                    options,
                ],
            ]
        )

    def set_time(self, time_: float):
        """Set the current time in an interface to associate to the received value.

        Parameters
        ----------
        time_ : float
            Current time
        """
        return self.__get_function([SlaveCommand.SET_TIME, [time_]])

    #   CouplingInterface functions
    def update_data(self, key: str, data: Any):
        """Replaces the interface data by the current value

        Parameters
        ----------
        key : str
            Data associated key
        data : Any
            New value
        """
        return self.__get_function([SlaveCommand.UPDATE_DATA, [key, data]])

    def append_data(self, key: str, data: Any):
        """Stores the data and associates it to the current time.

        Parameters
        ----------
        key : str
            Data associated key
        data : Any
            New value
        """
        return self.__get_function([SlaveCommand.APPEND_DATA, [key, data]])

    def update_mesh(self, key: str, data: Any):
        """Replaces the interface data and mesh by the current value

        Parameters
        ----------
        key : str
            Data associated key
        data : Any
            New value
        """
        return self.__get_function([SlaveCommand.UPDATE_MESH, [key, data]])

    def append_mesh(self, key: str, data: Any):
        """Stores the data and mesh and associate them to the current time.

        Parameters
        ----------
        key : str
            Data associated key
        data : Any
            New value
        """
        return self.__get_function([SlaveCommand.APPEND_MESH, [key, data]])

    def get_template(self, name: str):
        """Returns the template for the C3PO getOutputxxxFieldTemplate functions

        Parameters
        ----------
        name : str
            Field name
        """
        return self.__get_function([SlaveCommand.GET_TEMPLATE, name])

    def set_template(self, name: str, template: Any):
        """Sets the template returned by C3PO getOutputxxxFieldTemplate functions

        Parameters
        ----------
        name : str
            Field name
        template : Any
            Object to set as template
        """
        return self.__get_function([SlaveCommand.SET_TEMPLATE, [name, template]])

    @property
    def update_policy(self):
        """Returns the update_policy attribute from the interface

        Returns
        -------
        Any
            The update_policy value
        """
        return self.__get_function([SlaveCommand.GET_UPDATE_POLICY, None])

    @update_policy.setter
    def update_policy(self, value):
        """Sets the update_policy attribute on the interface

        Parameters
        ----------
        value : Any
            The update_policy value to set
        """
        return self.__get_function([SlaveCommand.SET_UPDATE_POLICY, value])

    def duplicate(
        self,
    ) -> "ComputeSlave":
        """Returns a duplicate of the current ComputeSlave. The copy is reseted, and reads the file history.

        Returns
        -------
        ComputeSlave
            ComputeSlave copy.
        """
        duplicata = ComputeSlave(self.code_interface)

        duplicata.reset()

        for f in self.file_read:
            duplicata.read_file(f[0], f[1])

        return duplicata

    def terminate(
        self,
    ):
        """Terminates the subprocess"""
        self.running = False
        if self.p is not None and self.p.is_alive():
            self.p.terminate()
            self.p.join(timeout=5)

    def get_result_or_error(self):
        """Gets the return value from the process. If an error was sent, raise the error instead."""
        while self.p.is_alive():
            try:
                # Try to get a result with a short timeout
                return self.q_returns.get(block=True, timeout=0.01)
            except queue.Empty:
                # No result yet, check for errors (non-blocking)
                try:
                    error = self.q_errors.get(block=False)
                    if self.allow_errors:
                        pn.state.notifications.error(f"Error {error}, restoring data.")
                    else:
                        raise error
                    return None
                except queue.Empty:
                    # No result or error, but process is still alive
                    continue

        # Process is no longer alive
        self.ongoing_request = False
        if not self.q_errors.empty():
            error = self.q_errors.get()
            if self.allow_errors:
                pn.state.notifications.error(f"Error {error}, restoring data.")
            else:
                raise error
            return None
        elif not self.q_returns.empty():
            return self.q_returns.get()
        else:
            return None


    def call_custom_function(self, function_name: str, arguments: Dict[str, Any]):
        """Call an custom function meant to ease extension developments

        Parameters
        ----------
        function_name : str
            Name of the interface function
        arguments : Dict[str, Any]
            Kwargs of the custom function

        Returns
        -------
        Any
            Function return
        """
        return self.__get_function([SlaveCommand.CUSTOM, [function_name, arguments]])

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
        return self.__get_function([SlaveCommand.SAVE, [file_path, include_files]])

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
        return self.__get_function([SlaveCommand.LOAD, [file_path, include_files]])


if __name__ == "__main__":
    pass
