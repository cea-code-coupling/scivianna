"""Interface module for scivianna.

This module provides a registry of all available interfaces.
"""

from typing import Dict, Type, Union
from scivianna.interface.generic_interface import GenericInterface
from scivianna.utils.interface_tools import GenericInterfaceEnum
from scivianna.interface.med_interface import MEDInterface
from scivianna.interface.vtk_interface import VTKInterface
from scivianna.interface.csv_result import CSVInterface
from scivianna.interface.structured_mesh_interface import StructuredMeshInterface
from scivianna.interface.time_dataframe import TimeDataFrame


# Default dictionary of built-in interfaces
# Users can add their own interfaces to this dictionary
INTERFACES: Dict[Union[str, GenericInterfaceEnum], Type[GenericInterface]] = {
    "MED": MEDInterface,
    "VTK": VTKInterface,
    "CSV": CSVInterface,
    "StructuredMesh": StructuredMeshInterface,
    "TimeDataFrame": TimeDataFrame,
}


def register_interface(
    key: Union[str, GenericInterfaceEnum], 
    interface_class: Type[GenericInterface]
) -> None:
    """Register a new interface class.
    
    Parameters
    ----------
    key : Union[str, GenericInterfaceEnum]
        Key to identify the interface
    interface_class : Type[GenericInterface]
        Interface class to register
    
    Raises
    ------
    TypeError
        If interface_class does not inherit from GenericInterface
    """
    if not issubclass(interface_class, GenericInterface):
        raise TypeError(
            f"Interface class must inherit from GenericInterface, got {interface_class}"
        )
    INTERFACES[key] = interface_class


def get_interface(key: Union[str, GenericInterfaceEnum]) -> Type[GenericInterface]:
    """Get an interface class by its key.
    
    Parameters
    ----------
    key : Union[str, GenericInterfaceEnum]
        Key identifying the interface
    
    Returns
    -------
    Type[GenericInterface]
        Interface class
    
    Raises
    ------
    KeyError
        If the interface key is not found
    """
    return INTERFACES[key]


def get_all_interfaces() -> Dict[Union[str, GenericInterfaceEnum], Type[GenericInterface]]:
    """Get all registered interfaces.
    
    Returns
    -------
    Dict[Union[str, GenericInterfaceEnum], Type[GenericInterface]]
        Dictionary of all registered interfaces
    """
    return INTERFACES.copy()