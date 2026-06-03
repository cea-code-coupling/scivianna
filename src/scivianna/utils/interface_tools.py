
from enum import Enum
from typing import Dict, Type, Union

from scivianna.interface.generic_interface import GenericInterface

class GenericInterfaceEnum(Enum):
    MED="Medcoupling"

def load_available_interfaces()->Dict[str, Type[GenericInterface]]:
    """Loads the built-in available interfaces

    Returns
    -------
    Dict[str, Type[GenericInterface]]
        Available interfaces linked to their name
    """
    
    available_code_interfaces:Dict[str, Type[GenericInterface]] = {}

    try:
        from scivianna.interface.med_interface import MEDInterface
        available_code_interfaces[GenericInterfaceEnum.MED] = MEDInterface
    except ImportError:
        pass

    # We except nothing as the interface is not implemented yet, so we will get in the except eitherway
    except:
        pass

    return available_code_interfaces
    
def get_interface_default_panel(interface:Union[GenericInterfaceEnum, str], title:str = ""):
    """Returns the default panel of built-in implemented code interfaces. Returns None if interface is unknown.

    Parameters
    ----------
    interface : GenericInterfaceEnum
        GenericInterfaceEnum representing the code interface

    Returns
    -------
    VisualizationPanel
        Panel of the code interface, or None
    """
    if interface == GenericInterfaceEnum.MED:
        from scivianna.notebook_tools import get_med_panel
        return get_med_panel(None, title=title)
    return None