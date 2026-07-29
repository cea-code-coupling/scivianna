"""
Scivianna: SCIentific VIsualizer for simulAtioN aNAlysis

Scivianna is an open-source Python simulation geometry and result visualizer
based on Holoviz Panel. It provides a powerful and flexible environment for
visualizing 1D plots, 2D geometries, 3D visualizations, and DataFrames simultaneously,
with support for real-time simulation result visualization through code coupling.

Key Features
------------
- Simultaneous 1D, 2D, 3D, and DataFrame visualization
- Real-time code coupling via C3PO/ICOCO interface
- Generic data interfaces for MED, VTK, CSV, and custom formats
- Remote server access for HPC cluster visualization
- Extensible architecture with plugin extensions

Quick Start
-----------
>>> from scivianna import ComputeSlave, Panel2D
>>> from scivianna.interface import MEDInterface
>>> 
>>> # Create a slave with your data interface
>>> slave = ComputeSlave(MEDInterface)
>>> slave.read_file("path/to/file.med", "geometry")
>>> 
>>> # Create and display a visualization panel
>>> panel = Panel2D(slave, name="My Panel")
>>> panel.show()

For more information, see the documentation at:
https://github.com/cea-code-coupling/scivianna

Package Version
---------------
The package version is read from the VERSION file and stored in __version__.
"""

from pathlib import Path
from typing import List

# Version information
try:
    with open(Path(__file__).parent / "VERSION") as f:
        __version__: str = f.read().strip()
except FileNotFoundError:
    __version__ = "0.0.0.dev0"

from scivianna.constants import (
    CSV,
    DEFAULT_COLORMAP,
    DEFAULT_ORIGIN,
    DEFAULT_SIZE,
    GEOMETRY,
    MATERIAL,
    MESH,
    X,
    Y,
    Z,
)
from scivianna.enums import (
    DataType,
    GeometryType,
    UpdateEvent,
    UpdatePolicy,
    VisualizationMode,
)

# Import core components for convenient access
from scivianna.slave import ComputeSlave

# Public API - defines what is exported when using 'from scivianna import *'
__all__: List[str] = [
    # Core components
    "ComputeSlave",
    # Panels
    "Panel2D",
    "Panel3D",
    "Panel1D",
    "PanelDataFrame",
    # Interfaces
    "GenericInterface",
    "Geometry2D",
    "Geometry2DPolygon",
    "Geometry2DGrid",
    "Geometry3D",
    "ValueAtLocation",
    "Value1DAtLocation",
    "DataFrameInterface",
    "CouplingInterface",
    # Data containers
    "Data2D",
    "Data3D",
    "Data1D",
    "DataContainer",
    # Enums
    "VisualizationMode",
    "PlotType",
    "GeometryType",
    "DataType",
    "UpdateEvent",
    "UpdatePolicy",
    # Constants
    "MESH",
    "MATERIAL",
    "GEOMETRY",
    "CSV",
    "X",
    "Y",
    "Z",
    "DEFAULT_ORIGIN",
    "DEFAULT_SIZE",
    "DEFAULT_COLORMAP",
    # Extensions
    "Extension",
    # Layout
    "GridStackLayout",
    "SplitLayout",
]

# Import panels (after __all__ definition to avoid circular imports)
try:
    from scivianna.panel.panel_1d import Panel1D
    from scivianna.panel.panel_2d import Panel2D
    from scivianna.panel.panel_3d import Panel3D
    from scivianna.panel.panel_dataframe import PanelDataFrame
except ImportError:
    # Panels may have additional dependencies that are not installed
    pass

# Import interfaces
try:
    from scivianna.interface.generic_interface import (
        CouplingInterface,
        DataFrameInterface,
        GenericInterface,
        Geometry2D,
        Geometry2DGrid,
        Geometry2DPolygon,
        Geometry3D,
        Value1DAtLocation,
        ValueAtLocation,
    )
except ImportError:
    pass

# Import data containers
try:
    from scivianna.data.data1d import Data1D
    from scivianna.data.data2d import Data2D
    from scivianna.data.data3d import Data3D
    from scivianna.data.data_container import DataContainer
except ImportError:
    pass

# Import extensions
try:
    from scivianna.extension.extension import Extension
except ImportError:
    pass

# Import layout
try:
    from scivianna.layout.gridstack import GridStackLayout
    from scivianna.layout.split import SplitLayout
except ImportError:
    pass


def get_version() -> str:
    """
    Get the Scivianna version string.

    Returns
    -------
    str
        Version string (e.g., "1.0.0")
    """
    return __version__


def get_info() -> dict:
    """
    Get information about the Scivianna installation.

    Returns
    -------
    dict
        Dictionary containing version and available components
    """
    import sys

    info = {
        "version": __version__,
        "python_version": sys.version,
        "available_components": {
            "panels": {},
            "interfaces": {},
            "plotters": {},
        },
    }

    # Check available panels
    try:
        from scivianna.panel.panel_2d import Panel2D

        info["available_components"]["panels"]["Panel2D"] = True
    except ImportError:
        info["available_components"]["panels"]["Panel2D"] = False

    try:
        from scivianna.panel.panel_3d import Panel3D

        info["available_components"]["panels"]["Panel3D"] = True
    except ImportError:
        info["available_components"]["panels"]["Panel3D"] = False

    # Check available interfaces
    try:
        from scivianna.interface.med_interface import MEDInterface

        info["available_components"]["interfaces"]["MED"] = True
    except ImportError:
        info["available_components"]["interfaces"]["MED"] = False

    try:
        from scivianna.interface.vtk_interface import VTKInterface

        info["available_components"]["interfaces"]["VTK"] = True
    except ImportError:
        info["available_components"]["interfaces"]["VTK"] = False

    return info
