"""
Serialization utilities for scivianna components.

This module provides serialization and deserialization functions at different levels:
- slave_only: Serialize/deserialize only the ComputeSlave
- panel2d_only: Serialize/deserialize only Panel2D (without slave data)
- panel1d_only: Serialize/deserialize only Panel1D (without slave data)
- layout: Serialize/deserialize complete SplitLayout with all components

Note: To avoid circular imports, actual module imports are done lazily inside functions.
Panel3D has an optional dependency (pyvista), so it is imported lazily with a stub fallback.
"""

from __future__ import annotations

import pickle
import json
import zipfile
import tempfile
import os
from pathlib import Path
from typing import Dict, Type, Union, TYPE_CHECKING, Optional, Any

if TYPE_CHECKING:
    from scivianna.panel.panel_2d import Panel2D
    from scivianna.panel.panel_1d import Panel1D
    from scivianna.panel.panel_dataframe import PanelDataFrame
    from scivianna.panel.visualisation_panel import VisualizationPanel
    from scivianna.layout.split import SplitLayout, SplitItem
    from scivianna.layout.gridstack import GridStackLayout

from scivianna.interface.generic_interface import GenericInterface
from scivianna.interface import INTERFACES, register_interface
from scivianna.slave import ComputeSlave
from scivianna.enums import UpdateEvent

# =============================================================================
# OPTIONAL PANEL3D IMPORT (pyvista dependency)
# =============================================================================

try:
    from scivianna.panel.panel_3d import Panel3D
except (ImportError, ModuleNotFoundError):
    class Panel3D:
        """Stub class for Panel3D when pyvista is not available."""
        @classmethod
        def from_json(*args, **kwargs):
            raise NotImplementedError("Panel3D could not be imported, please install scivianna")


# =============================================================================
# UTILITY FUNCTIONS FOR EXTENSIONS AND DATA
# =============================================================================

def _get_extension_class(ext_name: str, interface: Type[GenericInterface], panel_type: str) -> Optional[Type]:
    """
    Helper method to find an extension class by name.
    
    Searches in the following order:
    1. interface.extensions
    2. Panel's default_extensions
    
    Parameters
    ----------
    ext_name : str
        Name of the extension class to find
    interface : Type[GenericInterface]
        Interface class to search in first
    panel_type : str
        Type of panel ("Panel1D", "Panel2D", "Panel3D", or "PanelDataFrame")
        
    Returns
    -------
    Type
        Extension class if found, None otherwise
    """
    # First, try to find in interface.extensions
    for iface_ext in interface.extensions:
        if iface_ext.__name__ == ext_name:
            return iface_ext
    
    # Then try panel's default_extensions
    if panel_type == "Panel1D":
        from scivianna.panel.panel_1d import default_extensions as panel1d_default_extensions
        for ext in panel1d_default_extensions:
            if ext.__name__ == ext_name:
                return ext
    elif panel_type == "Panel2D":
        from scivianna.panel.panel_2d import default_extensions as panel2d_default_extensions
        for ext in panel2d_default_extensions:
            if ext.__name__ == ext_name:
                return ext
    elif panel_type == "Panel3D":
        from scivianna.panel.panel_3d import default_extensions as panel3d_default_extensions
        for ext in panel3d_default_extensions:
            if ext.__name__ == ext_name:
                return ext
    elif panel_type == "PanelDataFrame":
        from scivianna.panel.panel_dataframe import default_extensions as paneldf_default_extensions
        for ext in paneldf_default_extensions:
            if ext.__name__ == ext_name:
                return ext
    
    return None

def _save_current_data(current_data: Any, temp_path: Path, panel_name: str) -> str:
    """
    Save current_data (Data2D or Data3D) to a pickle file.
    
    Parameters
    ----------
    current_data : Any
        The data to save (typically Data2D or Data3D)
    temp_path : Path
        Temporary directory path
    panel_name : str
        Name of the panel (used for filename)
        
    Returns
    -------
    str
        Relative path to the saved file
    """
    data_path = temp_path / "data" / f"{panel_name}.pkl"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_path, "wb") as f:
        pickle.dump(current_data, f)
    return str(data_path.relative_to(temp_path))


def _load_current_data(temp_path: Path, relative_path: str) -> Optional[Any]:
    """
    Load current_data (Data2D or Data3D) from a pickle file.
    
    Parameters
    ----------
    temp_path : Path
        Temporary directory path containing extracted zip contents
    relative_path : str
        Relative path to the data file
        
    Returns
    -------
    Optional[Any]
        Loaded data or None if file doesn't exist
    """
    data_path = temp_path / relative_path
    if data_path.exists():
        with open(data_path, "rb") as f:
            return pickle.load(f)
    return None


def _build_panel_metadata(
    panel_type: str,
    panel: Union["Panel1D", "Panel2D", "Panel3D"],
    interface_key: str
) -> Dict:
    """
    Build panel metadata dictionary for JSON serialization.
    
    Parameters
    ----------
    panel_type : str
        Type of panel ("Panel1D", "Panel2D", "Panel3D", or "PanelDataFrame")
    panel : Union[Panel1D, Panel2D, Panel3D]
        Panel instance to serialize
    interface_key : str
        Interface key string
        
    Returns
    -------
    Dict
        Metadata dictionary
    """
    base_metadata = {
        "panel_type": panel_type,
        "panel_json": panel.to_json(),
        "extensions": [e.__class__.__name__ for e in panel.extensions],
        "extensions_data": {e.__class__.__name__: e.to_json() for e in panel.extensions},
        "interface_key": interface_key,
        "slave_file": "slave.pkl"
    }
    
    if panel_type in ("Panel2D", "Panel3D"):
        base_metadata["current_data_file"] = "current_data.pkl"
    
    return base_metadata


def _restore_extensions(
    extensions_data: Dict[str, Dict],
    saved_extensions: list,
    interface: Type[GenericInterface],
    panel_type: str
) -> list:
    """
    Restore extension classes with their saved state.
    
    Parameters
    ----------
    extensions_data : Dict[str, Dict]
        Dictionary mapping extension names to their state dicts
    saved_extensions : list
        List of extension names to restore
    interface : Type[GenericInterface]
        Interface class to search for extensions
    panel_type : str
        Type of panel ("Panel1D", "Panel2D", "Panel3D", or "PanelDataFrame")
        
    Returns
    -------
    list
        List of (extension_class, state_dict) tuples
    """
    extensions = []
    for ext_name in saved_extensions:
        ext_state = extensions_data.get(ext_name, {})
        ext_class = _get_extension_class(ext_name, interface, panel_type)
        
        if ext_class is not None:
            extensions.append((ext_class, ext_state))
    
    return extensions


# =============================================================================
# SLAVE ONLY SERIALIZATION
# =============================================================================

def save_slave_to_file(
    slave: ComputeSlave,
    file_path: Union[str, Path],
    include_files: bool = True
) -> Path:
    """
    Save a ComputeSlave state to a pickle file.
    
    Parameters
    ----------
    slave : ComputeSlave
        The slave to save
    file_path : Union[str, Path]
        Path to the output file
    include_files : bool = True
        If True, includes loaded files in the serialization
        
    Returns
    -------
    Path
        Path to the saved file
    """
    file_path = Path(file_path)
    slave.save(file_path, include_files=include_files)
    return file_path


def load_slave_from_file(
    file_path: Union[str, Path],
    interface_class: Type[GenericInterface],
    include_files: bool = True
) -> ComputeSlave:
    """
    Load a ComputeSlave state from a pickle file.
    
    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the input file
    interface_class : Type[GenericInterface]
        The interface class for the slave
    include_files : bool = True
        If True, includes loaded files in the deserialization
        
    Returns
    -------
    ComputeSlave
        The loaded slave
    """
    file_path = Path(file_path)
    slave = ComputeSlave(interface_class)
    slave.load(file_path, include_files=include_files)
    return slave


# =============================================================================
# PANEL2D ONLY SERIALIZATION
# =============================================================================

def save_panel2d_to_file(
    panel: "Panel2D",
    file_path: Union[str, Path],
    include_files: bool = True
) -> Path:
    """
    Save a Panel2D to a zip file containing panel configuration and slave data.
    
    The zip file contains:
    - panel.json: JSON file describing the panel configuration and interface info
    - slave.pkl: Serialized slave data
    - current_data.pkl: Serialized current_data for Panel2D
    
    Parameters
    ----------
    panel : Panel2D
        The panel to save
    file_path : Union[str, Path]
        Path to the output zip file
    include_files : bool = True
        If True, includes loaded files in the slave serialization
        
    Returns
    -------
    Path
        Path to the saved zip file
    """
    from scivianna.interface import INTERFACES
    
    file_path = Path(file_path)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Get interface info
        slave = panel.get_slave()
        interface_class = slave.code_interface
        
        # Find the key for this interface in INTERFACES dict
        interface_key = None
        for key, iface_class in INTERFACES.items():
            if iface_class == interface_class:
                interface_key = key if isinstance(key, str) else key.value
                break
        
        # If not found in built-in interfaces, use class name as fallback
        if interface_key is None:
            interface_key = interface_class.__name__
        
        # Save panel configuration as JSON using the utility function
        panel_data = _build_panel_metadata("Panel2D", panel, interface_key)
        
        panel_json_path = temp_path / "panel.json"
        with open(panel_json_path, "w") as f:
            json.dump(panel_data, f, indent=2)
        
        # Save slave data using the slave serialization function
        slave_data_path = temp_path / "slave.pkl"
        save_slave_to_file(slave, slave_data_path, include_files=include_files)
        
        # Also save current_data
        current_data_path = temp_path / "current_data.pkl"
        with open(current_data_path, "wb") as f:
            pickle.dump(panel.current_data, f)
        
        # Create the zip file
        zip_path = file_path.with_suffix(".zip") if not file_path.suffix == ".zip" else file_path
        
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(temp_path):
                for file in files:
                    file_path_inside = Path(root) / file
                    arcname = file_path_inside.relative_to(temp_path)
                    zf.write(file_path_inside, arcname)
    
    return zip_path


def load_panel2d_from_file(
    file_path: Union[str, Path],
    include_files: bool = True
) -> "Panel2D":
    """
    Load a Panel2D from a zip file. The slave is automatically recreated from the saved data.
    
    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the input zip file
    include_files : bool = True
        If True, includes loaded files in the slave deserialization
        
    Returns
    -------
    Panel2D
        The loaded panel with its associated slave
    """
    from scivianna.panel.panel_2d import Panel2D
    
    file_path = Path(file_path)
    zip_path = file_path.with_suffix(".zip") if not file_path.suffix == ".zip" else file_path
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(temp_path)
        
        # Read panel.json
        panel_json_path = temp_path / "panel.json"
        with open(panel_json_path, "r") as f:
            panel_data = json.load(f)
        
        # Get the interface class from the INTERFACES dictionary using the interface_key
        interface_key = panel_data.get("interface_key")
        interface = INTERFACES.get(interface_key)
        
        if interface is None:
            raise KeyError(f"Interface '{interface_key}' not found in INTERFACES dictionary")
        
        # Create slave and load data using the slave serialization function
        slave = load_slave_from_file(
            temp_path / panel_data.get("slave_file", "slave.pkl"),
            interface,
            include_files=include_files
        )
        
        # Load current_data
        current_data = None
        current_data_path = temp_path / panel_data.get("current_data_file", "current_data.pkl")
        if current_data_path.exists():
            with open(current_data_path, "rb") as f:
                current_data = pickle.load(f)
        
        # Rebuild extensions with their state using the utility function
        extensions_data = panel_data.get("extensions_data", {})
        saved_extensions = panel_data.get("extensions", [])
        extensions = _restore_extensions(extensions_data, saved_extensions, interface, "Panel2D")

        # Restore panel from json with the loaded data
        panel = Panel2D.from_json(panel_data["panel_json"], slave, current_data, extensions)
        
        return panel

# =============================================================================
# PANEL1D ONLY SERIALIZATION
# =============================================================================

def save_panel1d_to_file(
    panel: "Panel1D",
    file_path: Union[str, Path],
    include_files: bool = True
) -> Path:
    """
    Save a Panel1D to a zip file containing panel configuration and slave data.
    
    The zip file contains:
    - panel.json: JSON file describing the panel configuration and interface info
    - slave.pkl: Serialized slave data
    
    Parameters
    ----------
    panel : Panel1D
        The panel to save
    file_path : Union[str, Path]
        Path to the output zip file
    include_files : bool = True
        If True, includes loaded files in the slave serialization
        
    Returns
    -------
    Path
        Path to the saved zip file
    """
    from scivianna.interface import INTERFACES
    
    file_path = Path(file_path)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Get interface info
        slave = panel.get_slave()
        interface_class = slave.code_interface
        
        # Find the key for this interface in INTERFACES dict
        interface_key = None
        for key, iface_class in INTERFACES.items():
            if iface_class == interface_class:
                interface_key = key if isinstance(key, str) else key.value
                break
        
        # If not found in built-in interfaces, use class name as fallback
        if interface_key is None:
            interface_key = interface_class.__name__
        
        # Save panel configuration as JSON using the utility function
        panel_data = _build_panel_metadata("Panel1D", panel, interface_key)
        
        panel_json_path = temp_path / "panel.json"
        with open(panel_json_path, "w") as f:
            json.dump(panel_data, f, indent=2)
        
        # Save slave data using the slave serialization function
        slave_data_path = temp_path / "slave.pkl"
        save_slave_to_file(slave, slave_data_path, include_files=include_files)
        
        # Create the zip file
        zip_path = file_path.with_suffix(".zip") if not file_path.suffix == ".zip" else file_path
        
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(temp_path):
                for file in files:
                    file_path_inside = Path(root) / file
                    arcname = file_path_inside.relative_to(temp_path)
                    zf.write(file_path_inside, arcname)
    
    return zip_path


def load_panel1d_from_file(
    file_path: Union[str, Path],
    include_files: bool = True
) -> "Panel1D":
    """
    Load a Panel1D from a zip file. The slave is automatically recreated from the saved data.
    
    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the input zip file
    include_files : bool = True
        If True, includes loaded files in the slave deserialization
        
    Returns
    -------
    Panel1D
        The loaded panel with its associated slave
    """
    from scivianna.panel.panel_1d import Panel1D
    
    file_path = Path(file_path)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        with zipfile.ZipFile(file_path, "r") as zf:
            zf.extractall(temp_path)
        
        # Read panel.json
        panel_json_path = temp_path / "panel.json"
        with open(panel_json_path, "r") as f:
            panel_data = json.load(f)
        
        # Get the interface class from the INTERFACES dictionary using the interface_key
        interface_key = panel_data.get("interface_key")
        interface = INTERFACES.get(interface_key)
        
        if interface is None:
            raise KeyError(f"Interface '{interface_key}' not found in INTERFACES dictionary")
        
        # Create slave and load data using the slave serialization function
        slave = load_slave_from_file(
            temp_path / panel_data.get("slave_file", "slave.pkl"),
            interface,
            include_files=include_files
        )
        
        # Rebuild extensions with their state using the utility function
        extensions_data = panel_data.get("extensions_data", {})
        saved_extensions = panel_data.get("extensions", [])
        extensions = _restore_extensions(extensions_data, saved_extensions, interface, "Panel1D")
        
        # Restore panel from json
        panel = Panel1D.from_json(panel_data["panel_json"], slave, extensions)
        
        return panel


# =============================================================================
# PANEL3D ONLY SERIALIZATION
# =============================================================================

def save_panel3d_to_file(
    panel: "Panel3D",
    file_path: Union[str, Path],
    include_files: bool = True
) -> Path:
    """
    Save a Panel3D to a zip file containing panel configuration and slave data.
    
    The zip file contains:
    - panel.json: JSON file describing the panel configuration and interface info
    - slave.pkl: Serialized slave data
    - current_data.pkl: Serialized current_data (Data3D) for Panel3D
    
    Parameters
    ----------
    panel : Panel3D
        The panel to save
    file_path : Union[str, Path]
        Path to the output zip file
    include_files : bool = True
        If True, includes loaded files in the slave serialization
        
    Returns
    -------
    Path
        Path to the saved zip file
    """
    from scivianna.interface import INTERFACES
    
    file_path = Path(file_path)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Get interface info
        slave = panel.get_slave()
        interface_class = slave.code_interface
        
        # Find the key for this interface in INTERFACES dict
        interface_key = None
        for key, iface_class in INTERFACES.items():
            if iface_class == interface_class:
                interface_key = key if isinstance(key, str) else key.value
                break
        
        # If not found in built-in interfaces, use class name as fallback
        if interface_key is None:
            interface_key = interface_class.__name__
        
        # Save panel configuration as JSON
        panel_data = _build_panel_metadata("Panel3D", panel, interface_key)
        
        panel_json_path = temp_path / "panel.json"
        with open(panel_json_path, "w") as f:
            json.dump(panel_data, f, indent=2)
        
        # Save slave data using the slave serialization function
        slave_data_path = temp_path / "slave.pkl"
        save_slave_to_file(slave, slave_data_path, include_files=include_files)
        
        # Also save current_data (Data3D)
        current_data_path = temp_path / "current_data.pkl"
        with open(current_data_path, "wb") as f:
            pickle.dump(panel.current_data, f)
        
        # Create the zip file
        zip_path = file_path.with_suffix(".zip") if not file_path.suffix == ".zip" else file_path
        
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(temp_path):
                for file in files:
                    file_path_inside = Path(root) / file
                    arcname = file_path_inside.relative_to(temp_path)
                    zf.write(file_path_inside, arcname)
    
    return zip_path


def load_panel3d_from_file(
    file_path: Union[str, Path],
    include_files: bool = True
) -> "Panel3D":
    """
    Load a Panel3D from a zip file. The slave is automatically recreated from the saved data.
    
    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the input zip file
    include_files : bool = True
        If True, includes loaded files in the slave deserialization
        
    Returns
    -------
    Panel3D
        The loaded panel with its associated slave
    """
    from scivianna.panel.panel_3d import Panel3D
    file_path = Path(file_path)
    zip_path = file_path.with_suffix(".zip") if not file_path.suffix == ".zip" else file_path
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(temp_path)
        
        # Read panel.json
        panel_json_path = temp_path / "panel.json"
        with open(panel_json_path, "r") as f:
            panel_data = json.load(f)
        
        # Get the interface class from the INTERFACES dictionary using the interface_key
        interface_key = panel_data.get("interface_key")
        interface = INTERFACES.get(interface_key)
        
        if interface is None:
            raise KeyError(f"Interface '{interface_key}' not found in INTERFACES dictionary")
        
        # Create slave and load data using the slave serialization function
        slave = load_slave_from_file(
            temp_path / panel_data.get("slave_file", "slave.pkl"),
            interface,
            include_files=include_files
        )
        
        # Load current_data (Data3D)
        current_data = None
        current_data_path = temp_path / panel_data.get("current_data_file", "current_data.pkl")
        if current_data_path.exists():
            with open(current_data_path, "rb") as f:
                current_data = pickle.load(f)
        
        # Rebuild extensions with their state using the utility function
        extensions_data = panel_data.get("extensions_data", {})
        saved_extensions = panel_data.get("extensions", [])
        extensions = _restore_extensions(extensions_data, saved_extensions, interface, "Panel3D")

        # Restore panel from json with the loaded data
        panel = Panel3D.from_json(panel_data["panel_json"], slave, current_data, extensions)
        
        return panel


# =============================================================================
# PANELDATAFRAME ONLY SERIALIZATION
# =============================================================================

def save_paneldatframe_to_file(
    panel: "PanelDataFrame",
    file_path: Union[str, Path],
    include_files: bool = True
) -> Path:
    """
    Save a PanelDataFrame to a zip file containing panel configuration and slave data.
    
    The zip file contains:
    - panel.json: JSON file describing the panel configuration and interface info
    - slave.pkl: Serialized slave data
    - dataframe.pkl: Serialized DataFrame for PanelDataFrame
    
    Parameters
    ----------
    panel : PanelDataFrame
        The panel to save
    file_path : Union[str, Path]
        Path to the output zip file
    include_files : bool = True
        If True, includes loaded files in the slave serialization
        
    Returns
    -------
    Path
        Path to the saved zip file
    """
    from scivianna.interface import INTERFACES
    
    file_path = Path(file_path)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Get interface info
        slave = panel.get_slave()
        interface_class = slave.code_interface
        
        # Find the key for this interface in INTERFACES dict
        interface_key = None
        for key, iface_class in INTERFACES.items():
            if iface_class == interface_class:
                interface_key = key if isinstance(key, str) else key.value
                break
        
        # If not found in built-in interfaces, use class name as fallback
        if interface_key is None:
            interface_key = interface_class.__name__
        
        # Save panel configuration as JSON
        panel_data = _build_panel_metadata("PanelDataFrame", panel, interface_key)
        
        panel_json_path = temp_path / "panel.json"
        with open(panel_json_path, "w") as f:
            json.dump(panel_data, f, indent=2)
        
        # Save slave data using the slave serialization function
        slave_data_path = temp_path / "slave.pkl"
        save_slave_to_file(slave, slave_data_path, include_files=include_files)
        
        # Also save dataframe
        dataframe_dict = None
        df = panel.plotter.get_data()
        if df is not None:
            dataframe_dict = df.to_dict(orient="list")
        
        dataframe_path = temp_path / "dataframe.pkl"
        with open(dataframe_path, "wb") as f:
            pickle.dump(dataframe_dict, f)
        
        # Create the zip file
        zip_path = file_path.with_suffix(".zip") if not file_path.suffix == ".zip" else file_path
        
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(temp_path):
                for file in files:
                    file_path_inside = Path(root) / file
                    arcname = file_path_inside.relative_to(temp_path)
                    zf.write(file_path_inside, arcname)
    
    return zip_path


def load_paneldatframe_from_file(
    file_path: Union[str, Path],
    include_files: bool = True
) -> "PanelDataFrame":
    """
    Load a PanelDataFrame from a zip file. The slave is automatically recreated from the saved data.
    
    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the input zip file
    include_files : bool = True
        If True, includes loaded files in the slave deserialization
        
    Returns
    -------
    PanelDataFrame
        The loaded panel with its associated slave
    """
    from scivianna.panel.panel_dataframe import PanelDataFrame
    import pandas as pd
    
    file_path = Path(file_path)
    zip_path = file_path.with_suffix(".zip") if not file_path.suffix == ".zip" else file_path
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(temp_path)
        
        # Read panel.json
        panel_json_path = temp_path / "panel.json"
        with open(panel_json_path, "r") as f:
            panel_data = json.load(f)
        
        # Get the interface class from the INTERFACES dictionary using the interface_key
        interface_key = panel_data.get("interface_key")
        interface = INTERFACES.get(interface_key)
        
        if interface is None:
            raise KeyError(f"Interface '{interface_key}' not found in INTERFACES dictionary")
        
        # Create slave and load data using the slave serialization function
        slave = load_slave_from_file(
            temp_path / panel_data.get("slave_file", "slave.pkl"),
            interface,
            include_files=include_files
        )
        
        # Load dataframe
        dataframe_dict = None
        dataframe_path = temp_path / "dataframe.pkl"
        if dataframe_path.exists():
            with open(dataframe_path, "rb") as f:
                dataframe_dict = pickle.load(f)
        
        # Rebuild extensions with their state using the utility function
        extensions_data = panel_data.get("extensions_data", {})
        saved_extensions = panel_data.get("extensions", [])
        extensions = _restore_extensions(extensions_data, saved_extensions, interface, "PanelDataFrame")

        # Restore panel from json with the loaded dataframe
        panel = PanelDataFrame.from_json(panel_data["panel_json"], slave, extensions)
        
        if dataframe_dict is not None:
            panel.plotter.update_data(pd.DataFrame(dataframe_dict))
        
        return panel


# =============================================================================
# LAYOUT SERIALIZATION (Full SplitLayout with all components)
# =============================================================================

def save_layout_to_zip(
    layout: "SplitLayout",
    file_path: Union[str, Path],
    include_files: bool = True
) -> Path:
    """
    Saves the SplitLayout configuration and slave data to a zip file.
    
    The zip file contains:
    - layout.json: JSON file describing the layout structure, current frame, 
      and for each slave: panel name, slave info, and associated interface name
    - data/: Folder containing serialized data for each slave
    
    Parameters
    ----------
    layout : SplitLayout
        The layout to save
    file_path : Union[str, Path]
        Path to the output zip file
    include_files : bool = True
        If True, includes loaded files in the slave serialization
        
    Returns
    -------
    Path
        Path to the saved zip file
    """
    from scivianna.panel.panel_2d import Panel2D
    from scivianna.panel.panel_dataframe import PanelDataFrame
    
    file_path = Path(file_path)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Build layout metadata
        layout_data = {
            "split_item": _serialize_split_item(layout.split_item),
            "current_frame": layout.current_frame,
            "time_widget": layout.time_widget.to_json() if hasattr(layout, 'time_widget') else None,
            "panels": {}
        }
        
        # Collect panel and slave information
        for panel_name, panel in layout.visualisation_panels.items():
            slave = panel.get_slave()
            interface_class = slave.code_interface
            
            # Find the key for this interface in INTERFACES dict
            interface_key = None
            for key, iface_class in INTERFACES.items():
                if iface_class == interface_class:
                    interface_key = key if isinstance(key, str) else key.value
                    break
            
            # If not found in built-in interfaces, use class name as fallback
            if interface_key is None:
                interface_key = interface_class.__name__
            
            layout_data["panels"][panel_name] = {
                "slave_name": slave.code_interface.__name__,
                "interface_key": interface_key,
                "sync_field": panel.sync_field,
                "update_event": panel.update_event,
                "data_file": f"slave_data/{panel_name}.pkl",
            }
            
            # Save slave data to individual pickle file using the slave serialization function
            slave_data_path = temp_path / "slave_data" / f"{panel_name}.pkl"
            slave_data_path.parent.mkdir(parents=True, exist_ok=True)
            save_slave_to_file(slave, slave_data_path, include_files=include_files)
            
            # Save current_data for Panel2D and Panel3D using the utility function
            if isinstance(panel, (Panel2D, Panel3D)):
                _save_current_data(panel.current_data, temp_path, panel_name)
                layout_data["panels"][panel_name]["current_data"] = f"data/{panel_name}.pkl"
            
            # Save dataframe for PanelDataFrame
            elif isinstance(panel, PanelDataFrame):
                df = panel.plotter.get_data()
                if df is not None:
                    dataframe_dict = df.to_dict(orient="list")
                    dataframe_path = temp_path / "dataframe" / f"{panel_name}.pkl"
                    dataframe_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(dataframe_path, "wb") as f:
                        pickle.dump(dataframe_dict, f)
                    layout_data["panels"][panel_name]["dataframe"] = f"dataframe/{panel_name}.pkl"
        
        # Write layout.json
        layout_json_path = temp_path / "layout.json"
        with open(layout_json_path, "w") as f:
            json.dump(layout_data, f, indent=2)
        
        # Create the zip file
        zip_path = file_path.with_suffix(".zip") if not file_path.suffix == ".zip" else file_path
        
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(temp_path):
                for file in files:
                    file_path_inside = Path(root) / file
                    arcname = file_path_inside.relative_to(temp_path)
                    zf.write(file_path_inside, arcname)
    
    return zip_path


def load_layout_from_zip(
    file_path: Union[str, Path],
    include_files: bool = True,
    additional_interfaces: Dict[Union[str, Any], Type[GenericInterface]] = {}
) -> "SplitLayout":
    """
    Restores a SplitLayout from a zip file.
    
    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the input zip file
    include_files : bool = True
        If True, includes loaded files in the slave deserialization
    additional_interfaces : Dict = {}
        Additional interfaces to register
        
    Returns
    -------
    SplitLayout
        Restored SplitLayout instance
    """
    from scivianna.layout.split import SplitLayout
    
    file_path = Path(file_path)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        with zipfile.ZipFile(file_path, "r") as zf:
            zf.extractall(temp_path)
        
        # Read layout.json
        layout_json_path = temp_path / "layout.json"
        with open(layout_json_path, "r") as f:
            layout_data = json.load(f)
        
        # Register additional interfaces first
        from scivianna.interface import INTERFACES as IFACE_DICT
        for key, iface_class in additional_interfaces.items():
            register_interface(key, iface_class)
        
        # Rebuild the split item structure with pre-loaded data
        split_item = _deserialize_split_item(layout_data["split_item"], layout_data, temp_path, include_files)
        
        # Create the layout instance
        layout = SplitLayout(
            split_item=split_item,
            additional_interfaces=additional_interfaces
        )
        
        # Restore time_widget if it was saved
        time_widget_data = layout_data.get("time_widget")
        if time_widget_data is not None:
            from scivianna.extension.coupling import CouplingExtension
            layout.time_widget = CouplingExtension(layout, None, None, None)
            CouplingExtension.from_json(layout.time_widget, time_widget_data)
            layout.gui.add_extension(layout.time_widget)

            for panel in layout.visualisation_panels.values():
                panel.panel_coupling_extension = layout.time_widget
        
        # Set the current frame
        layout.set_to_frame(layout_data["current_frame"])
        
        return layout


def _extract_panel_to_temp_zip(
    temp_path: Path,
    panel_name: str,
    full_data: Dict,
    panel_json_data: Dict = None,
    extensions: list = None,
    extensions_data: Dict = None
) -> Path:
    """
    Extract a panel's data from the layout structure into a temporary zip file
    matching the single-panel serialization structure.
    
    This enables reusing load_panelX_from_file functions for layout deserialization.
    
    Parameters
    ----------
    temp_path : Path
        Temporary directory containing extracted layout zip
    panel_name : str
        Name of the panel to extract
    full_data : Dict
        Full layout serialized data (layout.json contents)
    panel_json_data : Dict, optional
        Panel JSON data from the split_item tree. If None, reads from full_data.
    extensions : list, optional
        List of extension names. If None, reads from full_data panels dict.
    extensions_data : Dict, optional
        Extension state dict. If None, reads from full_data panels dict.
        
    Returns
    -------
    Path
        Path to the created temporary zip file
    """
    import shutil
    
    panel_info = full_data["panels"][panel_name]
    
    # Use provided panel_json_data or read from full_data (GridStackLayout format)
    if panel_json_data is None:
        panel_json_data = panel_info.get("panel_json", {})
    
    # Use provided extensions or read from full_data (SplitLayout format)
    if extensions is None:
        extensions = panel_info.get("extensions", [])
    if extensions_data is None:
        extensions_data = panel_info.get("extensions_data", {})
    
    # Create a temporary directory for the panel zip contents
    panel_temp_dir = temp_path / f"_panel_{panel_name}_temp"
    panel_temp_dir.mkdir(exist_ok=True)
    
    # Build panel.json in the single-panel format
    panel_json = {
        "panel_type": panel_info.get("panel_type", "Panel2D"),
        "panel_json": panel_json_data,
        "extensions": extensions,
        "extensions_data": extensions_data,
        "interface_key": panel_info["interface_key"],
        "slave_file": "slave.pkl",
    }
    
    # Add current_data_file for Panel2D/Panel3D
    if panel_info.get("current_data"):
        panel_json["current_data_file"] = "current_data.pkl"
    
    panel_json_path = panel_temp_dir / "panel.json"
    with open(panel_json_path, "w") as f:
        json.dump(panel_json, f, indent=2)
    
    # Copy slave data
    slave_src = temp_path / panel_info["data_file"]
    slave_dst = panel_temp_dir / "slave.pkl"
    if slave_src.exists():
        shutil.copy2(slave_src, slave_dst)
    
    # Copy current_data for Panel2D/Panel3D
    if panel_info.get("current_data"):
        current_data_src = temp_path / panel_info["current_data"]
        current_data_dst = panel_temp_dir / "current_data.pkl"
        if current_data_src.exists():
            shutil.copy2(current_data_src, current_data_dst)
    
    # Copy dataframe for PanelDataFrame
    if panel_info.get("dataframe"):
        dataframe_src = temp_path / panel_info["dataframe"]
        dataframe_dst = panel_temp_dir / "dataframe.pkl"
        if dataframe_src.exists():
            shutil.copy2(dataframe_src, dataframe_dst)
    
    # Create the zip file
    zip_path = temp_path / f"_panel_{panel_name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(panel_temp_dir):
            for file in files:
                file_path_inside = Path(root) / file
                arcname = file_path_inside.relative_to(panel_temp_dir)
                zf.write(file_path_inside, arcname)
    
    # Clean up the temporary directory
    shutil.rmtree(panel_temp_dir, ignore_errors=True)
    
    return zip_path


def _serialize_split_item(item: Union["SplitItem", "VisualizationPanel"]) -> Dict:
    """
    Recursively serializes a SplitItem to a dictionary.
    
    Parameters
    ----------
    item : Union[SplitItem, VisualizationPanel]
        Item to serialize
        
    Returns
    -------
    Dict
        Serialized representation of the item
    """
    from scivianna.layout.split import SplitItem
    from scivianna.panel.visualisation_panel import VisualizationPanel
    
    if isinstance(item, SplitItem):
        return {
            "type": "SplitItem",
            "direction": item.direction.name,
            "panel_1": _serialize_split_item(item.panel_1),
            "panel_2": _serialize_split_item(item.panel_2)
        }
    elif isinstance(item, VisualizationPanel):
        return {
            "type": str(item.__class__.__name__),
            "panel_name": item.panel_name,
            "panel_json": item.to_json(),
            "extensions": [e.__class__.__name__ for e in item.extensions],
            "extensions_data": {e.__class__.__name__: e.to_json() for e in item.extensions}
        }
    else:
        raise TypeError(f"Expected SplitItem or VisualizationPanel, got {type(item)}")


def _deserialize_split_item(
    data: Dict,
    full_data: Dict,
    temp_path: Path = None,
    include_files: bool = True
) -> Union["SplitItem", "VisualizationPanel"]:
    """
    Recursively deserializes a SplitItem from a dictionary.
    
    Uses load_panelX_from_file functions for panel deserialization,
    extracting panel data into temporary zip files matching the single-panel structure.
    
    Parameters
    ----------
    data : Dict
        Serialized representation of the item
    full_data : Dict
        Full layout serialized data
    temp_path : Path
        Temporary directory path containing extracted zip contents
    include_files : bool
        If True, includes loaded files in the slave deserialization
        
    Returns
    -------
    Union[SplitItem, VisualizationPanel]
        Deserialized item
    """
    from scivianna.layout.split import SplitItem, SplitDirection
    
    if data["type"] == "SplitItem":
        direction = SplitDirection[data["direction"]]
        return SplitItem(
            panel_1=_deserialize_split_item(data["panel_1"], full_data, temp_path, include_files),
            panel_2=_deserialize_split_item(data["panel_2"], full_data, temp_path, include_files),
            direction=direction
        )
    elif data["type"] in ("Panel1D", "Panel2D", "Panel3D", "PanelDataFrame"):
        panel_name = data["panel_name"]
        panel_json_data = data.get("panel_json", {})
        extensions = data.get("extensions", [])
        extensions_data = data.get("extensions_data", {})
        
        # Extract panel data into a temporary zip matching single-panel structure
        panel_zip_path = _extract_panel_to_temp_zip(
            temp_path, panel_name, full_data, panel_json_data,
            extensions, extensions_data
        )
        
        try:
            # Use the appropriate load_panelX_from_file function
            if data["type"] == "Panel1D":
                panel = load_panel1d_from_file(panel_zip_path, include_files=include_files)
                # Restore layout-specific attributes not in single-panel format
                panel_json_data = data["panel_json"]
                panel.sync_field = panel_json_data.get("sync_field", False)
                panel.update_event = panel_json_data.get("update_event", UpdateEvent.RANGE_CHANGE)
            elif data["type"] == "Panel2D":
                panel = load_panel2d_from_file(panel_zip_path, include_files=include_files)
            elif data["type"] == "Panel3D":
                panel = load_panel3d_from_file(panel_zip_path, include_files=include_files)
            elif data["type"] == "PanelDataFrame":
                panel = load_paneldatframe_from_file(panel_zip_path, include_files=include_files)
            else:
                raise ValueError(f"Unknown panel type: {data['type']}")
            
            return panel
        finally:
            # Clean up temporary zip
            if panel_zip_path.exists():
                panel_zip_path.unlink()
    else:
        raise ValueError(f"Unknown item type: {data['type']}")


# =============================================================================
# UTILITY FUNCTIONS FOR EXTENSIONS
# =============================================================================

def restore_extensions_state(
    panel: "VisualizationPanel",
    split_item_data: Dict,
    full_data: Dict
):
    """
    Restores the state of extensions for a given panel.
    
    Parameters
    ----------
    panel : VisualizationPanel
        Panel whose extensions to restore
    split_item_data : Dict
        Serialized split item data containing extension info
    full_data : Dict
        Full layout serialized data
    """
    if split_item_data["type"] == "SplitItem":
        # Recursively search for the panel in the split structure
        restore_extensions_state(panel, split_item_data["panel_1"], full_data)
        restore_extensions_state(panel, split_item_data["panel_2"], full_data)
    elif split_item_data["type"] == panel.__class__.__name__ and split_item_data.get("panel_name") == panel.panel_name:
        # Found the panel data, restore extensions
        extensions_data = split_item_data.get("extensions_data", {})
        for ext in panel.extensions:
            ext_class_name = ext.__class__.__name__
            if ext_class_name in extensions_data:
                ext_state = extensions_data[ext_class_name]
                ext.from_json(ext, ext_state)


# =============================================================================
# GRIDSTACK LAYOUT SERIALIZATION
# =============================================================================

def save_gridstack_to_zip(
    layout: "GridStackLayout",
    file_path: Union[str, Path],
    include_files: bool = True
) -> Path:
    """
    Saves the GridStackLayout configuration and slave data to a zip file.
    
    The zip file contains:
    - layout.json: JSON file describing the layout structure, current frame, 
      bounds, and for each panel: slave info and associated interface name
    - data/: Folder containing serialized data for each slave
    
    Parameters
    ----------
    layout : GridStackLayout
        The layout to save
    file_path : Union[str, Path]
        Path to the output zip file
    include_files : bool = True
        If True, includes loaded files in the slave serialization
        
    Returns
    -------
    Path
        Path to the saved zip file
    """
    from scivianna.panel.panel_2d import Panel2D
    
    file_path = Path(file_path)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Build layout metadata
        layout_data = {
            "current_frame": layout.current_frame,
            "bounds_x": {k: list(v) for k, v in layout.bounds_x.items()},
            "bounds_y": {k: list(v) for k, v in layout.bounds_y.items()},
            "time_widget": layout.time_widget.to_json() if hasattr(layout, 'time_widget') else None,
            "panels": {}
        }
        
        # Collect panel and slave information
        for panel_name, panel in layout.visualisation_panels.items():
            slave = panel.get_slave()
            interface_class = slave.code_interface
            
            # Find the key for this interface in INTERFACES dict
            interface_key = None
            for key, iface_class in INTERFACES.items():
                if iface_class == interface_class:
                    interface_key = key if isinstance(key, str) else key.value
                    break
            
            # If not found in built-in interfaces, use class name as fallback
            if interface_key is None:
                interface_key = interface_class.__name__
            
            # Determine panel type
            if isinstance(panel, Panel2D):
                panel_type = "Panel2D"
            elif hasattr(panel, "displayed_field") and not isinstance(panel, Panel1D):
                panel_type = "Panel3D"
            elif hasattr(panel, "plotter") and hasattr(panel.plotter, "get_data"):
                panel_type = "PanelDataFrame"
            else:
                panel_type = "Panel1D"
            
            layout_data["panels"][panel_name] = {
                "panel_type": panel_type,
                "slave_name": slave.code_interface.__name__,
                "interface_key": interface_key,
                "sync_field": panel.sync_field,
                "update_event": panel.update_event,
                "data_file": f"slave_data/{panel_name}.pkl",
                "panel_json": panel.to_json(),
                "extensions": [e.__class__.__name__ for e in panel.extensions],
                "extensions_data": {e.__class__.__name__: e.to_json() for e in panel.extensions}
            }
            
            # Save slave data to individual pickle file using the slave serialization function
            slave_data_path = temp_path / "slave_data" / f"{panel_name}.pkl"
            slave_data_path.parent.mkdir(parents=True, exist_ok=True)
            save_slave_to_file(slave, slave_data_path, include_files=include_files)
            
            # Save current_data for Panel2D and Panel3D using the utility function
            if panel_type in ("Panel2D", "Panel3D"):
                _save_current_data(panel.current_data, temp_path, panel_name)
                layout_data["panels"][panel_name]["current_data"] = f"data/{panel_name}.pkl"
            
            # Save dataframe for PanelDataFrame
            elif panel_type == "PanelDataFrame":
                df = panel.plotter.get_data()
                if df is not None:
                    dataframe_dict = df.to_dict(orient="list")
                    dataframe_path = temp_path / "dataframe" / f"{panel_name}.pkl"
                    dataframe_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(dataframe_path, "wb") as f:
                        pickle.dump(dataframe_dict, f)
                    layout_data["panels"][panel_name]["dataframe"] = f"dataframe/{panel_name}.pkl"
        
        # Write layout.json
        layout_json_path = temp_path / "layout.json"
        with open(layout_json_path, "w") as f:
            json.dump(layout_data, f, indent=2)
        
        # Create the zip file
        zip_path = file_path.with_suffix(".zip") if not file_path.suffix == ".zip" else file_path
        
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(temp_path):
                for file in files:
                    file_path_inside = Path(root) / file
                    arcname = file_path_inside.relative_to(temp_path)
                    zf.write(file_path_inside, arcname)
    
    return zip_path


def load_gridstack_from_zip(
    file_path: Union[str, Path],
    include_files: bool = True,
    additional_interfaces: Dict[Union[str, Any], Type[GenericInterface]] = {},
) -> "GridStackLayout":
    """
    Restores a GridStackLayout from a zip file.
    
    Uses load_panelX_from_file functions for panel deserialization,
    extracting panel data into temporary zip files matching the single-panel structure.
    
    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the input zip file
    include_files : bool = True
        If True, includes loaded files in the slave deserialization
    additional_interfaces : Dict = {}
        Additional interfaces to register
        
    Returns
    -------
    GridStackLayout
        Restored GridStackLayout instance
    """
    from scivianna.layout.gridstack import GridStackLayout
    
    file_path = Path(file_path)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        with zipfile.ZipFile(file_path, "r") as zf:
            zf.extractall(temp_path)
        
        # Read layout.json
        layout_json_path = temp_path / "layout.json"
        with open(layout_json_path, "r") as f:
            layout_data = json.load(f)
        
        # Register additional interfaces first
        from scivianna.interface import INTERFACES as IFACE_DICT
        for key, iface_class in additional_interfaces.items():
            register_interface(key, iface_class)
        
        # Rebuild panels with their loaded data using load_panelX_from_file functions
        restored_panels = {}
        bounds_x = {}
        bounds_y = {}
        
        for panel_name, panel_info in layout_data["panels"].items():
            # Extract panel data into a temporary zip matching single-panel structure
            panel_zip_path = _extract_panel_to_temp_zip(temp_path, panel_name, layout_data)
            
            try:
                # Determine panel type from saved JSON data
                panel_type = panel_info.get("panel_type", "Panel2D")
                
                # Use the appropriate load_panelX_from_file function
                if panel_type == "Panel2D":
                    panel = load_panel2d_from_file(panel_zip_path, include_files=include_files)
                elif panel_type == "Panel3D":
                    panel = load_panel3d_from_file(panel_zip_path, include_files=include_files)
                elif panel_type == "PanelDataFrame":
                    panel = load_paneldatframe_from_file(panel_zip_path, include_files=include_files)
                else:  # Panel1D
                    panel = load_panel1d_from_file(panel_zip_path, include_files=include_files)
                
                restored_panels[panel_name] = panel
            finally:
                # Clean up temporary zip
                if panel_zip_path.exists():
                    panel_zip_path.unlink()
            
            bounds_x[panel_name] = tuple(layout_data["bounds_x"][panel_name])
            bounds_y[panel_name] = tuple(layout_data["bounds_y"][panel_name])
        
        # Create the layout instance
        layout = GridStackLayout(
            visualisation_panels=restored_panels,
            bounds_x=bounds_x,
            bounds_y=bounds_y,
            additional_interfaces=additional_interfaces
        )
        
        # Restore time_widget if it was saved
        time_widget_data = layout_data.get("time_widget")
        if time_widget_data is not None:
            from scivianna.extension.coupling import CouplingExtension
            layout.time_widget = CouplingExtension(layout, None, None, None)
            CouplingExtension.from_json(layout.time_widget, time_widget_data)
            layout.gui.add_extension(layout.time_widget)

            for panel in layout.visualisation_panels.values():
                panel.panel_coupling_extension = layout.time_widget
        
        # Set the current frame
        layout.set_to_frame(layout_data["current_frame"])
        
        return layout
