"""
Serialization utilities for scivianna components.

This module provides serialization and deserialization functions at different levels:
- slave_only: Serialize/deserialize only the ComputeSlave
- panel2d_only: Serialize/deserialize only Panel2D (without slave data)
- panel1d_only: Serialize/deserialize only Panel1D (without slave data)
- layout: Serialize/deserialize complete SplitLayout with all components

Note: To avoid circular imports, actual module imports are done lazily inside functions.
"""

from __future__ import annotations

import pickle
import json
import zipfile
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Tuple, Type, Union, TYPE_CHECKING, Optional, Any

if TYPE_CHECKING:
    from scivianna.panel.panel_2d import Panel2D
    from scivianna.panel.panel_1d import Panel1D
    from scivianna.panel.visualisation_panel import VisualizationPanel
    from scivianna.layout.split import SplitLayout, SplitItem

from scivianna.interface.generic_interface import GenericInterface
from scivianna.interface import INTERFACES, register_interface
from scivianna.slave import ComputeSlave

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
    from scivianna.panel.visualisation_panel import ComputeSlave
    
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
        
        # Save panel configuration as JSON
        panel_data = {
            "panel_type": "Panel2D",
            "panel_json": panel.to_json(),
            "extensions": [e.__class__.__name__ for e in panel.extensions],
            "extensions_data": {e.__class__.__name__: e.to_json() for e in panel.extensions},
            "interface_key": interface_key,
            "slave_file": "slave.pkl",
            "current_data_file": "current_data.pkl"
        }
        
        panel_json_path = temp_path / "panel.json"
        with open(panel_json_path, "w") as f:
            json.dump(panel_data, f, indent=2)
        
        # Save slave data
        slave_data_path = temp_path / "slave.pkl"
        slave.save(slave_data_path, include_files=include_files)
        
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
    from scivianna.panel.visualisation_panel import ComputeSlave
    
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
        
        # Create slave and load data
        slave = ComputeSlave(interface)
        
        # Load slave data from the pickle file
        slave_data_path = temp_path / panel_data.get("slave_file", "slave.pkl")
        if slave_data_path.exists():
            slave.load(slave_data_path, include_files=include_files)
        
        # Load current_data
        current_data = None
        current_data_path = temp_path / panel_data.get("current_data_file", "current_data.pkl")
        if current_data_path.exists():
            with open(current_data_path, "rb") as f:
                current_data = pickle.load(f)
        
        # Rebuild extensions with their state
        extensions = []
        extensions_data = panel_data.get("extensions_data", {})
        saved_extensions = panel_data.get("extensions", [])
        
        for ext_name in saved_extensions:
            ext_state = extensions_data.get(ext_name, {})
            ext_class = _get_panel2d_extension_class(ext_name, interface)
            
            if ext_class is not None:
                extensions.append((ext_class, ext_state))

        # Restore panel from json with the loaded data
        panel = Panel2D.from_json(panel_data["panel_json"], slave, current_data, extensions)
        
        return panel


def _get_panel2d_extension_class(ext_name: str, interface: Type[GenericInterface]) -> Optional[Type]:
    """Helper method to find a Panel2D extension class by name."""
    # First, try to find in interface.extensions
    for iface_ext in interface.extensions:
        if iface_ext.__name__ == ext_name:
            return iface_ext
    
    # Then try panel's default_extensions
    from scivianna.panel.panel_2d import default_extensions as panel2d_default_extensions
    for ext in panel2d_default_extensions:
        if ext.__name__ == ext_name:
            return ext
    
    return None


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
        
        # Save panel configuration as JSON
        panel_data = {
            "panel_type": "Panel1D",
            "panel_json": panel.to_json(),
            "extensions": [e.__class__.__name__ for e in panel.extensions],
            "extensions_data": {e.__class__.__name__: e.to_json() for e in panel.extensions},
            "interface_key": interface_key,
            "slave_file": "slave.pkl"
        }
        
        panel_json_path = temp_path / "panel.json"
        with open(panel_json_path, "w") as f:
            json.dump(panel_data, f, indent=2)
        
        # Save slave data
        slave_data_path = temp_path / "slave.pkl"
        slave.save(slave_data_path, include_files=include_files)
        
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
    from scivianna.panel.visualisation_panel import ComputeSlave
    
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
        
        # Create slave and load data
        slave = ComputeSlave(interface)
        
        # Load slave data from the pickle file
        slave_data_path = temp_path / panel_data.get("slave_file", "slave.pkl")
        if slave_data_path.exists():
            slave.load(slave_data_path, include_files=include_files)
        
        # Rebuild extensions with their state
        extensions = []
        extensions_data = panel_data.get("extensions_data", {})
        saved_extensions = panel_data.get("extensions", [])
        
        for ext_name in saved_extensions:
            ext_state = extensions_data.get(ext_name, {})
            ext_class = _get_panel1d_extension_class(ext_name, interface)
            
            if ext_class is not None:
                extensions.append((ext_class, ext_state))
        
        # Restore panel from json
        panel = Panel1D.from_json(panel_data["panel_json"], slave, extensions)
        
        return panel


def _get_panel1d_extension_class(ext_name: str, interface: Type[GenericInterface]) -> Optional[Type]:
    """Helper method to find a Panel1D extension class by name."""
    # First, try to find in interface.extensions
    for iface_ext in interface.extensions:
        if iface_ext.__name__ == ext_name:
            return iface_ext
    
    # Then try panel's default_extensions
    from scivianna.panel.panel_1d import default_extensions as panel1d_default_extensions
    for ext in panel1d_default_extensions:
        if ext.__name__ == ext_name:
            return ext
    
    return None


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
    
    file_path = Path(file_path)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Build layout metadata
        layout_data = {
            "split_item": _serialize_split_item(layout.split_item),
            "current_frame": layout.current_frame,
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
                "current_data": f"data/{panel_name}.pkl"
            }
            
            # Save slave data to individual pickle file
            slave_data_path = temp_path / "slave_data" / f"{panel_name}.pkl"
            slave_data_path.parent.mkdir(parents=True, exist_ok=True)
            slave.save(slave_data_path, include_files=include_files)
            
            # Save current_data for Panel2D
            if isinstance(panel, Panel2D):
                data_path = temp_path / "data" / f"{panel_name}.pkl"
                data_path.parent.mkdir(parents=True, exist_ok=True)
                with open(data_path, "wb") as f:
                    pickle.dump(panel.current_data, f)
        
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
    additional_interfaces: Dict[Union[str, Any], Type[GenericInterface]] = {},
    add_run_button: bool = False
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
    add_run_button : bool = False
        Whether to add a run button for coupling simulations
        
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
            additional_interfaces=additional_interfaces,
            add_run_button=add_run_button
        )
        
        # Set the current frame
        layout.set_to_frame(layout_data["current_frame"])
        
        return layout


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
    from scivianna.panel.visualisation_panel import ComputeSlave
    from scivianna.panel.panel_1d import Panel1D
    from scivianna.panel.panel_2d import Panel2D
    
    if data["type"] == "SplitItem":
        direction = SplitDirection[data["direction"]]
        return SplitItem(
            panel_1=_deserialize_split_item(data["panel_1"], full_data, temp_path, include_files),
            panel_2=_deserialize_split_item(data["panel_2"], full_data, temp_path, include_files),
            direction=direction
        )
    elif data["type"] == "Panel1D":
        panel_name = data["panel_name"]
        panel_info = full_data["panels"][panel_name]
        panel_json_data = data["panel_json"]
        
        # Get the interface class from the INTERFACES dictionary using the interface_key
        interface_key = panel_info["interface_key"]
        interface = INTERFACES.get(interface_key)
        
        if interface is None:
            raise KeyError(f"Interface '{interface_key}' not found in INTERFACES dictionary")
        
        # Create slave and load data first (before panel creation)
        slave = ComputeSlave(interface)
        
        # Load slave data from the pickle file BEFORE creating the panel
        if temp_path:
            slave_data_path = temp_path / panel_info["data_file"]
            if slave_data_path.exists():
                slave.load(slave_data_path, include_files=include_files)
        
        # Get all saved extensions and their states
        extensions = []
        extensions_data = data.get("extensions_data", {})
        saved_extensions = data.get("extensions", [])
        
        for ext_name in saved_extensions:
            ext_state = extensions_data.get(ext_name, {})
            ext_class = _get_extension_class(ext_name, interface, "Panel1D")
            
            # If found, add it with its state
            if ext_class is not None:
                extensions.append((ext_class, ext_state))
        
        # Restore panel from json, passing extensions with their state
        panel = Panel1D.from_json(panel_json_data, slave, extensions)
        panel.sync_field = panel_json_data["sync_field"]
        panel.update_event = panel_json_data["update_event"]
        
        return panel
        
    elif data["type"] == "Panel2D":
        panel_name = data["panel_name"]
        panel_info = full_data["panels"][panel_name]
        panel_json_data = data["panel_json"]
        
        # Get the interface class from the INTERFACES dictionary using the interface_key
        interface_key = panel_info["interface_key"]
        interface = INTERFACES.get(interface_key)
        
        if interface is None:
            raise KeyError(f"Interface '{interface_key}' not found in INTERFACES dictionary")
        
        # Create slave and load data first (before panel creation)
        slave = ComputeSlave(interface)
        
        # Load slave data from the pickle file BEFORE creating the panel
        if temp_path:
            slave_data_path = temp_path / panel_info["data_file"]
            if slave_data_path.exists():
                slave.load(slave_data_path, include_files=include_files)
        
        # Load current_data from pickle file for Panel2D
        current_data = None
        if temp_path:
            data_path = temp_path / panel_info["current_data"]
            if data_path.exists():
                with open(data_path, "rb") as f:
                    current_data = pickle.load(f)
        
        # Get all saved extensions and their states
        extensions = []
        extensions_data = data.get("extensions_data", {})
        saved_extensions = data.get("extensions", [])
        
        for ext_name in saved_extensions:
            ext_state = extensions_data.get(ext_name, {})
            ext_class = _get_extension_class(ext_name, interface, "Panel2D")
            
            # If found, add it with its state
            if ext_class is not None:
                extensions.append((ext_class, ext_state))
        
        # Restore panel from json with the loaded data
        panel = Panel2D.from_json(panel_json_data, slave, current_data, extensions)
        
        return panel
        
    else:
        raise ValueError(f"Unknown item type: {data['type']}")


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
        Type of panel ("Panel1D" or "Panel2D")
        
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
    
    return None


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
            panel_type = "Panel2D" if isinstance(panel, Panel2D) else "Panel1D"
            
            layout_data["panels"][panel_name] = {
                "panel_type": panel_type,
                "slave_name": slave.code_interface.__name__,
                "interface_key": interface_key,
                "sync_field": panel.sync_field,
                "update_event": panel.update_event,
                "data_file": f"slave_data/{panel_name}.pkl",
                "current_data": f"data/{panel_name}.pkl" if isinstance(panel, Panel2D) else None
            }
            
            # Save slave data to individual pickle file
            slave_data_path = temp_path / "slave_data" / f"{panel_name}.pkl"
            slave_data_path.parent.mkdir(parents=True, exist_ok=True)
            slave.save(slave_data_path, include_files=include_files)
            
            # Save current_data for Panel2D
            if isinstance(panel, Panel2D):
                data_path = temp_path / "data" / f"{panel_name}.pkl"
                data_path.parent.mkdir(parents=True, exist_ok=True)
                with open(data_path, "wb") as f:
                    pickle.dump(panel.current_data, f)
        
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
    add_run_button: bool = False
) -> "GridStackLayout":
    """
    Restores a GridStackLayout from a zip file.
    
    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the input zip file
    include_files : bool = True
        If True, includes loaded files in the slave deserialization
    additional_interfaces : Dict = {}
        Additional interfaces to register
    add_run_button : bool = False
        Whether to add a run button for coupling simulations
        
    Returns
    -------
    GridStackLayout
        Restored GridStackLayout instance
    """
    from scivianna.layout.gridstack import GridStackLayout
    from scivianna.panel.panel_2d import Panel2D
    from scivianna.panel.panel_1d import Panel1D
    
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
        
        # Rebuild panels with their loaded data
        restored_panels = {}
        bounds_x = {}
        bounds_y = {}
        
        for panel_name, panel_info in layout_data["panels"].items():
            # Get the interface class from the INTERFACES dictionary using the interface_key
            interface_key = panel_info["interface_key"]
            interface = INTERFACES.get(interface_key)
            
            if interface is None:
                raise KeyError(f"Interface '{interface_key}' not found in INTERFACES dictionary")
            
            # Create slave and load data
            slave = ComputeSlave(interface)
            slave_data_path = temp_path / panel_info["data_file"]
            if slave_data_path.exists():
                slave.load(slave_data_path, include_files=include_files)
            
            # Determine panel type from saved JSON data
            panel_type = panel_info.get("panel_type", "Panel2D")
            
            # Load current_data for Panel2D
            current_data = None
            if panel_type == "Panel2D" and panel_info.get("current_data"):
                data_path = temp_path / panel_info["current_data"]
                if data_path.exists():
                    with open(data_path, "rb") as f:
                        current_data = pickle.load(f)
            
            # Create the panel - Panel2D needs data, Panel1D doesn't
            if panel_type == "Panel2D":
                panel = Panel2D(slave, name=panel_name, data=current_data)
            else:
                panel = Panel1D(slave, name=panel_name)
            
            restored_panels[panel_name] = panel
            bounds_x[panel_name] = tuple(layout_data["bounds_x"][panel_name])
            bounds_y[panel_name] = tuple(layout_data["bounds_y"][panel_name])
        
        # Create the layout instance
        layout = GridStackLayout(
            visualisation_panels=restored_panels,
            bounds_x=bounds_x,
            bounds_y=bounds_y,
            additional_interfaces=additional_interfaces,
            add_run_button=add_run_button
        )
        
        # Set the current frame
        layout.set_to_frame(layout_data["current_frame"])
        
        return layout


