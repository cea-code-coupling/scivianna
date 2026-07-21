# Scivianna Interface Module

The `interface` module provides data interfaces for reading simulation results from various file formats. Interfaces abstract data access and provide a unified API for visualization.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initialization file, registers available interfaces |
| `generic_interface.py` | Base class `GenericInterface` with common functionality |
| `med_interface.py` | Interface for Salome MED file format |
| `vtk_interface.py` | Interface for VTK/VTU file formats |
| `csv_result.py` | Interface for CSV time-series data |
| `structured_mesh_interface.py` | Interface for structured mesh data |
| `time_dataframe.py` | Utility class for time-indexed data frames |

## Interface Hierarchy

### GenericInterface (Base Class)
Provides core functionality for all interfaces:

| Function | Description |
|----------|-------------|
| `read_file(file_path, file_label)` | Reads input files to extract data |
| `get_labels()` | Returns list of available fields/results |
| `get_label_coloring_mode(label)` | Returns whether a field is colored by string or float values |
| `get_file_input_list()` | Returns list of loaded files with descriptions |
| `serialize(obj, key)` | Serializes objects for multiprocessing queue transmission |
| `save(file_path, include_files)` | Saves interface state to a pickle file |
| `load(file_path, include_files)` | Loads interface state from a pickle file |
| `get_slave()` | Returns a ComputeSlave instance for the interface class |

### Data Capability Interfaces
Interfaces inherit from these capability classes based on what they provide:

| Interface | Purpose |
|-----------|---------|
| `Geometry2D` | Provides 2D cell geometry and field values |
| `Geometry2DPolygon` | Provides 2D geometry as a list of polygons |
| `Geometry2DGrid` | Provides 2D geometry as a numpy array (rasterized) |
| `ValueAtLocation` | Provides values at specific locations/cells/materials |
| `Value1DAtLocation` | Provides 1D data profiles at locations |
| `CouplingInterface` | Supports runtime coupling updates (C3PO/Icoco) |

### Geometry2D Interface
Extends GenericInterface with 2D geometry capabilities:

| Function | Description |
|----------|-------------|
| `compute_2D_data(u, v, origin, size_u, size_v, w_value, q_tasks, options, caller)` | Returns 2D geometry polygons for a given frame |
| `get_value_dict(value_label, cells, options, caller)` | Returns cell name to field value mapping |

### Geometry2DPolygon Interface
Subclass of Geometry2D that provides polygon-based geometry. Inherits all Geometry2D methods.

### Geometry2DGrid Interface
Subclass of Geometry2D that provides grid-based (rasterized) geometry. Inherits all Geometry2D methods.

### ValueAtLocation Interface
Provides point-value access capabilities:

| Function | Description |
|----------|-------------|
| `get_value(position, cell_index, material_name, field, options)` | Returns field value at a specific location/cell/material |
| `get_values(positions, cell_indexes, material_names, field, options)` | Returns field values at multiple locations/cells/materials |

### Value1DAtLocation Interface
Provides 1D data profile access:

| Function | Description |
|----------|-------------|
| `get_1D_value(position, cell_index, material_name, field, options)` | Returns 1D data series (pd.Series) at a specific location/cell/material |

### CouplingInterface
Provides runtime coupling capabilities for C3PO/Icoco integration:

| Function | Description |
|----------|-------------|
| `set_time(time)` | Sets the current time for coupling updates |
| `update_data(key, data)` | Replaces interface data with new value |
| `append_data(key, data)` | Appends data associated with current time |
| `update_mesh(key, data)` | Replaces interface mesh and data with new value |
| `append_mesh(key, data)` | Appends mesh and data associated with current time |
| `get_template(name)` | Returns template for C3PO getOutputxxxFieldTemplate functions |
| `set_template(name, template)` | Sets the template for C3PO functions |

## Available Interfaces

### MED Interface
Reads Salome MED files containing:
- Mesh geometry (2D/3D)
- Field data on cells/nodes
- Time-step information

```python
from scivianna.interface.med_interface import MEDInterface

interface = MEDInterface()
interface.read_file("result.med")
labels = interface.get_labels()  # ["temperature", "pressure", ...]
```

### VTK Interface
Reads VTK/VTU files for unstructured grids:
- Polygon/polyhedron cells
- Point and cell data
- Multiple time steps

### CSV Interface
Reads structured CSV data:
- Time-series results
- Column-based variables
- Automatic parsing with pandas

## Creating Custom Interfaces

Custom interfaces can be easily created, the scivianna.interface.generic_interface file defines what functions need to be provided for a new interface.

```python
from scivianna.interface.generic_interface import GenericInterface, Geometry2D

class MyCustomInterface(GenericInterface, Geometry2D):
    def read_file(self, filepath):
        # Implement file reading logic
        pass
    
    def get_labels(self):
        # Return available field names
        return ["field1", "field2"]
    
    def get_geometry_2d(self, field_name):
        # Return 2D geometry for the field
        pass
```

## Interface Registration

Built-in interfaces are automatically discovered and registered via `__init__.py`. Additional user interfaces can be implemeted by calling:

```
from scivianna.interface import register_interface

register_interface("MyInterfaceKey", MyInterfaceClass)
```