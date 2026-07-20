# Scivianna Data Module

The `data` module provides core data structures and workers for managing simulation data in Scivianna, including 1D plots, 2D geometries, and 3D meshes.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initialization file |
| `data_container.py` | Base container class for holding and managing visualization data |
| `data1d.py` | Data structures for 1D plot data (line charts, time series) |
| `data2d.py` | Data structures for 2D geometry data (meshes, fields, polygons) |
| `data3d.py` | Data structures for 3D geometry data (VTK polydata or unstructured mesh, cell values) |
| `data_2d_worker.py` | Worker process for handling 2D data operations asynchronously |

## Data Types

### Data1D
Handles one-dimensional data for line plots:
- Time series data
- Line charts with multiple variables

### Data2D
Handles two-dimensional geometry data:
- Mesh cells with associated values
- Color maps (RGBA per cell)
- Alpha/transparency values
- Polygon-based or grid-based geometries

### Data3D
Handles three-dimensional VTK-based geometry data:
- VTK PolyData or UnstructuredGrid references
- Cell IDs, values, colors, and edge colors
- Built from `pv.PolyData` or `pv.UnstructuredGrid` objects via `from_vtk()` class method
- Supports deep copy and cell data updates

### DataContainer
Base container that manages visualization data properties:
- Stores current field values
- Manages color bar properties
- Tracks data metadata and labels

## Key Features

- **Color Management**: Built-in support for RGBA color arrays (0-255 range)
- **Field Labels**: Automatic tracking of available fields for selection widgets
- **Data Validation**: Methods to verify data integrity before visualization (`check_valid()`)
- **3D Support**: VTK-based 3D data structures with PolyData integration

## Integration

The data module integrates with:
- **Interfaces**: Data is loaded via interface modules (MED, VTK, CSV, structured mesh)
- **Plotters**: Data is rendered by 1D, 2D, and 3D plotter backends
- **Agents**: AI agent can manipulate Data2D properties programmatically
