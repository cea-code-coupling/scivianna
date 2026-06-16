# Scivianna Utils Module

The `utils` module provides utility functions and helper classes used throughout Scivianna for mesh processing, file handling, serialization, and visualization tools.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initialization file |
| `color_tools.py` | Color manipulation utilities (RGBA handling, colormap generation) |
| `extruded_mesh.py` | Tools for visualizing extruded 2D meshes in 3D |
| `file_cleaner.py` | Utilities for cleaning input files when the visualizer is closed |
| `interface_tools.py` | Helper functions for interface implementations |
| `polygon_sorter.py` | Algorithms for sorting and organizing polygon data |
| `polygonize_tools.py` | Tools for converting grid data to polygon representations |
| `serialization.py` | Save/load functionality for panel states and layouts |
| `structured_mesh.py` | Utilities for structured mesh handling |

## Key Utilities

### Color Tools (`color_tools.py`)
- RGBA color array manipulation
- Colormap generation and application
- Alpha/transparency adjustments
- Color interpolation

### Extruded Mesh (`extruded_mesh.py`)
Converts 2D meshes to 3D extruded representations:
- Takes 2D polygon data
- Extrudes along Z-axis
- Generates 3D visualization coordinates

### Polygonize Tools (`polygonize_tools.py`)
Converts raster/grid data to vector polygons:
- Contour extraction from scalar fields
- Polygon boundary detection
- Mesh cell polygonization

### Serialization (`serialization.py`)
Save and load Scivianna states:
- GridStack layout serialization
- Panel state persistence
- ZIP-based save files

```python
from scivianna.utils.serialization import save_gridstack_to_zip, load_gridstack_from_zip

# Save layout
save_gridstack_to_zip(layout, "my_save.zip")

# Load layout
layout = load_gridstack_from_zip("my_save.zip")
```

### Structured Mesh (`structured_mesh.py`)
Utilities for structured mesh operations:
- Grid generation
- Coordinate transformations
- Cell indexing utilities

### Interface Tools (`interface_tools.py`)
Common helpers for interface implementations:
- Default panel creation
- Interface enumeration
- File validation

### Polygon Sorter (`polygon_sorter.py`)
Algorithms for organizing polygon data:
- Spatial sorting for rendering order
- Adjacency-based ordering
- Performance optimization for large datasets

## Usage Examples

### Color Manipulation
```python
from scivianna.utils.color_tools import rgba_to_hex, hex_to_rgba

# Convert between formats
hex_color = rgba_to_hex([255, 128, 0, 255])  # "#ff8000"
rgba = hex_to_rgba("#ff8000")  # [255, 128, 0, 255]
```
