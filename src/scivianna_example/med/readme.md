# MED Coupling Example

This example demonstrates how to visualize Salome MED file data using Scivianna, with synchronized multi-view displays and interactive exploration.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initialization file |
| `single_med.py` | Single MED file visualization example |
| `split_item_example.py` | Multi-view synchronized panel example |
| `grid_stack_example.py` | GridStack layout with MED data |
| `split_serialization.py` | Serialization for split layouts |
| `gridstack_serialization.py` | Serialization for GridStack layouts |
| `plot_api.py` | Plotting API for MED visualization |
| `description.md` | Example description for the demonstrator |
| `readme.md` | This documentation file |

## Overview

The MED Coupling example showcases:
- Reading and visualizing Salome MED file format
- Synchronized multi-view displays (3 views of same mesh)
- Click interaction to offset views at click location
- GridStack layout management
- State serialization and restoration

## Features

### Multi-View Synchronization
Three views of the same MED mesh are displayed simultaneously:
- View 1: Full geometry view
- View 2: Zoomed region
- View 3: Alternative angle or field

### Interactive Exploration
- Click in one view to offset all views at that location
- Field selection synchronized across views
- Color map shared between panels

### GridStack Layout
Panels arranged in a draggable, resizable grid:
- Rearrange views by dragging
- Resize panels dynamically
- Save and restore layout state

## Usage

### Single MED File
```python
from scivianna_example.med.single_med import get_panel

# Create panel from MED file
panel = get_panel("path/to/file.med")
```

### Multi-View Split Layout
```python
from scivianna_example.med.split_item_example import get_panel

# Create synchronized multi-view panel
panel = get_panel(None)  # None for demo mode with mock data
```

### GridStack Example
```python
from scivianna_example.med.grid_stack_example import make_layout

# Create GridStack layout with MED visualization
layout = make_layout()
layout.show()
```

## Serialization

Save and restore panel configurations:

```python
from scivianna_example.med.split_serialization import save_split, load_split

# Save state
save_split(panel, "save_state.zip")

# Load state
panel = load_split("save_state.zip", data_container)
```

## MED File Format

The example handles Salome MED files containing:
- 2D/3D mesh geometry
- Field data on cells or nodes
- Multiple time steps
- Multiple fields per file

## Plot API

The `plot_api.py` provides:
- Field selection widgets
- Color map configuration
- View synchronization callbacks
- Click event handlers for multi-view updates