# Scivianna Panel Module

The `panel` module provides the core visualization panel classes that display and interact with simulation data. Panels are the primary user interface for visualizing 1D and 2D data in Scivianna.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initialization file |
| `visualisation_panel.py` | Base class `VisualizationPanel` for all panel types |
| `panel_1d.py` | 1D plot panel for line charts and time series |
| `panel_2d.py` | 2D plot panel for geometry visualization |
| `gui.py` | GUI manager for panel controls and sidebars |
| `demo.py` | Demonstrator application with example panels |

## VisualizationPanel Class

The `VisualizationPanel` is the base class for all visualization panels:

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `panel_name` | str | Name identifier for the panel |
| `slave` | ComputeSlave | Data slave providing geometry and field data |
| `plotter` | Plotter2D | 2D plotting backend (Bokeh/Matplotlib) |
| `main_frame` | pmui.Container | Main display container |
| `extensions` | List[Extension] | Attached extension components |
| `figure` | Overlay | Figure with overlay controls |
| `current_data` | DataContainer | Currently displayed data |
| `update_event` | UpdateEvent | Event triggering panel updates |
| `sync_field` | bool | Whether to sync fields with other panels |

### Key Methods

| Method | Purpose |
|--------|---------|
| `recompute()` | Refresh the panel display |
| `get_slave()` | Return the data slave |
| `duplicate()` | Create a copy of the panel |
| `set_field(field_name)` | Change displayed field |
| `set_colormap(colormap)` | Change color map |
| `outline_color(color)` | Set border color |
| `provide_on_clic_callback(fn)` | Register click handler |
| `provide_on_mouse_move_callback(fn)` | Register mouse move handler |
| `recompute_at(position, cell_id)` | Trigger recompute at location |

## Panel Types

### Panel1D
Displays one-dimensional data:
- Line charts
- Time series plots

### Panel2D
Displays two-dimensional geometries:
- Mesh visualizations
- Field color maps
- Polygon-based displays

## GUI Integration

Each panel includes a GUI manager (`gui.py`) that handles:
- Extension buttons and sidebars
- Drawer navigation
- Active extension tracking
- Button visibility management

## Usage Example

```python
from scivianna.panel.panel_2d import Panel2D
from scivianna.slave import ComputeSlave
from scivianna.interface.med_interface import MEDInterface

# Create slave with interface
slave = ComputeSlave(MEDInterface)
slave.read_file("result.med")

# Create panel with extensions
panel = VisualizationPanel(
    slave=slave,
    name="Temperature View",
    extensions=[FieldSelector, LayoutExtension]
)

# Display the panel
panel.show()
```

## Event Handling

Panels respond to user interactions:
- **Mouse click**: Trigger `on_mouse_clic` callbacks
- **Mouse hover**: Trigger `on_mouse_move` callbacks  
- **Field change**: Update all synced panels

## Serialization

Panels can be serialized for save/load:
```python
# Save panel state
state = panel.to_json()

# Restore panel
restored_panel = Panel2D.from_json(state, data_container)