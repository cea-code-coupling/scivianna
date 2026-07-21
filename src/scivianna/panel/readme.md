# Scivianna Panel Module

The `panel` module provides the core visualization panel classes that display and interact with simulation data. Panels are the primary user interface for visualizing 1D, 2D, and 3D data in Scivianna.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initialization file |
| `visualisation_panel.py` | Base class `VisualizationPanel` for all panel types |
| `panel_1d.py` | 1D plot panel for line charts and time series |
| `panel_2d.py` | 2D plot panel for geometry visualization (polygons, grids) |
| `panel_3d.py` | 3D plot panel for VTK-based volumetric visualization |
| `panel_dataframe.py` | DataFrame panel for displaying pandas DataFrames |
| `gui.py` | GUI manager for panel controls and sidebars |
| `demo.py` | Demonstrator application with example panels |

## VisualizationPanel Class

The `VisualizationPanel` is the base class for all visualization panels:

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `panel_name` | str | Name identifier for the panel |
| `slave` | ComputeSlave | Data slave providing geometry and field data |
| `plotter` | Plotter2D / Plotter3D | Plotting backend (Bokeh/Matplotlib for 2D, VTK.js for 3D) |
| `main_frame` | pmui.Container | Main display container |
| `extensions` | List[Extension] | Attached extension components |
| `figure` | Overlay | Figure with overlay controls |
| `current_data` | DataContainer (Data1D / Data2D / Data3D) | Currently displayed data |
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
- Mesh visualizations (polygon or grid rendering)
- Field color maps
- 2D slices from 3D geometry
- Supports zoom, pan, and interactive clipping

### Panel3D
Displays three-dimensional VTK-based geometry:
- PolyData and UnstructuredGrid visualization
- Interactive clip planes for slicing
- 3D camera controls (rotate, pan, zoom)
- Color mapping on 3D cell values
- Keyboard shortcuts: `C` (toggle clipping), `X`/`Y`/`Z` (clip axis)

### PanelDataFrame
Displays pandas DataFrames in an interactive table view:
- Linked to 2/3DD panels via `MOUSE_CELL_CHANGE` events
- Automatically updates when hovering over cells in a linked geometry panel

## UpdateEvent System

Panels can interact with each other through **UpdateEvents**. When panels are placed in the same layout, events from one panel can trigger updates in others.

### Available Events

| Event | Value | Description |
|-------|-------|-------------|
| `RECOMPUTE` | 0 | Manual recompute only (default, no auto-update) |
| `CLIC` | 1 | Update on mouse click (sends/receives mouse location and cell ID) |
| `MOUSE_POSITION_CHANGE` | 2 | Update on mouse move over 2D plot |
| `MOUSE_CELL_CHANGE` | 3 | Update when mouse enters a new cell on 2D plot |
| `PERIODIC` | 4 | Periodic updates for real-time simulation coupling |
| `RANGE_CHANGE` | 5 | Update when zoom/pan bounds change on 2D plot |
| `AXES_CHANGE` | 6 | Update when axes orientation or origin changes |

### Inter-Panel Communication

Panels can send and receive events based on their type:

#### Sending Panel

| Event | Panel1D | Panel2D | Panel3D |
|-------|:-------:|:-------:|:-------:|
| CLIC | - | Sends mouse location and cell ID on click | Sends mouse location and cell ID on click |
| MOUSE_POSITION_CHANGE | - | Sends mouse location and cell ID on mouse move | Sends mouse location and cell ID on mouse move |
| MOUSE_CELL_CHANGE | - | Sends mouse location and cell ID on hovered cell | Sends mouse location and cell ID on hovered cell |
| RANGE_CHANGE | - | Sends frame center and size on zoom/drag | - |
| AXES_CHANGE | - | Sends new axes on axes change | Sends clip plane origin and axes on translation/rotation |

#### Receiving Panel

| Event | Panel1D | Panel2D | Panel3D |
|-------|:-------:|:-------:|:-------:|
| CLIC | Updates at mouse location/cell ID | Updates moving origin to mouse location | Moves clip plane to new origin |
| MOUSE_POSITION_CHANGE | Updates at mouse location/cell ID | Updates moving origin to mouse location | Moves clip plane to new origin |
| MOUSE_CELL_CHANGE | Updates at mouse location/cell ID | Updates moving origin to mouse location | Moves clip plane to new origin |
| RANGE_CHANGE | - | Updates for new origin/range | Moves clip plane to new origin |
| AXES_CHANGE | - | Updates for new origin/axes | Moves clip plane to new origin/axes |

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
from scivianna.enums import UpdateEvent

# Create slave with interface
slave = ComputeSlave(MEDInterface)
slave.read_file("result.med")

# Create panel with extensions and update event
panel = Panel2D(
    slave=slave,
    name="Temperature View",
    update_event=UpdateEvent.CLIC,  # Update on click
)

# Display the panel
panel.show()
```

## Event Handling

Panels respond to user interactions:
- **Mouse click**: Trigger `on_mouse_clic` callbacks
- **Mouse hover**: Trigger `on_mouse_move` callbacks
- **Field change**: Update all synced panels
- **Zoom/Pan**: Trigger `RANGE_CHANGE` events if configured
- **Axes/Clip changes**: Trigger `AXES_CHANGE` events if configured

## Serialization

Panels can be serialized for save/load:
```python
# Save panel state
state = panel.to_json()

# Restore panel
restored_panel = Panel2D.from_json(state, data_container)
```
