# Scivianna Extension Module

The `extension` module provides extensible components that add functionality to visualization panels in Scivianna. Extensions can be attached to panels to provide additional controls, tools, and features.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initialization file |
| `extension.py` | Base class for all panel extensions |
| `axes.py` | 3D coordinates and axis configuration extension |
| `coupling.py` | Coupling-specific controls (time widget, play button) |
| `field_selector.py` | Widget for selecting displayed fields |
| `file_loader.py` | File upload and loading interface |
| `layout.py` | Layout management and panel arrangement tools |
| `line_selector.py` | Tool for setting 1D plots |
| `save_load_extension.py` | Serialization and state save/load functionality |
| `slice_3d.py` | 3D clip plane controls for slicing geometry |
| `ai_assistant.py` | AI-powered assistant for visualization help |
| `icons/` | Icon images for extension UI elements |

## Extension System

Extensions are modular components that can be:
- Added to any visualization panel
- Configured independently
- Serialized and restored with panel state
- Triggered by user interactions or events

## Built-in Extensions

### Layout Extension (`layout.py`)
Manages panel arrangement, frame switching, and interface selection.

### Coupling Extension (`coupling.py`)
Provides time management widget and play/pause controls for coupled simulations.

### Field Selector (`field_selector.py`)
Dropdown widget for selecting which data field to visualize.

### File Loader (`file_loader.py`)
Interface for uploading and loading data files from local or server storage.

### Line Selector (`line_selector.py`)
Interactive tool for setting 1D plots.

### Axes Extension (`axes.py`)
Controls for 3D coordinate display and axis configuration.

### Slice Plane Extension (`slice_3d.py`)
Interactive 3D clip plane controls:
- Enable/disable clipping with checkbox
- Select clipping axis (X, Y, Z)
- Adjust clip plane position with slider
- Keyboard shortcuts in 3D viewer: `C` (toggle), `X`/`Y`/`Z` (axis), `↑`/`↓` (move)

### Save/Load Extension (`save_load_extension.py`)
Enables saving panel configurations and restoring them later.

### AI Assistant (`ai_assistant.py`)
Provides AI-powered help and suggestions for visualization tasks.

## Creating Custom Extensions

Custom extensions can be created, see the extension.py file for full API.

```python
from scivianna.extension.extension import Extension

class MyCustomExtension(Extension):
    def __init__(self, slave, plotter, panel):
        super().__init__(slave, plotter, panel)
        # Initialize extension
        
    def make_panel(self):
        # Return Panel UI for this extension
        return pn.Column(...)
        
    def on_mouse_clic(self, screen_location, space_location, cell_id):
        # Handle mouse click events
        pass
        
    def on_file_load(self, file_path, file_label):
        # Handle file loading events
        pass
```

## Extension Lifecycle

1. **Creation**: Extensions are instantiated with slave, plotter, and panel references
2. **Panel Building**: `make_panel()` returns the UI components
3. **Event Registration**: Callbacks are registered for mouse events, file loads, etc.
4. **State Management**: `to_json()` and `from_json()` handle serialization

### Callback Signatures

| Callback | Parameters | Description |
|----------|-----------|-------------|
| `on_file_load(file_path, file_label)` | `str`, `str` | Called when a file is loaded |
| `on_field_change(field_name)` | `str` | Called when displayed field changes |
| `on_updated_data(data)` | `Data2D` / `Data3D` | Called when displayed data updates |
| `on_range_change(u_bounds, v_bounds, w_value)` | `tuple`, `tuple`, `float` | Called when viewport zoom/pan changes |
| `on_frame_change(u_vector, v_vector)` | `tuple`, `tuple` | Called when viewport orientation changes |
| `on_mouse_move(screen_location, space_location, cell_id)` | `tuple`, `tuple`, `str/int` | Called on mouse hover over plot |
| `on_mouse_clic(screen_location, space_location, cell_id)` | `tuple`, `tuple`, `str/int` | Called on mouse click on plot |
| `on_key_pressed(key)` | `str` | Called when a keyboard key is pressed while the mouse hovers the panel |
| `on_scale_change(axis_name, value)` | `str`, `str` | Called when the 1D plot scale is changed |

### Mouse Event Parameters

- **screen_location**: `(x, y)` - Mouse position in screen coordinates (pixels from top-left)
- **space_location**: `(x, y, z)` - Mouse position in 3D world coordinates
- **cell_id**: `str` or `int` - ID of the cell currently under the mouse / clicked
