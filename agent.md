# Scivianna - Developer Onboarding Guide

## Project Overview

**Scivianna** is a modular 2D visualization framework for scientific/computational data, built on Panel/Holoviz and Bokeh. It provides an interactive web-based GUI for exploring 2D slices of meshes/fields from various simulation codes (e.g., MED/Cathare files).

The key design principle: **a decoupled architecture where a multiprocessing worker handles code-specific I/O**, while the main process handles UI rendering. This avoids GIL issues and keeps heavy data (meshes, fields) in the worker process.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| UI Framework | [Panel](https://panel.holoviz.org/) (Holoviz) + [Panel Material UI](https://panel-materialui.holoviz.org/) |
| Plotting | Bokeh (via HoloViews overlay) |
| Data Processing | MEDCoupling, NumPy, Pandas |
| IPC | Python `multiprocessing` with `Queue` |
| Serialization | `dill` for cross-process transfer, `pickle` for save/load |

---

## Directory Structure

```
scivianna/
├── src/scivianna/
│   ├── __init__.py              # Package entry, version
│   ├── constants.py             # Global string constants (MESH, GEOMETRY, CSV, etc.)
│   ├── enums.py                 # Enumeration types (GeometryType, VisualizationMode, UpdateEvent)
│   ├── slave.py                 # ComputeSlave + worker process
│   │
│   ├── interface/
│   │   ├── generic_interface.py  # Abstract base classes for all code interfaces
│   │   ├── med_interface.py      # MEDCoupling (.med file) implementation
│   │   └── ...                   # Additional interfaces (e.g., CSV, TriToPO5)
│   │
│   ├── data/
│   │   ├── data2d.py             # Data2D: geometry + cell values for 2D plotting
│   │   ├── data_container.py     # Container for field metadata
│   │   └── ...
│   │
│   ├── plotter_2d/
│   │   ├── generic_plotter.py    # Abstract Plotter2D interface
│   │   ├── polygon/
│   │   │   └── bokeh.py          # Bokeh-based 2D polygon renderer
│   │   └── ...
│   │
│   ├── panel/
│   │   ├── visualisation_panel.py  # Abstract VisualizationPanel base class
│   │   ├── panel_2d.py             # 2D panel implementation (recompute, layout)
│   │   └── gui.py                  # GUI builder for extensions
│   │
│   ├── extension/
│   │   ├── extension.py            # Base Extension class with lifecycle hooks
│   │   ├── save_load_extension.py  # Save/load session state extension
│   │   └── ...
│   │
│   ├── component/
│   │   └── overlay_component.py    # HoloViews Overlay wrapper for figure + buttons
│   │
│   ├── utils/
│   │   ├── polygonize_tools.py     # PolygonElement, PolygonCoords dataclasses
│   │   └── ...
│   │
│   └── icon/                      # SVG icons (settings_applications.svg, salome.svg, etc.)
│
├── tests/                         # Unit and integration tests
├── pyproject.toml                 # Project metadata, dependencies
└── readme.md                      # User-facing documentation
```

---

## Core Architecture

### 1. Interface Hierarchy (Code Adapters)

All code-specific adapters inherit from abstract base classes in `interface/generic_interface.py`:

```
GenericInterface          # File I/O: read_file, get_labels, save/load
    │
    ├── Geometry2D        # + compute_2d_data, get_value_dict, get_geometry_type
    │
    ├── ValueAtLocation   # + get_value, get_values (result query at cell/position)
    │
    ├── Value1DAtLocation # + get_1D_value (tabular 1D results)
    │
    └── OverLine          # + compute_1d_line_data (line extraction)


IcocoInterface            # Input coupling: getInputMEDDoubleFieldTemplate, setInputMEDDoubleField, setTime
```

**Key abstract methods every interface must implement:**

| Base Class | Method | Purpose |
|------------|--------|---------|
| `GenericInterface` | `read_file(file_path, file_label)` | Load a simulation file |
| `GenericInterface` | `get_labels()` | Return list of displayable field names |
| `GenericInterface` | `get_file_input_list()` | Return [(label, description)] for file picker |
| `GenericInterface` | `save(file_path, include_files)` | Pickle save state |
| `GenericInterface` | `load(file_path, include_files)` | Load pickled state |
| `Geometry2D` | `compute_2D_data(u, v, u_min, u_max, v_min, v_max, w_value, q_tasks, options)` | Compute 2D slice geometry + polygons |
| `Geometry2D` | `get_value_dict(value_label, cells, options)` | Return {cell_id: value} for a field |
| `Geometry2D` | `get_geometry_type()` | Return `GeometryType` enum |

**Concrete example:** `MEDInterface` (in `med_interface.py`) implements both `Geometry2DPolygon` and `IcocoInterface`. It reads `.med` files using the MEDCoupling library, builds 2D slices via `buildSlice3D`, and maps field values to polygon cells.

### 2. ComputeSlave + Worker Process (Multiprocessing IPC)

The `ComputeSlave` class (`slave.py`) manages a **background subprocess** that runs a `worker()` function. This isolates heavy data (meshes, fields) from the main UI process.

```
Main Process                          Worker Process
┌──────────────┐                     ┌──────────────────┐
│  ComputeSlave │  ──send task──►    │  code_interface()│
│  (UI thread)  │  ◄──return result │  (worker func)   │
└──────────────┘                     └──────────────────┘
       │                                      │
       ├─ q_tasks (Queue)   ─────────►        │
       ├─ q_returns (Queue) ◄─────────        │
       └─ q_errors  (Queue) ◄─────────        │
```

**How it works:**

1. **Initialization:** `ComputeSlave.__init__()` spawns a `multiprocessing.Process` running `worker()`.
2. **Task submission:** `slave.read_file()`, `slave.get_labels()`, `slave.compute_2D_data()`, etc. push tuples `(SlaveCommand, args)` to `q_tasks`.
3. **Worker loop:** The `worker()` function pops tasks from `q_tasks`, dispatches by `SlaveCommand` enum to the appropriate method on the `code_interface` instance.
4. **Result retrieval:** Results are sent back via `q_returns`. Errors go via `q_errors`.
5. **Serialization:** Arguments are serialized with `dill` (to handle unpicklable objects). Results are returned through the queue.

**SlaveCommand enum values:** `READ_FILE`, `GET_LABELS`, `GET_LABEL_COLORING_MODE`, `COMPUTE_2D_DATA`, `GET_VALUE_DICT`, `GET_GEOMETRY_TYPE`, `GET_VALUE`, `GET_VALUES`, `GET_1D_VALUE`, `COMPUTE_1D_LINE_DATA`, `SET_TIME`, `SET_INPUT_MED_DOUBLEFIELD`, `CUSTOM`, etc.

**Key pattern for adding a new command:**
1. Add a constant to `SlaveCommand` class in `slave.py`
2. Add handler in the `worker()` function's if/elif chain
3. Add a wrapper method on `ComputeSlave` to push the task

### 3. Extension System

Extensions are modular plugins that customize panel behavior. They hook into lifecycle events:

```python
class Extension:
    def on_file_load(self, file_path, file_key): ...
    def on_field_change(self, field_name): ...
    def on_updated_data(self, data: Data2D): ...
    def on_range_change(self, u_bounds, v_bounds, w_value): ...
    def on_frame_change(self, u_vector, v_vector): ...
    def on_mouse_move(self, screen_loc, space_loc, cell_id): ...
    def on_mouse_clic(self, screen_loc, space_loc, cell_id): ...
    def provide_options(self) -> Dict[str, Any]: ...
    def make_gui(self) -> pn.viewable.Viewable: ...
    def to_json(self) -> dict: ...
    @classmethod
    def from_json(cls, extension, info_dict): ...
```

**Extension lifecycle:**
1. Created during `VisualizationPanel.__init__()` with references to `slave`, `plotter`, and `panel`
2. `make_gui()` is called to build the sidebar UI tab
3. Lifecycle callbacks fire in response to user actions (file load, field change, mouse events)
4. `provide_options()` is called before each `compute_2D_data` call to pass custom options to the interface

**Example:** `MEDCouplingExtension` adds iteration/order selectors and a W-coordinate slider for MED field display parameters.

### 4. Plotter2D (2D Rendering)

The `Plotter2D` abstract class (`plotter_2d/generic_plotter.py`) defines the rendering interface:

```python
class Plotter2D:
    def display_borders(self, display: bool): ...
    def update_colorbar(self, display: bool, range: Tuple[float, float]): ...
    def set_color_map(self, color_map_name: str): ...
    def plot_2d_frame(self, data: Data2D): ...      # Initial render
    def update_2d_frame(self, data: Data2D): ...     # Full frame update
    def update_colors(self, data: Data2D): ...       # Color-only update (efficient)
    def make_panel(self) -> pn.viewable.Viewable: ...  # Returns Bokeh/Panel widget
    def provide_on_mouse_move_callback(self, callback): ...
    def provide_on_clic_callback(self, callback): ...
    def set_axes(self, u, v, w): ...
    def enable_highlight(self, enable: bool = True): ...
```

**Implementation:** `BokehPlotter` (`plotter_2d/polygon/bokeh.py`) uses Bokeh's `QuadContourFormatter` or `Patch` glyphs to render polygon geometries with color mapping. It manages a `ColumnDataSource` for efficient data updates.

### 5. VisualizationPanel (UI Container)

The `VisualizationPanel` (`panel/visualisation_panel.py`) is the main UI component:

```
┌─────────────────────────────────────────────┐
│  VisualizationPanel                          │
│  ┌──────────┐  ┌──────────────────────────┐ │
│  │ GUI Panel│  │ Figure (Overlay)         │ │
│  │ (Tabs)   │  │ ┌──────────────────────┐ │ │
│  │          │  │ │ Plotter2D panel      │ │ │
│  │ Extension│  │ │                      │ │ │
│  │ tabs...  │  │ │ 2D polygon plot      │ │ │
│  └──────────┘  │ └──────────────────────┘ │ │
│                └──────────────────────────┘ │
└─────────────────────────────────────────────┘
```

Key methods: `recompute()`, `set_field()`, `set_colormap()`, `outline_color()`, `provide_on_clic_callback()`, `to_json()`/`from_json()` for serialization.

### 6. Data Classes

**Data2D** (`data/data2d.py`): Holds geometry + field values for plotting:
- `cell_ids`: List of cell identifiers (strings)
- `cell_values`: List of numeric/string values per cell
- `polygons`: List of `PolygonElement` objects (exterior + holes with x/y coords)

**PolygonElement** (`utils/polygonize_tools.py`): Represents a single polygon:
```python
@dataclass
class PolygonCoords:
    x_coords: list[float]
    y_coords: list[float]

@dataclass  
class PolygonElement:
    exterior_polygon: PolygonCoords
    holes: List[PolygonCoords]
    cell_id: str
```

---

## Key Design Patterns

### Pattern 1: Abstract Base + Concrete Implementation

All major components follow an interface-segregation pattern:
- `GenericInterface` → `MEDInterface`, `CSVInterface`, etc.
- `Plotter2D` → `BokehPlotter`
- `VisualizationPanel` → `Panel2D`

This allows swapping implementations independently (e.g., new file format, new renderer).

### Pattern 2: Extension Hook System

Extensions provide plug-in points without modifying core code. Each lifecycle method is a no-op in the base class and can be overridden. The panel iterates over all extensions to dispatch events.

### Pattern 3: Multiprocessing Isolation

Heavy data lives in the worker process. The main process only receives serialized results (polygon coordinates, field values). This prevents memory bloat in the UI process and avoids MEDCoupling/GIL issues.

### Pattern 4: Command Dispatch

The `SlaveCommand` enum acts as a RPC-like dispatch mechanism. Each command maps to a specific method on the interface. New capabilities require adding a command constant + worker handler + slave wrapper.

---

## Adding a New Code Interface

1. **Define the interface class** in `src/scivianna/interface/new_interface.py`:
```python
from scivianna.interface.generic_interface import Geometry2D
from scivianna.data.data2d import Data2D
from scivianna.enums import GeometryType, VisualizationMode

class NewInterface(Geometry2D):
    geometry_type = GeometryType._2D_INFINITE
    
    def read_file(self, file_path: str, file_label: str):
        # Parse and store the file
        
    def compute_2D_data(self, u, v, u_min, u_max, v_min, v_max, w_value, q_tasks, options) -> Tuple[Data2D, bool]:
        # Return (Data2D, polygons_updated)
        
    def get_labels(self) -> List[str]:
        # Return list of displayable field names
        
    def get_value_dict(self, value_label, cells, options) -> Dict:
        # Return {cell_id: value} mapping
        
    def get_label_coloring_mode(self, label: str) -> VisualizationMode:
        return VisualizationMode.FROM_VALUE
        
    def get_file_input_list(self) -> List[Tuple[str, str]]:
        return [("NEW", "New file type description")]
        
    # Implement save/load if needed
```

2. **Register the interface** in the main application entry point.

3. **Test** with `pytest scivianna/tests/`.

---

## Adding a New Extension

1. **Create extension class** inheriting from `Extension`:
```python
from scivianna.extension.extension import Extension

class MyExtension(Extension):
    def __init__(self, slave, plotter, panel):
        super().__init__("My Extension", icon_svg, slave, plotter, panel)
        # Initialize UI components
        
    def make_gui(self):
        return pn.Row(...)  # Panel viewable for sidebar tab
        
    def provide_options(self):
        return {"my_option": self.some_value}
        
    def on_field_change(self, field_name):
        # React to field change
        pass
```

2. **Pass to panel** during creation:
```python
panel = Panel2D(slave, name="My Panel", extensions=[MyExtension])
```

---

## Testing

Tests live in `scivianna/tests/`:
- `tests/unit_test/` - Unit tests for individual components
- `tests/integration_test/` - Integration tests (e.g., `test_serialization.py`)

Run tests:
```bash
cd scivianna
pytest tests/
```

---

## Development Workflow

1. **Install dependencies:**
```bash
cd scivianna
pip install -e .
```

2. **Run a quick demo** (if available):
```python
from scivianna.slave import ComputeSlave
from scivianna.interface.med_interface import MEDInterface
from scivianna.panel.panel_2d import Panel2D

slave = ComputeSlave(MEDInterface)
slave.read_file("path/to/file.med", "geometry")
panel = Panel2D(slave, name="Demo")
panel.show()  # or use _show_panel from scivianna.notebook_tools
```

3. **Profile** (optional): Set `VIZ_PROFILE=1` environment variable to enable timing logs in interfaces.

---

## Common Pitfalls

1. **Unpicklable objects:** When sending data to the worker via `dill`, ensure all arguments are serializable. Redefine `serialize()` in the interface if needed.
2. **GIL awareness:** Never pass large numpy arrays or MEDCoupling meshes back to the main process unnecessarily. Only transfer what's needed for display.
3. **Queue blocking:** The worker processes tasks sequentially. Long-running `compute_2D_data` calls block subsequent requests. Use `allow_errors=True` in `ComputeSlave` to prevent UI freeze on errors.
4. **Extension feedback loops:** Set `_restoring = True` during `from_json()` restoration to prevent callback triggers.

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/scivianna/slave.py` | Multiprocessing worker + ComputeSlave |
| `src/scivianna/interface/generic_interface.py` | Abstract interface base classes |
| `src/scivianna/interface/med_interface.py` | MEDCoupling .med file adapter |
| `src/scivianna/panel/panel_2d.py` | 2D visualization panel implementation |
| `src/scivianna/panel/visualisation_panel.py` | Abstract panel base class |
| `src/scivianna/extension/extension.py` | Extension base class with hooks |
| `src/scivianna/plotter_2d/generic_plotter.py` | Plotter2D abstract interface |
| `src/scivianna/plotter_2d/polygon/bokeh.py` | Bokeh polygon renderer |
| `src/scivianna/data/data2d.py` | Data2D container for geometry+values |
| `src/scivianna/enums.py` | GeometryType, VisualizationMode, UpdateEvent |
| `src/scivianna/constants.py` | Global string constants |