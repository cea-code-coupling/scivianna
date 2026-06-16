# Scivianna 2D Plotter Module

The `plotter_2d` module provides plotting backends for rendering two-dimensional geometries and field data in Scivianna. It supports both grid-based and polygon-based visualizations.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initialization file |
| `api.py` | Public API exports for 2D plotting using matplotlib |
| `generic_plotter.py` | Base class `GenericPlotter2D` defining the 2D plotting API |
| `grid/` | Grid-based (raster) visualization backend |
| `polygon/` | Polygon-based vector visualization backend |

## Submodules

### Grid Backend (`grid/`)
Renders 2D data as colored grid cells:
- `bokeh.py`: Bokeh-based interactive grid renderer
- `matplotlib.py`: Matplotlib static grid renderer
- `grid_tools.py`: Utility functions for grid manipulation

### Polygon Backend (`polygon/`)
Renders 2D data as vector polygons:
- `bokeh.py`: Bokeh-based interactive polygon renderer
- `matplotlib.py`: Matplotlib static polygon renderer

## GenericPlotter2D (Base Class)

Defines the interface for all 2D plotters:

### Key Methods

| Method | Purpose |
|--------|---------|
| `make_panel()` | Create and return the Panel display object |
| `update_geometry(cells, values)` | Update plot with new geometry and field data |
| `set_colormap(colormap)` | Set the color map for field visualization |
| `set_clim(vmin, vmax)` | Set color scale limits |
| `provide_on_mouse_move_callback(fn)` | Register mouse hover handler |
| `provide_on_clic_callback(fn)` | Register click handler |

## Grid vs Polygon Rendering

### Grid Renderer
Best for:
- Regular grids
- Large or complex datasets where polygons are hard to compute

Features:
- Fast rendering for large datasets
- Pixel-based display
- Countour display

### Polygon Renderer
Best for:
- Unstructured meshes
- Irregular cell geometries
- When cell boundaries must be visible

Features:
- Vector-based precise cell shapes
- Cell boundary outlines
- Hover/click per-cell detection

## Interactive Features

Bokeh plotters can send back events when the mouse is moved or clicked on the geometry. Callback functions can be registered using the functions:

```python
def provide_on_mouse_move_callback(self, callback: Callable):
def provide_on_clic_callback(self, callback: Callable):
```