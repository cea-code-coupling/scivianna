# Scivianna 3D Plotter Module

The `plotter_3d` module provides plotting backends for rendering three-dimensional geometry and field data in Scivianna. It is built on top of `scivianna_vtk` for interactive 3D visualization with clip plane slicing.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initialization file |
| `generic_plotter.py` | Base class `Plotter3D` defining the 3D plotting API |
| `vtk_3d_plotter.py` | VTK-based implementation `Plotter3D` for interactive 3D plots |

## Plotter3D (Base Class)

Defines the interface for all 3D plotters in `generic_plotter.py`:

### Key Methods

| Method | Purpose |
|--------|---------|
| `plot(data)` | Add a new 3D plot from a `Data3D` object |
| `update_plot(data)` | Update the existing plot with new field values |
| `make_panel()` | Create and return the Panel display object |
| `update_colorbar(display, value_range)` | Show/hide color bar and set its range |
| `set_color_map(color_map_name)` | Set the colormap for field visualization |
| `provide_on_mouse_move_callback(fn)` | Register mouse hover handler |
| `provide_on_clic_callback(fn)` | Register click handler |
| `provide_on_axes_change_callback(fn)` | Register clip plane / axes change handler |
| `move_slice_to(u, v, origin)` | Move the clip plane to a new position/orientation |
| `get_slice_normal()` | Returns the normal vector of the current slice plane |
| `set_slice_origin(origin)` | Sets the origin of the slice plane |
| `get_slice_origin()` | Returns the origin of the current slice plane |
| `get_uv()` | Returns the u and v axes of the slice plane as `(u, v)` |
| `get_mouse_location()` | Returns current mouse position in 3D world coordinates as `(x, y, z)` |

## VTK-Based Implementation

The `vtk_3d_plotter.py` implementation wraps `scivianna_vtk.plotter.VTKPlotter` and provides:

### Features
- Interactive 3D rendering with mouse rotation, pan, and zoom
- Clip plane slicing for viewing interior geometry
- Hover detection returning cell ID and 3D position
- Click event propagation
- Automatic computation of u/v axes from clip plane normal
- Callback dispatch for mouse move, click, and axes change events

### Slice Plane Management

The 3D plotter maintains a slice (clip) plane that can be moved programmatically or interactively:

```python
# Move the clip plane to a specific position
plotter.move_slice_to(u=u_vector, v=v_vector, origin=position)

# Query current slice state
normal = plotter.get_slice_normal()
origin = plotter.get_slice_origin()
u, v = plotter.get_uv()
```

### Events

The VTK plotter dispatches three types of events to registered callbacks:

| Event | Callback Params | Description |
|-------|-----------------|-------------|
| Mouse move | `(screen_location, space_location, cell_id)` | Mouse hovers over geometry |
| Click | `(screen_location, space_location, cell_id)` | User clicks on geometry |
| Axes change | `(u, v, origin, size_u, size_v)` | Clip plane is moved or rotated |

## Optional Dependency

This module depends on `scivianna_vtk` (which in turn requires `pyvista`). When the dependency is not available, a stub class is used as a fallback.

Install with:

```bash
pip install scivianna[3d]
```
