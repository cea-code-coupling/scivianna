# Scivianna 1D Plotter Module

The `plotter_1d` module provides plotting backends for rendering one-dimensional data such as line charts, time series, and XY plots in Scivianna.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initialization file |
| `generic_plotter.py` | Base class `Plotter1D` defining the 1D plotting API |
| `bokeh_1d_plotter.py` | Bokeh-based implementation `BokehPlotter1D` for interactive 1D plots |

## Plotter1D (Base Class)

Defines the interface for all 1D plotters in `generic_plotter.py`:

### Key Methods

| Method | Purpose |
|--------|---------|
| `make_panel()` | Create and return the Panel display object |
| `plot(name, serie)` | Add a new plot with given name and pandas Series data |
| `update_plot(name, serie)` | Update existing plot or create if not exists |
| `set_visible(names)` | Set which plots are visible (list of names) |
| `set_x_scale(scale)` | Set X axis scale to "log" or "lin" |
| `set_y_scale(scale)` | Set Y axis scale to "log" or "lin" |

## BokehPlotter1D

Implementation using Bokeh for interactive 1D visualization in `bokeh_1d_plotter.py`:

### Features
- Interactive hover tools
- Multiple line overlays
- Zoom and pan controls
- Legend management
- Responsive sizing
- Color palette based on plot visibility order

### Supported Plot Types
- Line charts (single or multi-line)
- Time series with time navigation
- Variable comparisons

## Customization

Bokeh plot properties can be customized via the underlying Bokeh figure:

- Line colors and styles (automatically set based on visibility order)
- Axis ranges
- Grid visibility
- Toolbar configuration

## Additional Methods

| Method | Purpose |
|--------|---------|
| `get_x_bounds()` | Returns (min, max) of displayed data on X axis |
| `get_y_bounds()` | Returns (min, max) of displayed data on Y axis |
| `_disable_interactions(val)` | Enable/disable plot interactions |