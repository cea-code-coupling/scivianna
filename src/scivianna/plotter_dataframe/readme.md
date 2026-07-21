# Scivianna DataFrame Plotter Module

The `plotter_dataframe` module provides a simple plotter for displaying pandas DataFrames in Scivianna visualization panels. It wraps Panel's `DataFrame` pane to render tabular data with scrollable, sortable tables.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initialization file, exports `DataframePlotter` |
| `dataframe_plotter.py` | Implementation of `DataframePlotter` for displaying pandas DataFrames |

## DataframePlotter

The `DataframePlotter` class is a lightweight plotter that renders pandas DataFrames in an interactive table view. It is used by `PanelDataFrame` to display tabular simulation results.

### Key Methods

| Method | Purpose |
|--------|---------|
| `update_data(data)` | Replace the displayed DataFrame with new data |
| `get_data()` | Returns the currently displayed DataFrame |
| `make_panel()` | Returns the Panel viewable for embedding in the UI |
| `provide_on_mouse_move_callback(callback)` | Register a callback for mouse hover events (stored but no-op) |
| `provide_on_clic_callback(callback)` | Register a callback for click events (stored but no-op) |
| `enable_highlight(enable)` | Enable hover highlight (no-op for dataframe) |

### Usage

```python
from scivianna.plotter_dataframe import DataframePlotter
import pandas as pd

plotter = DataframePlotter()

# Display a DataFrame
df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
plotter.update_data(df)

# Get the Panel viewable for embedding
panel_view = plotter.make_panel()

# Retrieve current data
current_df = plotter.get_data()
```

### Integration with PanelDataFrame

The `DataframePlotter` is tightly coupled with `PanelDataFrame`. When a linked 2D panel triggers a `MOUSE_CELL_CHANGE` event, the `PanelDataFrame` calls `slave.get_dataframe()` and passes the result to `plotter.update_data()`.

### Event Handling

Mouse move and click callbacks are stored but do not trigger actions on the DataFrame itself. This is intentional — DataFrames are typically updated by linked geometry panels rather than direct interaction. The `enable_highlight()` method is a no-op since DataFrame rows do not support hover highlighting.
