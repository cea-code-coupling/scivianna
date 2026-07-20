# Scivianna Layout Module

The `layout` module provides layout managers for arranging multiple visualization panels in flexible, interactive configurations.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initialization file |
| `generic_layout.py` | Base class `GenericLayout` with common layout functionality |
| `gridstack.py` | GridStack-based layout manager for draggable, resizable panels |
| `split.py` | Split-pane layout manager for dividing panels horizontally/vertically |

## Layout Managers

### GenericLayout (Base Class)
Provides core functionality for all layout types:
- Panel registration and management
- Frame switching between active panels
- Mouse interaction callbacks (click, hover)
- Field synchronization across panels
- Time widget integration for coupling

### GridStack Layout
Implements a grid-based layout system:
- Panels can be dragged to new positions
- Panels can be resized dynamically
- Supports multiple panels in a single view
- Serializable layout state

### Split Layout
Provides binary split divisions:
- Split horizontally or vertically
- Draggable divider between sections
- Recursive splitting for complex layouts

## Key Features

### Panel Management
```python
from scivianna.layout.split import SplitLayout, SplitItem, SplitDirection

# Create layout with multiple panels
split_panel = SplitLayout(
    SplitItem(
        map_panel,
        line_panel,
        SplitDirection.VERTICAL
    ),
    additional_interfaces={"EuropeGrid": EuropeGridInterface, "TimeSeries": CountryTimeSeriesInterface},
)

```

### Frame Switching
Navigate between panels using the frame selector:
```python
layout.set_to_frame("panel1")  # Switch to panel1
layout.current_frame  # Get active panel name
```

### Mouse Interaction Callbacks
The layout handles and distributes mouse events:

| Callback | Trigger |
|----------|---------|
| `on_clic_callback` | User clicks on a plot |
| `mouse_move_callback` | User moves mouse over a plot |
| `field_change_callback` | User changes displayed field |

### Update Events

Panels can be configured to update on different events via `UpdateEvent`:

| Event | Value | Description |
|-------|-------|-------------|
| `UpdateEvent.RECOMPUTE` | 0 | Manual recompute only (default) |
| `UpdateEvent.CLIC` | 1 | Update on mouse click (sends/receives mouse location and cell ID) |
| `UpdateEvent.MOUSE_POSITION_CHANGE` | 2 | Update on mouse move over 2D plot |
| `UpdateEvent.MOUSE_CELL_CHANGE` | 3 | Update when hovered cell changes on 2D plot |
| `UpdateEvent.PERIODIC` | 4 | Periodic updates for real-time simulation coupling |
| `UpdateEvent.RANGE_CHANGE` | 5 | Update when zoom/pan bounds change (2D) or clip plane origin changes (3D) |
| `UpdateEvent.AXES_CHANGE` | 6 | Update when axes orientation or origin changes (2D axes or 3D clip plane) |

### Time Widget Integration
For coupled simulations, add time management:
```python
layout.add_time_widget()  # Adds play/pause and time navigation
```

## Synchronization

Panels can be synchronized to update together:
```python
# Enable field sync - changing field in one panel updates others
panel.sync_field = True
```

## Periodic Updates

For real-time coupling, enable periodic refresh (updates only if changes were done using the Icoco coupling interface):
```python
layout.add_periodic_update()  # Start 100ms refresh loop
layout.stop_periodic_update()  # Stop refresh
```

## Inter-Panel Communication

When panels share a layout and have compatible `update_event` settings, they can communicate:

1. **2D → 2D**: Clicking in one panel moves the slice origin in others
2. **2D → 3D**: Clicking in a 2D panel moves the 3D clip plane
3. **3D → 2D**: Rotating/panning the 3D view updates 2D slice axes
4. **3D → 3D**: Clip plane changes propagate between 3D panels
5. **2D → 1D**: Clicking or moving in one panel changes the plot location/cell id data
6. **3D → 1D**: Clicking or moving in one panel changes the plot location/cell id data
