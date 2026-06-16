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
from scivianna.layout.splot import SplitLayout, SplitItem, SplitDirection

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
Panels can be configured to update on different events:
- `UpdateEvent.RECOMPUTE`: Manual recompute trigger
- `UpdateEvent.CLIC`: Update on click
- `UpdateEvent.MOUSE_POSITION_CHANGE`: Update on mouse move
- `UpdateEvent.MOUSE_CELL_CHANGE`: Update when hovered cell changes

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