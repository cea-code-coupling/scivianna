# Scivianna Component Module

The `component` module provides custom UI components for building interactive Scivianna dashboards using Panel.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initialization file |
| `gridstack_component.py` | GridStack layout component for resizable, draggable panels |
| `overlay_component.py` | Overlay component for layering UI elements over plots |
| `server_file_browser.py` | File browser component for server-side file navigation |
| `splitjs_component.py` | Split-pane component using Split.js for resizable divisions |
| `dist/` | Compiled JavaScript bundles for custom components |

## Components Overview

### GridStack Component
Implements a grid-based layout system where panels can be:
- Resized dynamically
- Dragged to new positions
- Added or removed interactively

### Overlay Component
Provides a container for overlaying buttons, titles, and controls on top of visualization plots.

### SplitJS Component
Creates split-pane layouts with draggable dividers for flexible horizontal or vertical divisions.

### Server File Browser
Enables browsing and loading files from the server filesystem directly within the Panel application.
