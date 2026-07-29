"""
Enumeration types for Scivianna.

This module defines all enumeration types used throughout Scivianna for
type-safe configuration and state management.

Enums are organized by category:
- VisualizationMode: How colors are mapped to data
- GeometryType: Dimensionality of geometry (2D or 3D)
- DataType: Data representation format (grid or polygons)
- UpdateEvent: Triggers for plot updates
- UpdatePolicy: Data update strategies for coupling
"""

from enum import Enum, auto


class VisualizationMode(Enum):
    """
    Defines how colors are mapped to data in visualizations.

    This enum determines whether cell colors are derived from numeric values
    (using a colormap), string values (using random assignment), or not used
    at all (mesh-only display).

    Attributes
    ----------
    FROM_VALUE : int
        Color is determined by a colormap based on float values.
        Use this for continuous fields (temperature, pressure, etc.)
    FROM_STRING : int
        Color is randomly assigned based on unique string values.
        Use this for categorical fields (material names, regions, etc.)
    NONE : int
        No coloring is applied; only mesh borders are displayed.
        Use this for geometry-only visualization.
    """

    FROM_VALUE = 0
    """Color from colormap based on float value."""

    FROM_STRING = 1
    """Color randomly assigned based on string value."""

    NONE = 2
    """Only mesh displayed, no coloring."""


class GeometryType(Enum):
    """
    Defines the dimensionality and display mode of geometry.

    This enum specifies whether the geometry is 2D or 3D, and whether
    the display shows a selected window or the entire geometry.

    Attributes
    ----------
    _2D : int
        2D geometry with selected window display (zoomable).
    _2D_INFINITE : int
        2D geometry with full display (no zoom, everything visible).
    _3D : int
        3D geometry requiring U, V axes and w_value for slicing,
        with selected window display.
    _3D_INFINITE : int
        3D geometry requiring U, V axes and w_value for slicing,
        with full display.
    """

    _2D = 0
    """2D geometry: selected window display (zoomable)."""

    _2D_INFINITE = 1
    """2D geometry: full display (no zoom)."""

    _3D = 2
    """3D geometry: requires slicing, selected window display."""

    _3D_INFINITE = 3
    """3D geometry: requires slicing, full display."""


class DataType(Enum):
    """
    Defines the data representation format.

    This enum specifies whether spatial data is stored as a regular grid
    (numpy array) or as an irregular polygon list. This affects which
    plotter is used and how data is processed.

    Attributes
    ----------
    GRID : int
        Data stored in a numpy array (rasterized representation).
        Efficient for regular grids, requires rasterization for polygons.
    POLYGONS : int
        Data stored as a list of PolygonElement objects.
        Suitable for irregular meshes, preserves exact geometry.
    """

    GRID = 0
    """Data in numpy array (rasterized)."""

    POLYGONS = 1
    """Data as list of polygons."""


class UpdateEvent(int, Enum):
    """
    Defines what triggers a plot update.

    This enum specifies the events that cause visualization panels to
    recompute and redraw. Different panels can be configured to respond
    to different events, enabling sophisticated inter-panel interactions.

    Attributes
    ----------
    RECOMPUTE : int
        Manual update only (user presses recompute button).
        Most efficient for expensive computations.
    CLIC : int
        Update when mouse clicks on a linked 2D/3D plot.
        Updates use the click location and cell ID.
    MOUSE_POSITION_CHANGE : int
        Update when mouse moves over a linked 2D/3D plot.
        Updates use the mouse location and cell ID.
        Can be expensive if triggered frequently.
    MOUSE_CELL_CHANGE : int
        Update when mouse enters a new cell.
        More efficient than MOUSE_POSITION_CHANGE.
    PERIODIC : int
        Update at regular time intervals.
        Used for real-time coupling with simulations.
    RANGE_CHANGE : int
        Update when (u, v) ranges or origin change (zoom/pan).
        Standard for interactive exploration.
    AXES_CHANGE : int
        Update when axes (u, v vectors) or origin change.
        Used when changing slice orientation.

    Note
    ----
    Multiple events can be specified as a list to trigger updates on
    any of the events.
    """

    RECOMPUTE = 0
    """Manual update via recompute button."""

    CLIC = 1
    """Update on mouse click (sends location and cell ID)."""

    MOUSE_POSITION_CHANGE = 2
    """Update on mouse movement (sends location and cell ID)."""

    MOUSE_CELL_CHANGE = 3
    """Update when hovered cell changes."""

    PERIODIC = 4
    """Periodic update for real-time coupling."""

    RANGE_CHANGE = 5
    """Update on zoom/pan (range change)."""

    AXES_CHANGE = 6
    """Update on axes/orientation change."""


class UpdatePolicy(Enum):
    """
    Defines how code interfaces manage field updates during coupling.

    This enum controls whether data is appended (time history) or updated
    (replaced) during real-time simulation coupling. It also distinguishes
    between mesh updates and data-only updates.

    Attributes
    ----------
    APPEND_DATA : auto
        Keep mesh constant, append data with time stamps.
        Use when mesh is static but data evolves (e.g., temperature field).
    UPDATE_DATA : auto
        Keep mesh constant, replace data at each time step.
        Use when only current state matters (e.g., steady-state).
    APPEND_MESH : auto
        Append both mesh and data with time stamps.
        Use when mesh deforms over time (e.g., structural mechanics).
        Falls back to APPEND_DATA if mesh doesn't change.
    UPDATE_MESH : auto
        Replace both mesh and data at each time step.
        Use when only current deformed state matters.
        Falls back to UPDATE_DATA if mesh doesn't change.
    """

    APPEND_DATA = auto()
    """Append data to time history, mesh constant."""

    UPDATE_DATA = auto()
    """Replace data, mesh constant."""

    APPEND_MESH = auto()
    """Append mesh and data to time history."""

    UPDATE_MESH = auto()
    """Replace mesh and data."""
