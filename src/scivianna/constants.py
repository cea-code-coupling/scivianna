"""
Global constants for Scivianna.

This module centralizes all configuration constants, magic numbers, and default values
used throughout the Scivianna package. 

Constants are organized by category:
- Plot elements (XS, YS, CELL_NAMES, etc.)
- Default field names (MESH, MATERIAL)
- Default file names (GEOMETRY, CSV)
- Axis coordinates (X, Y, Z)
- Default values (DEFAULT_ORIGIN, DEFAULT_SIZE)
- Color constants (COLOR_SCALE_MAX, DEFAULT_EDGE_OFFSET)
- Timeout values (QUEUE_TIMEOUT_SHORT, QUEUE_TIMEOUT_LONG)
- Logging constants (DEFAULT_LOG_LEVEL)
"""

import numpy as np

# ============================================================================
# PLOT ELEMENTS - Names used in ColumnDataSource and plotting
# ============================================================================
XS = "xs"
"""Polygons X coordinates for 2D plotting."""

YS = "ys"
"""Polygons Y coordinates for 2D plotting."""

CELL_NAMES = "cell_names"
"""Cell identifiers."""

CELL_VALUES = "cell_values"
"""Cell field values."""

COLORS = "colors"
"""Cell fill colors."""

EDGE_COLORS = "edge_colors"
"""Cell edge/border colors."""

POLYGONS = "polygons"
"""Polygon geometry data."""

GRID = "grid"
"""Grid/rasterized geometry data."""

FILL_ALPHA = "fill_alpha"
"""Fill transparency value."""

EDGE_ALPHA = "edge_alpha"
"""Edge transparency value."""

# ============================================================================
# DEFAULT FIELD NAMES - Standard field labels
# ============================================================================
MESH = "Mesh"
"""Default field name for mesh visualization."""

MATERIAL = "Material"
"""Default field name for material visualization."""

# ============================================================================
# DEFAULT FILE NAMES - Standard file type labels
# ============================================================================
GEOMETRY = "Geometry"
"""Default label for geometry files."""

CSV = "CSV"
"""Default label for CSV files."""

# ============================================================================
# AXIS COORDINATES - Standard 3D direction vectors
# ============================================================================
X = (1.0, 0.0, 0.0)
"""Unit vector along X axis."""

Y = (0.0, 1.0, 0.0)
"""Unit vector along Y axis."""

Z = (0.0, 0.0, 1.0)
"""Unit vector along Z axis."""

# ============================================================================
# SPECIAL VALUES
# ============================================================================
OUTSIDE = np.inf
"""Cell name/value to define the outside world (infinity)."""

# ============================================================================
# DEFAULT VALUES - Panel initialization defaults
# ============================================================================
DEFAULT_ORIGIN = [0.01, 0.01, 0.01]
"""Default origin for 2D/3D panels when none is provided."""

DEFAULT_SIZE = 1.0
"""Default size for 2D panels when none is provided."""

DEFAULT_COLORMAP = "BuRd"
"""Default colormap name for field visualization."""

# ============================================================================
# COLOR CONSTANTS
# ============================================================================
COLOR_SCALE_MAX = 255
"""Maximum value for RGB/RGBA color components (0-255 range)."""

DEFAULT_EDGE_OFFSET = -20
"""Default offset for edge color darkening (negative value darkens)."""

DEFAULT_FILL_ALPHA = 1.0
"""Default fill opacity (1.0 = fully opaque)."""

DEFAULT_EDGE_ALPHA = 1.0
"""Default edge opacity (1.0 = fully opaque)."""

# ============================================================================
# TIMEOUT VALUES - Queue and async operation timeouts
# ============================================================================
QUEUE_TIMEOUT_SHORT = 0.01
"""Short timeout for queue operations in seconds (10ms)."""

QUEUE_TIMEOUT_MEDIUM = 0.1
"""Medium timeout for queue operations in seconds (100ms)."""

QUEUE_TIMEOUT_LONG = 1.0
"""Long timeout for queue operations in seconds."""

PROCESS_JOIN_TIMEOUT = 5.0
"""Timeout for process join operations in seconds."""

# ============================================================================
# LOGGING CONSTANTS
# ============================================================================
DEFAULT_LOG_LEVEL = "INFO"
"""Default logging level for Scivianna."""

# ============================================================================
# EXTENSION DEFAULTS
# ============================================================================
DEFAULT_ICON_SIZE = "1em"
"""Default icon size for extension UI elements."""

# ============================================================================
# NUMERICAL CONSTANTS
# ============================================================================
EPSILON = 1e-10
"""Small value for numerical comparisons (avoid division by zero)."""

DEFAULT_DPI = 100
"""Default DPI for figure rendering."""
