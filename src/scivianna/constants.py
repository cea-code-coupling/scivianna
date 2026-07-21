import numpy as np

#   Plots elements names
XS = "xs"
YS = "ys"
CELL_NAMES = "cell_names"
CELL_VALUES = "cell_values"
COLORS = "colors"
EDGE_COLORS = "edge_colors"
POLYGONS = "polygons"
GRID = "grid"
FILL_ALPHA = "fill_alpha"
EDGE_ALPHA = "edge_alpha"

#   Default field names
MESH = "Mesh"
MATERIAL = "Material"

#   Default file names
GEOMETRY = "Geometry"
CSV = "CSV"

# Axis coordinates
X = (1., 0., 0.)
Y = (0., 1., 0.)
Z = (0., 0., 1.)

# Cell name to define the outside world
OUTSIDE = np.inf

#   Default values
DEFAULT_ORIGIN = [0.01, 0.01, 0.01]
"""Default origin for 2D/3D panels when none is provided."""

DEFAULT_SIZE = 1.0
"""Default size for 2D panels when none is provided."""

#   Color constants
COLOR_SCALE_MAX = 255
"""Maximum value for RGB color components (0-255)."""

DEFAULT_EDGE_OFFSET = -20
"""Default offset for edge color darkening."""
