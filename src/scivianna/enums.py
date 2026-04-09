from enum import Enum


class VisualizationMode(Enum):
    """How is the color mapped on an input value."""

    FROM_VALUE = 0
    """ The color is got from a colormap based on a float value.
    """
    FROM_STRING = 1
    """ The color is got randomly based on the string value.
    """
    NONE = 2
    """ Only the mesh is displayed
    """


class PlotType(Enum):
    """Is the plot 1D or 2D."""

    _1D = 0
    """ 1D plot : expects some (x, y) line data
    """
    _2D = 1
    """ 2D plot : expects a set of polygons and their associated colors
    """


class GeometryType(Enum):
    """Is the geometry 2D or 3D."""

    _2D = 0
    """ 2D geometry : A selected window of the geometry is displayed 
    """
    _2D_INFINITE = 0
    """ 2D geometry : Everything is displayed 
    """
    _3D = 1
    """ 3D geometry : U and V axis and w_value are required to slice the geometry, only a selected window is displayed
    """
    _3D_INFINITE = 1
    """ 3D geometry : U and V axis and w_value are required to slice the geometry
    """

class DataType(Enum):
    """Are the data returned in a grid or polygons"""
    GRID = 0
    """Data are contained in a numpy array"""
    POLYGONS = 1
    """Data are contained in a list of polygons"""

class UpdateEvent(Enum):
    """What triggers a plot update"""

    RECOMPUTE = 0
    """ The plot is only updated by pressing the mouse button
    """
    CLIC = 1
    """ The plot is updated when the mouse is clics on a 2D plot, the applicable plots are updated requesting the mouse space location.
    """
    MOUSE_POSITION_CHANGE = 2
    """ The plot is updated when the mouse is moved over a 2D plot, the applicable plots are updated requesting the mouse space location.
    """
    MOUSE_CELL_CHANGE = 3
    """ The plot is updated when the mouse is moved over a 2D plot and enters a new cell.
    """
    PERIODIC = 4
    """ The plot is preiodically updated, applicable in real-time code coupling simulations
    """
    RANGE_CHANGE = 5
    """ The plot is updated when the (u, v) ranges change
    """
