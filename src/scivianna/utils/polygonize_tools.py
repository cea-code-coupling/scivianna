import math
from typing import Any, Dict, List, Tuple, Type, Union
import numpy as np

try:
    import rasterio
    import rasterio.features
    from rasterio.transform import Affine 
    rasterio_imported = True
except ImportError as e:
    rasterio_imported = False
    rasterio_error = e

import shapely
from shapely.geometry.polygon import Polygon
from shapely.geometry import shape

class PolygonCoords:
    """Object ontaining the X and Y coordinates of a polygon
    """
    def __init__(self, 
                    x_coords:Union[List[float], np.ndarray], 
                    y_coords:Union[List[float], np.ndarray]):
        """PolygonCoords object constructor.

        Parameters
        ----------
        x_coords : Union[List[float], np.ndarray]
            X coordinates of the polygon vertices
        y_coords : Union[List[float], np.ndarray]
            Y coordinates of the polygon vertices
        """
        if type(x_coords) not in (list, np.ndarray):
            raise TypeError(f"x_coords must be a numpy array or list, found {type(x_coords)}")
        if type(y_coords) not in (list, np.ndarray):
            raise TypeError(f"y_coords must be a numpy array or list, found {type(y_coords)}")
        
        if len(x_coords) != len(y_coords):
            raise ValueError(f"Given polygons coords must have the same length, found {len(x_coords)} and {len(y_coords)}")
        
        self.x_coords:np.ndarray = np.array(x_coords)
        """ X coordinate of each vertex of a polygon
        """
        self.y_coords:np.ndarray = np.array(y_coords)
        """ Y coordinate of each vertex of a polygon
        """
        
    def translate(self, dx:float, dy:float):
        """Translates the PolygonCoords by (dx, dy)

        Parameters
        ----------
        dx : float
            Horizontal offset
        dy : float
            Vertical offset
        """
        self.x_coords += dx
        self.y_coords += dy

    def rotate(self, origin:Tuple[float, float], angle:float):
        """Rotate the PolygonElement by the angle around the origin

        Parameters
        ----------
        origin : Tuple[float, float]
            Rotation origin
        angle : float
            Angle (in radians)
        """
        dx = self.x_coords - origin[0]
        dy = self.y_coords - origin[1]

        rotation_matrix = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])

        diff = np.array([dx, dy])

        new_d = np.matmul(diff.T, rotation_matrix)
        
        self.x_coords = new_d[:, 0]+origin[0]
        self.y_coords = new_d[:, 1]+origin[1]

class PolygonElement:
    """Object containing the exterior polygon and the holes of a polygonal object
    """
    def __init__(self, 
                    exterior_polygon:PolygonCoords, 
                    holes:List[PolygonCoords], 
                    cell_id:str):
        """PolygonCoords object constructor.

        Parameters
        ----------
        exterior_polygon : PolygonCoords
            Polygon that surrounds a polygonal object
        holes : List[PolygonCoords]
            List of polygonal holes in a polygon object
        cell_id : str
            cell, associated to the polygon, id
        """
        if not isinstance(exterior_polygon, PolygonCoords):
            raise TypeError(f"exterior_polygon expects a PolygonCoords object, found {type(exterior_polygon)}")
        if not isinstance(holes, list):
            raise TypeError(f"holes expects a list, found {type(holes)}")
        
        for i in range(len(holes)):
            if not isinstance(holes[i], PolygonCoords):
                raise TypeError(f"holes expects a list of PolygonCoords objects, found {type(holes[i])} at item {i}")
    
        self.exterior_polygon:PolygonCoords = exterior_polygon
        """ Polygon that surrounds a polygonal object
        """
        self.holes:List[PolygonCoords] = holes
        """ List of polygonal holes in a polygon object
        """
        self.cell_id:str = cell_id
        """ Cell, associated to the polygon, id
        """
        self.compo:str = ""
        """ Composition in the polygon
        """

    def translate(self, dx:float, dy:float):
        """Translates the PolygonElement by (dx, dy)

        Parameters
        ----------
        dx : float
            Horizontal offset
        dy : float
            Vertical offset
        """
        self.exterior_polygon.translate(dx, dy)
        for poly in self.holes:
            poly.translate(dx, dy)

    def rotate(self, origin:Tuple[float, float], angle:float):
        """Rotate the PolygonElement by the angle around the origin

        Parameters
        ----------
        origin : Tuple[float, float]
            Rotation origin
        angle : float
            Angle (in radians)
        """
        self.exterior_polygon.rotate(origin, angle)
        for poly in self.holes:
            poly.rotate(origin, angle)

    def to_shapely(self, z_coord: float = None) -> shapely.Polygon:
        """Returns a shapely Polygon version of self at the vertical coordinate z_coord

        Parameters
        ----------
        z_coord : float, optional
            If provided, the polygon is 3D at the given height, by default None

        Returns
        -------
        shapely.Polygon
            Shapely polygon
        """
        if z_coord is None:
            return shapely.Polygon(
                np.array([
                    self.exterior_polygon.x_coords, 
                    self.exterior_polygon.y_coords
                ]),
                [
                    np.array([
                        h.x_coords, 
                        h.y_coords
                    ]) for h in self.holes
                ]
            )
        else:
            return shapely.Polygon(
                np.array([
                    self.exterior_polygon.x_coords, 
                    self.exterior_polygon.y_coords,
                    [z_coord] * len(self.exterior_polygon.y_coords)
                ]).T,
                [
                    np.array([
                        h.x_coords, 
                        h.y_coords,
                        [z_coord] * len(h.y_coords)
                    ]).T for h in self.holes
                ]
            )

def numpy_2D_array_to_polygons(x:Union[List[float], np.ndarray], 
                                    y:Union[List[float], np.ndarray], 
                                    arr:np.ndarray, 
                                    simplify:bool) -> List[PolygonElement]:
    """Converts a 2D array mapping the cell id to a list of PolygonElements using the python module rasterio

    Parameters
    ----------
    x : Union[List[float], np.ndarray]
        Points coordinates along the X axis
    y : Union[List[float], np.ndarray]
        Points coordinates along the Y acis
    arr : np.ndarray
        2D cell index mapping
    simplify : bool
        Simplify the polygons to smoothen the edges

    Returns
    -------
    List[PolygonElement]
        List of PolygonElements
    """
    if not rasterio_imported:
        raise rasterio_error
    
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"arr must be a numpy array, found {type(arr)}") 
    if type(x) not in (list, np.ndarray):
        raise TypeError(f"x must be a numpy array or a list, found {type(x)}") 
    if type(y) not in (list, np.ndarray):
        raise TypeError(f"y must be a numpy array or a list, found {type(y)}") 
    if not len(x) == arr.shape[1]:
        raise ValueError(f"len(x) must have the same length as arr first coordinate, found {len(x)} and {arr.shape[1]}")
    if not len(y) == arr.shape[0]:
        raise ValueError(f"len(y) must have the same length as arr second coordinate, found {len(y)} and {arr.shape[0]}")


    x0 = min(x)
    x1 = max(x)
    y0 = min(y)
    y1 = max(y)

    # Simplify tolerance
    delta = math.sqrt(math.pow(x[1] - x[0], 2) + math.pow(y[1] - y[0], 2))

    polygon_element_list:List[PolygonElement] = []

    #   We build a new array, we list the string values, and replace them by their index to accept very large values and non floats
    values, direct, inv = np.unique(arr.flatten(), return_index = True, return_inverse=True)   #   Values found

    # values : found values
    # direct : first index of the value in arr.flatten()
    # inv : for each element in arr.flatten(), index in values of the element

    # values[inv].reshape(arr.shape) = arr

    index_arr = np.array(range(len(values)))[inv].reshape(arr.shape).astype(np.int32)

    transform1 = Affine.translation(x0 - (x1-x0)/len(x) / 2, y0 - (y1-y0)/len(y) / 2) * Affine.scale((x1-x0)/len(x), (y1-y0)/len(y))
    shape_gen = ((shape(s), val) for s, val in rasterio.features.shapes(index_arr, transform=transform1))

    s:Polygon
    for s, val in shape_gen:
        #   Checking the polygons of value 1
        if simplify:
            s = s.simplify(delta)
        polygon_element_list.append(
                        PolygonElement(exterior_polygon=PolygonCoords(x_coords=np.array([vert[0] for vert in s.exterior.coords]), 
                                                                    y_coords=np.array([vert[1] for vert in s.exterior.coords])),
                                        holes=[PolygonCoords(x_coords=np.array([vert[0] for vert in interior.coords]),
                                                             y_coords=np.array([vert[1] for vert in interior.coords])) 
                                                             for interior in s.interiors],
                                        cell_id=values[int(val)])
                    )
        
    return polygon_element_list


if __name__ == "__main__":
    x_vals = np.arange(10)
    y_vals = np.arange(10)

    z_vals = x_vals*np.expand_dims(y_vals, axis=0).T

    numpy_2D_array_to_polygons(x_vals, y_vals, z_vals, False)