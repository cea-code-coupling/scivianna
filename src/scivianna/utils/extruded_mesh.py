import time
from typing import Any, Dict, List, Tuple
import numpy as np
from scivianna.interface.generic_interface import Geometry2D
from scivianna.utils.color_tools import get_edges_colors
import shapely
import shapely.coords

try:
    import pyvista as pv
    from pyvista import core
except ImportError:
    raise ImportError(
        "Failed to import pyvista, install scivianna using the command pip install scivianna[pyvista]"
    )

from scivianna.utils.polygonize_tools import PolygonCoords, PolygonElement

class ExtrudedStructuredMesh(Geometry2D):
    """Structured mesh build from a set of PolygonElement on the XY plane, extruded at a set of Z values
    """
    def __init__(
        self, xy_mesh: List[PolygonElement], z_coords: np.ndarray, extrusion_vector: Tuple[float, float, float] = (0, 0, 1)
    ):
        """Builds the mesh based on the (r, theta, phi) bins.

        Parameters
        ----------
        xy_mesh : List[PolygonElement]
            List of PolygonElement to extrude along the z axis
        z_coords : np.ndarray
            Bins on the Z axis
        """
        super().__init__()
        
        # Create structured grid
        self.base_polygons = xy_mesh
        self.z_coords = z_coords
        self.extrusion_vector = extrusion_vector

        self.build_pyvista_geometry()
        
        self.grids:Dict[str, Dict[str, Any]] = {}

        self.past_computation = []

    def build_pyvista_geometry(self,):
        """Builds the PyVista geometry from the list of polygons
        """
        count_cells = len(self.base_polygons)

        # Preparing point and face per base polygon

        polygons = [p.to_shapely(z_coord=0) for p in self.base_polygons]
        polygon_points:List[shapely.coords.CoordinateSequence] = []
        polygon_faces = []
        cell_ids = []

        remap_cells = any([len(polygons[k].interiors) for k in range(count_cells)])

        print("Warning : PyVista remapping cell IDs")

        for k in range(count_cells):
            if (len(polygons[k].interiors) == 0):
                poly = polygons[k]
                points = poly.exterior.coords
                faces = [len(points)] + list(range(len(points)))
                cell_ids += [k] if remap_cells else [self.base_polygons[k].cell_id]
            else:
                triangulated = shapely.constrained_delaunay_triangles(polygons[k])
                points = []
                faces = []
                for triangle in triangulated.geoms:
                    points.extend(np.array(triangle.exterior.coords))

                    # Create face indices (3 for triangle)
                    face = [3] + [len(points) - 3 + i for i in range(3)]
                    faces.extend(face)
                    cell_ids += [k]

            polygon_points.append(points)
            polygon_faces.append(faces)

        all_points = []
        all_cells = []
        points_ids = []
        current_point_count = 0

        for k in range(count_cells):
            all_points += polygon_points[k]
            faces = np.array(polygon_faces[k])
            faces[1:] += current_point_count
            all_cells += [faces]

            current_point_count += len(polygon_points[k])

            points_ids += [k for _ in range(len(polygon_points[k]))]

        all_cells = np.concatenate(all_cells, axis=0)

        mesh = pv.PolyData(
                np.array(all_points),
                all_cells,
            )#.clean()

        mesh.cell_data["cell_id"] = cell_ids

        extruded_cells = [
            mesh.copy(deep=True)
                .translate([0., 0., self.z_coords[i]])
                .extrude_trim(
                    self.extrusion_vector, 
                    pv.Plane(
                        center=(0, 0, self.z_coords[i+1]), 
                        direction=self.extrusion_vector, 
                        i_size=1e6, 
                        j_size=1e6
                    ), 
                    inplace=True
                )
            for i in range(len(self.z_coords)-1)
        ]

        for i in range(len(extruded_cells)):
            extruded_cells[i].cell_data["cell_id"] += i*count_cells

        full_mesh = pv.merge(
            extruded_cells, 
            merge_points = False
        )

        self.unstructured_mesh: pv.UnstructuredGrid = pv.UnstructuredGrid(full_mesh, deep = True)

    def set_values(self, name:str, grid:Dict[int, Any]):
        """Setting a Numpy array grid to the given name. The numpy array must be of size (nx, ny, nz) and the data are called in the XYZ order.

        Parameters
        ----------
        name : str
            Field name
        grid : np.ndarray
            Field value
        """
        self.grids[name] = grid

    def get_cells_values(self, name:str, cell_ids:List[int]) -> np.ndarray:
        """Returns a field values for a list of cell indexes

        Parameters
        ----------
        name : str
            field name
        cell_ids : List[int]
            cells indexes

        Returns
        -------
        np.ndarray
            List of values per cell

        Raises
        ------
        RuntimeError
            Requested a field before defining it
        """
        if name not in self.grids:
            raise RuntimeError(f"Field {name} is not defined. Found {list(self.grids.keys())}.")
        if len(cell_ids) == 0:
            return []
            
        return [self.grids[name][c] for c in cell_ids]

    def compute_2D_slice(
        self,
        origin: Tuple[float, float, float],
        u: Tuple[float, float, float],
        v: Tuple[float, float, float],
    ) -> List[PolygonElement]:
        """Computes the PolygonElement list for a slice of the mesh

        Parameters
        ----------
        origin : Tuple[float, float, float]
            Slice origin
        u : Tuple[float, float, float]
            First axis vector
        v : Tuple[float, float, float]
            Second axis vector

        Returns
        -------
        List[PolygonElement]
            List of polygon elements defining the cut

        Raises
        ------
        ValueError
            U and V are either parallel or one is of zero length.
        """
        if self.past_computation == [*list(u), *list(v), *list(origin)]:
            return self.polygons
        
        u = np.array(u)/np.linalg.norm(u)
        v = np.array(v)/np.linalg.norm(v)
        w = np.cross(u, v)

        if np.linalg.norm(w) == 0.:
            raise ValueError(f"u and v must be both non zero and non parallel, found {u}, {v}")
        
        w /= np.linalg.norm(w)

        mesh_slice: pv.PolyData = self.unstructured_mesh.slice(normal = w, origin = origin)

        polygon_elements = {}

        cell_per_id = {i: [] for i in set(mesh_slice.cell_data["cell_id"])}

        ids = mesh_slice.cell_data["cell_id"]
        
        for j in range(len(ids)):
            cell_per_id[ids[j]].append(mesh_slice.get_cell(j).point_ids)
            
        for cell_id in cell_per_id:
            rings = []

            edges = cell_per_id[cell_id]
            current_loop = []
            found = False

            while len(edges) > 0:
                if current_loop == []:
                    current_loop.append(edges[0])
                    current_point = current_loop[0][1]
                    edges.remove(edges[0])
                else:
                    found = False
                    for s in edges:
                        if current_point in [s[0], s[1]]:
                            if s[1] == current_point:
                                current_loop.append([s[1], s[0]])
                                current_point = s[0]
                            else:
                                current_loop.append(s)
                                current_point = s[1]
                            edges.remove(s)
                            found = True
                            break

                    if not found:
                        rings.append(current_loop)
                        current_loop = []

            if current_loop != []:
                rings.append(current_loop)

            polys: List[shapely.Polygon] = []

            for loop in rings:
                ids = np.array([e[0] for e in loop] + [loop[-1][1]] + [loop[0][0]])

                polys.append(shapely.Polygon(shell=np.array([mesh_slice.points[ids].dot(u), mesh_slice.points[ids].dot(v)]).T, holes=[]))

            if len(polys) == 0:
                continue
                
            if len(polys) == 1:
                polygon_elements[cell_id] = PolygonElement(
                    PolygonCoords(
                        np.array(polys[0].exterior.coords)[:, 0], np.array(polys[0].exterior.coords)[:, 1]
                    ),
                    [],
                    cell_id,
                )
            else:
                main = polys[0]
                holes = []
                for poly in polys[1:]:
                    if main.contains(poly):
                        holes.append(poly)
                    else:
                        holes.append(main)
                        main = poly
                        
                polygon_elements[cell_id] = PolygonElement(
                    PolygonCoords(
                        np.array(main.exterior.coords)[:, 0], np.array(main.exterior.coords)[:, 1]
                    ),
                    [PolygonCoords(
                        np.array(h.exterior.coords)[:, 0], np.array(h.exterior.coords)[:, 1]
                    ) for h in holes],
                    cell_id,
                )
        self.polygons = list(polygon_elements.values())
        self.past_computation == [*list(u), *list(v), *list(origin)]
        
        return self.polygons

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from scivianna.plotter_2d.polygon.matplotlib import Matplotlib2DPolygonPlotter
    from scivianna.data.data2d import Data2D
    from scivianna.utils.color_tools import interpolate_cmap_at_values

    
    outer_square = [(0, 0), (2, 0), (2, 2), (0, 2)]
    inner_hole = [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]
    inner_hole_0 = [(0.5, 0.5), (1., 0.5), (1., 1.5), (0.5, 1.5)]
    inner_hole_1 = [(1., 0.5), (1.5, 0.5), (1.5, 1.5), (1., 1.5)]

    outer_coords = PolygonCoords(
        [e[0] for e in outer_square],
        [e[1] for e in outer_square]
    )
    
    inner_coords = PolygonCoords(
        [e[0] for e in inner_hole],
        [e[1] for e in inner_hole]
    )
    
    inner_coords_0 = PolygonCoords(
        [e[0] for e in inner_hole_0],
        [e[1] for e in inner_hole_0]
    )
    
    inner_coords_1 = PolygonCoords(
        [e[0] for e in inner_hole_1],
        [e[1] for e in inner_hole_1]
    )
    
    p0 = PolygonElement(outer_coords, [inner_coords], 0 )

    p1 = PolygonElement(inner_coords_0, [], 1)

    p2 = PolygonElement(inner_coords_1, [], 2)


    mesh = ExtrudedStructuredMesh(
        [p0, p1, p2], list(range(5))
    )
    mesh.set_values("id", {i:i for i in range(4*3)})

    polygons = mesh.compute_2D_slice((1., 1., 0.5), (1, 0, 0), (0, 1, 0))

    data = Data2D.from_polygon_list(polygons)

    data.cell_values = mesh.get_cells_values("id", [p.cell_id for p in polygons])
    data.cell_colors = interpolate_cmap_at_values("viridis", np.array(data.cell_values)/(4*3))
    data.cell_edge_colors = get_edges_colors(data.cell_colors)
    
    plotter = Matplotlib2DPolygonPlotter()
    plotter.plot_2d_frame(data)
    plotter.figure.savefig("test_extruded_0.png")
    plt.close()

    polygons = mesh.compute_2D_slice((1., 1., 0.5), (1, 0, 0), (0, 0, 1))
    data = Data2D.from_polygon_list(polygons)

    data.cell_values = mesh.get_cells_values("id", [p.cell_id for p in polygons])
    data.cell_colors = interpolate_cmap_at_values("viridis", np.array(data.cell_values)/(4*3))
    data.cell_edge_colors = get_edges_colors(data.cell_colors)

    plotter = Matplotlib2DPolygonPlotter()
    plotter.plot_2d_frame(data)
    plotter.figure.savefig("test_extruded_1.png")
    plt.close()
