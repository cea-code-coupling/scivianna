from typing import Dict, List, Tuple
import numpy as np

try:
    import pyvista as pv
    from pyvista import core
    _PYVISTA_AVAILABLE = True
except ImportError:
    pv = None  # type: ignore[assignment]
    core = None  # type: ignore[assignment]
    _PYVISTA_AVAILABLE = False

def _require_pyvista():
    """Raise an error if pyvista is not available."""
    if not _PYVISTA_AVAILABLE:
        raise ImportError("Pyvista could not be imported, please pip install pyvista to use the vtk interface")

from scivianna.utils.polygonize_tools import PolygonCoords, PolygonElement


class StructuredMesh:
    """Generic structured mesh class built to help used to define their own structured mesh based geometries/results
    """
    grids: Dict[str, np.ndarray]
    """Numpy array allocating values to cells, coordinates (x, y, z) are expected."""

    def __init__(self):
        """Initializing the grids object
        """
        self.mesh: pv.StructuredGrid = None
        """Pyvista structured grid"""
        self.grids = {}

    def set_values(self, name:str, grid:np.ndarray):
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
            
        xs, ys, zs = np.unravel_index(cell_ids, self.grids[name].shape)

        return self.grids[name][xs, ys, zs]

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
        _require_pyvista()
        u = np.array(u)/np.linalg.norm(u)
        v = np.array(v)/np.linalg.norm(v)
        w = np.cross(u, v)

        if np.linalg.norm(w) == 0.:
            raise ValueError(f"u and v must be both non zero and non parallel, found {u}, {v}")
        
        w /= np.linalg.norm(w)

        mesh_slice: core.pointset.PolyData = self.mesh.slice(
            origin=origin, normal=w
        )

        cells = [c for c in mesh_slice.cell]
        verts = np.array(mesh_slice.points) - origin

        u_vals = np.dot(verts, u)
        v_vals = np.dot(verts, v)

        return [
            PolygonElement(
                exterior_polygon=PolygonCoords(
                    x_coords=[u_vals[j] for j in cells[i].point_ids],
                    y_coords=[v_vals[j] for j in cells[i].point_ids],
                ),
                holes=[],
                cell_id=mesh_slice["cell_id"][i],
            )
            for i in range(len(cells))
        ]

class CarthesianStructuredMesh(StructuredMesh):
    """Carthesian structured mesh
    """
    def __init__(
        self, x_coords: np.ndarray, y_coords: np.ndarray, z_coords: np.ndarray
    ):
        """Builds the mesh based on the (x, y, z) bins.

        Parameters
        ----------
        x_coords : np.ndarray
            Bins on the X axis
        y_coords : np.ndarray
            Bins on the Y axis
        z_coords : np.ndarray
            Bins on the Z axis
        """
        _require_pyvista()
        super().__init__()
        x, y, z = np.meshgrid(
            np.array(x_coords), np.array(y_coords), np.array(z_coords), indexing="ij"
        )
        self.mesh = pv.StructuredGrid(x, y, z)
        self.mesh["cell_id"] = np.arange(self.mesh.n_cells)


class CylindricalStructuredMesh(StructuredMesh):
    """Cylindrical structured mesh
    """
    def __init__(
        self, r_coords: np.ndarray, theta_coords: np.ndarray, z_coords: np.ndarray
    ):
        """Builds the mesh based on the (r, theta, z) bins.

        Parameters
        ----------
        r_coords : np.ndarray
            Bins on the R axis
        theta_coords : np.ndarray
            Bins on the Theta axis
        z_coords : np.ndarray
            Bins on the Z axis
        """
        _require_pyvista()
        super().__init__()
        assert r_coords.min() >= 0

        # Create meshgrid in cylindrical coordinates
        R, THETA, Z = np.meshgrid(r_coords, theta_coords, z_coords, indexing="ij")


        # Convert to Cartesian
        X = R * np.cos(THETA)
        Y = R * np.sin(THETA)
        Z = Z

        # Create structured grid
        self.mesh = pv.StructuredGrid(X, Y, Z)
        self.mesh["cell_id"] = np.arange(self.mesh.n_cells)


class SphericalStructuredMesh(StructuredMesh):
    """Spherical structured mesh
    """
    def __init__(
        self, r_coords: np.ndarray, theta_coords: np.ndarray, phi_coords: np.ndarray
    ):
        """Builds the mesh based on the (r, theta, phi) bins.

        Parameters
        ----------
        r_coords : np.ndarray
            Bins on the R axis
        theta_coords : np.ndarray
            Bins on the Theta axis
        zphi_coords : np.ndarray
            Bins on the Phi axis
        """
        _require_pyvista()
        super().__init__()
        # Create meshgrid in cylindrical coordinates
        R, THETA, PHI = np.meshgrid(r_coords, theta_coords, phi_coords, indexing="ij")

        # Convert to Cartesian
        X = R * np.sin(PHI) * np.cos(THETA)
        Y = R * np.sin(PHI) * np.sin(THETA)
        Z = R * np.cos(PHI)

        # Create structured grid
        self.mesh = pv.StructuredGrid(X, Y, Z)
        self.mesh["cell_id"] = np.arange(self.mesh.n_cells)


if __name__ == "__main__":
    import math
    import matplotlib.pyplot as plt

    from scivianna.plotter_2d.polygon.matplotlib import Matplotlib2DPolygonPlotter
    from scivianna.data.data2d import Data2D
    from scivianna.utils.color_tools import interpolate_cmap_at_values

    print("Computing carthesian mesh")
    mesh = CarthesianStructuredMesh(
        np.linspace(0, 4, 4),
        np.linspace(0, 4, 4),
        np.linspace(0, 4, 4),
    )
    mesh.set_values("id", np.arange(4*4*4).reshape(4, 4, 4))
    polygons = mesh.compute_2D_slice((1.5, 1.5, 1.5), (1, 0, 0), (0, 0, 1))

    data = Data2D.from_polygon_list(polygons)

    data.cell_values = mesh.get_cells_values("id", [p.cell_id for p in polygons])
    data.cell_colors = interpolate_cmap_at_values("viridis", data.cell_values/(4*4*4))
    
    plotter = Matplotlib2DPolygonPlotter()
    plotter.plot_2d_frame(data)
    plotter.figure.savefig("test_carthesian.png")



    print("Computing cylindrical mesh")
    mesh = CylindricalStructuredMesh(
        np.linspace(0, 4, 4),
        np.linspace(0, math.pi*2, 16),
        np.linspace(0, 4, 4),
    )
    mesh.set_values("id", np.arange(4*16*4).reshape(4, 16, 4))
    polygons = mesh.compute_2D_slice((1.5, 1.5, 1.5), (1, 0, 0), (0, 1, 0))

    data = Data2D.from_polygon_list(polygons)

    data.cell_values = mesh.get_cells_values("id", [p.cell_id for p in polygons])
    data.cell_colors = interpolate_cmap_at_values("viridis", data.cell_values/(4*16*4))
    
    plotter = Matplotlib2DPolygonPlotter()
    plotter.plot_2d_frame(data)
    plotter.figure.savefig("test_cylindrical.png")
    plt.close()




    print("Computing spherical mesh")
    mesh = SphericalStructuredMesh(
        np.linspace(0, 4, 4),
        np.linspace(0, math.pi*2, 16),
        np.linspace(0, math.pi, 16),
    )
    mesh.set_values("id", np.arange(4*16*16).reshape(4, 16, 16))
    polygons = mesh.compute_2D_slice((0., 0., 0.), (1, 0, 1), (0, 1, 0))

    data = Data2D.from_polygon_list(polygons)

    data.cell_values = mesh.get_cells_values("id", [p.cell_id for p in polygons])
    data.cell_colors = interpolate_cmap_at_values("viridis", data.cell_values/(4*16*16))
    
    plotter = Matplotlib2DPolygonPlotter()
    plotter.plot_2d_frame(data)
    plotter.figure.savefig("test_spherical.png")
    plt.close()
