"""
Extruded mesh utilities for Scivianna.

This module provides utilities for creating extruded structured meshes from 2D polygons.

This version uses MEDCoupling (``medcoupling``) instead of VTK/PyVista as the
underlying 3D mesh engine.

Design notes
------------
* The base 2D geometry is triangulated per polygon with
  ``shapely.constrained_delaunay_triangles`` (same approach as the original
  ``polygons_to_polydata`` helper), so that holes are handled correctly: a
  polygon with a hole becomes several triangle cells, none of which cover
  the hole. All triangles coming from the same ``PolygonElement`` share that
  element's ``cell_id``.
* The 2D triangulated mesh's nodes are merged (``mergeNodes``), mirroring
  the original ``pyvista`` ``PolyData.clean()`` call. This makes adjacent
  triangles (whether from the same polygon or from touching polygons) share
  point ids, which is required later to stitch cut edges back into closed
  loops when slicing.
* The 3D mesh is built with MEDCoupling's native
  ``MEDCouplingUMesh.buildExtrudedMesh``, extruding the 2D triangulated mesh
  along a 1D path mesh whose nodes sit at ``z_coords[i] * extrusion_vector``.
  Policy ``0`` ("translation only") reproduces the per-slab translation used
  by the previous VTK ``extrude_trim`` implementation. Each 2D triangle
  becomes one ``NORM_PENTA6`` (triangular prism) cell per layer, and
  MEDCoupling numbers the resulting cells level by level
  (``layer * n_base_cells + base_cell_index``), which is exactly what makes
  offsetting the ``cell_id`` field per layer straightforward.
* Because a single logical polygon can be represented by several prism
  cells (from triangulation) sharing one ``cell_id``, slicing needs to
  reassemble their cut edges into one polygon boundary, exactly like the
  original VTK implementation did. ``MEDCouplingUMesh.buildSlice3D`` returns
  one cut loop per elementary 3D cell (not merged across cells sharing a
  field value), so ``compute_2D_slice`` below re-implements the same
  point-id edge-walking reconstruction as the original code, just sourced
  from MEDCoupling's slice output instead of VTK's.

Caveats
-------
The exact Python import name (``medcoupling`` vs ``MEDCoupling``) and the
precise signature of ``buildSlice3D`` / ``buildExtrudedMesh`` can differ
slightly between MEDCoupling releases; adjust as needed for your install.
"""

import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np
from scivianna.constants import GEOMETRY
from scivianna.interface.med_interface import MEDInterface
import shapely
import shapely.coords

from scivianna.data.data2d import Data2D
from scivianna.interface.generic_interface import Geometry2D, Geometry3D
from scivianna.logging_config import get_logger
from scivianna.utils.color_tools import get_edges_colors

if TYPE_CHECKING:
    from scivianna.data.data3d import Data3D

logger = get_logger(__name__)

try:
    import medcoupling as mc
except ImportError:
    try:
        import MEDCoupling as mc  # noqa: N811 - older packaging exposes this capitalised name
    except ImportError:
        raise ImportError(
            "Failed to import medcoupling, install scivianna using the command "
            "pip install scivianna[medcoupling] (or install the MEDCoupling / MEDLoader "
            "Python bindings, e.g. via Salome or conda-forge's `medcoupling` package)."
        )

from scivianna.utils.polygonize_tools import PolygonCoords, PolygonElement


class ExtrudedStructuredMesh(Geometry2D, Geometry3D):
    """Structured mesh build from a set of PolygonElement on the XY plane, extruded at a set of Z values"""

    def __init__(
        self,
        xy_mesh: List[PolygonElement],
        z_coords: np.ndarray,
        extrusion_vector: Tuple[float, float, float] = (0, 0, 1),
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

        self.base_polygons = xy_mesh
        self.z_coords = np.asarray(z_coords, dtype=float)
        self.extrusion_vector = np.array(extrusion_vector, dtype=float)

        self.med_interface = MEDInterface()

        self.build_medcoupling_geometry()

        self.grids: Dict[str, Dict[int, Any]] = {}
        self.fields: Dict[str, Any] = {}

        self.past_computation = []

        # Cache for compute_3D_data(); invalidated whenever a field is set
        # (the mesh itself never changes after construction).
        self._data3d: "Data3D" = None


    # ------------------------------------------------------------------
    # Geometry construction
    # ------------------------------------------------------------------
    def _build_base_2d_mesh(self) -> Tuple["mc.MEDCouplingUMesh", np.ndarray]:
        """Triangulates every base polygon (accounting for holes) and returns a
        2D MEDCouplingUMesh (lying in 3D space, z=0) together with an array
        mapping each of its cells back to the originating PolygonElement's
        ``cell_id``.
        """
        all_points: List[List[float]] = []
        triangle_conn: List[List[int]] = []
        base_cell_polygon_id: List[int] = []

        point_offset = 0

        for polygon in self.base_polygons:
            pol = polygon.to_shapely(z_coord=0)

            if not pol.is_valid:
                pol = shapely.make_valid(pol)

            triangulated = shapely.constrained_delaunay_triangles(pol)

            for triangle in triangulated.geoms:
                ext_x, ext_y = triangle.exterior.xy
                # shapely repeats the first point at the end of the ring; drop it.
                n = len(ext_x) - 1
                if n < 3:
                    continue

                ids = list(range(point_offset, point_offset + n))
                for k in range(n):
                    all_points.append([ext_x[k], ext_y[k], 0.0])
                point_offset += n

                triangle_conn.append(ids)
                base_cell_polygon_id.append(polygon.cell_id)

        coords = np.array(all_points) if all_points else np.zeros((0, 3))

        mesh2d = mc.MEDCouplingUMesh("base_2d_mesh", 2)
        mesh2d.setCoords(mc.DataArrayDouble(coords))
        mesh2d.allocateCells(len(triangle_conn))
        for conn in triangle_conn:
            mesh2d.insertNextCell(mc.NORM_TRI3, conn)
        mesh2d.finishInsertingCells()

        # Merge coincident points (mirrors PyVista's PolyData.clean()) so that
        # adjacent triangles - whether from the same polygon (across a
        # triangulation seam) or from touching polygons - share point ids.
        mesh2d.mergeNodes(1e-8)
        mesh2d.checkConsistencyLight()

        return mesh2d, np.array(base_cell_polygon_id, dtype=np.int64)

    def _build_1d_path_mesh(self) -> "mc.MEDCouplingUMesh":
        """Builds the 1D path mesh along which the 2D mesh is extruded: one
        node per z_coords entry, positioned at z_coords[i] * extrusion_vector.
        """
        points = np.outer(self.z_coords, self.extrusion_vector)

        mesh1d = mc.MEDCouplingUMesh("extrusion_path", 1)
        mesh1d.setCoords(mc.DataArrayDouble(points))
        mesh1d.allocateCells(len(self.z_coords) - 1)
        for i in range(len(self.z_coords) - 1):
            mesh1d.insertNextCell(mc.NORM_SEG2, [i, i + 1])
        mesh1d.finishInsertingCells()

        return mesh1d

    def build_medcoupling_geometry(self):
        """Builds the MEDCoupling 3D unstructured mesh from the list of polygons,
        using MEDCoupling's native ``buildExtrudedMesh``.
        """
        count_cells = len(self.base_polygons)
        n_layers = len(self.z_coords) - 1

        mesh2d, base_cell_polygon_id = self._build_base_2d_mesh()
        mesh1d = self._build_1d_path_mesh()

        self.base_cell_polygon_id_dict = dict(zip(range(len(base_cell_polygon_id)), base_cell_polygon_id))

        # Policy 0 = "translation only": each level of the resulting 3D mesh
        # is a translated copy of mesh2d, following the vectors of mesh1d's
        # segments - equivalent to the previous per-slab translate+extrude.
        mesh3d = mesh2d.buildExtrudedMesh(mesh1d, 0)
        mesh3d.checkConsistencyLight()

        n_base_cells = mesh2d.getNumberOfCells()

        # MEDCoupling numbers the extruded mesh's cells level by level:
        # cell (layer, base_cell) -> layer * n_base_cells + base_cell.
        cell_id_values = np.empty(n_base_cells * n_layers, dtype=np.int64)
        for layer in range(n_layers):
            start = layer * n_base_cells
            cell_id_values[start:start + n_base_cells] = layer * count_cells + base_cell_polygon_id

        self.unstructured_mesh = mesh3d
        self.cell_id_values: np.ndarray = cell_id_values
        # Kept around (rather than discarded as a local var) so it can be
        # inspected or saved for debugging - see save_debug_meshes().
        self.base_2d_mesh = mesh2d

        self.med_interface.read_file("extruded_mesh_debug_3d.med", GEOMETRY)

    def save_debug_meshes(
        self,
        output_dir: str = ".",
        prefix: str = "extruded_mesh_debug",
        write_fields: bool = True,
    ) -> Dict[str, str]:
        """Writes the base 2D (triangulated) mesh and the resulting 3D
        extruded mesh to ``.med`` files, for inspection in ParaVis/Salome (or
        by reloading them with ``medcoupling.ReadMeshFromFile``). Handy when
        an extrusion or a slice doesn't look right and you want to look at
        the actual MEDCoupling geometry rather than at Scivianna's polygons.

        The 3D file also always gets a ``cell_id`` cell field (matching
        ``self.cell_id_values``), plus - if ``write_fields`` is True - every
        field previously registered through ``set_values``, so the mapping
        between mesh cells and Scivianna's logical polygon ids can be
        checked directly in the viewer.

        Parameters
        ----------
        output_dir : str
            Directory the .med files are written into (created if needed).
        prefix : str
            Filename prefix; files are written as ``{prefix}_2d.med`` and
            ``{prefix}_3d.med``.
        write_fields : bool
            If True, also write every field previously registered through
            ``set_values`` into the 3D .med file.

        Returns
        -------
        Dict[str, str]
            ``{"2d": <path>, "3d": <path>}`` of the written files.
        """
        if self.base_2d_mesh is None or self.unstructured_mesh is None:
            raise RuntimeError(
                "Meshes are not available; build_medcoupling_geometry() must run first."
            )

        os.makedirs(output_dir, exist_ok=True)

        path_2d = os.path.join(output_dir, f"{prefix}_2d.med")
        path_3d = os.path.join(output_dir, f"{prefix}_3d.med")

        self.base_2d_mesh.setName("base_2d_mesh")
        mc.WriteUMesh(path_2d, self.base_2d_mesh, True)

        self.unstructured_mesh.setName("extruded_3d_mesh")
        mc.WriteUMesh(path_3d, self.unstructured_mesh, True)

        # Always write cell_id, so the field <-> logical polygon mapping can
        # be inspected even if set_values() was never called.
        cell_id_field = mc.MEDCouplingFieldDouble(mc.ON_CELLS, mc.ONE_TIME)
        cell_id_field.setName("cell_id")
        cell_id_field.setMesh(self.unstructured_mesh)
        cell_id_field.setArray(mc.DataArrayDouble(self.cell_id_values.astype(float)))
        cell_id_field.checkConsistencyLight()
        mc.WriteFieldUsingAlreadyWrittenMesh(path_3d, cell_id_field)

        if write_fields:
            for name, field in self.fields.items():
                field.setName(name)
                mc.WriteFieldUsingAlreadyWrittenMesh(path_3d, field)

        logger.info("Wrote debug meshes to %s and %s", path_2d, path_3d)

        return {"2d": path_2d, "3d": path_3d}
        # Kept around (rather than discarded as a local var) so it can be
        # inspected or saved for debugging - see save_debug_meshes().
        self.base_2d_mesh = mesh2d

    # ------------------------------------------------------------------
    # Field handling
    # ------------------------------------------------------------------
    def set_values(self, name: str, grid: Dict[int, Any]):
        """Setting a dict grid (cell_id -> value) to the given name.

        Parameters
        ----------
        name : str
            Field name
        grid : Dict[int, Any]
            Field value, keyed by cell_id
        """
        self.grids[name] = grid
        self._data3d = None  # invalidate cached 3D data, it needs this field baked in

        cell_ids = self.cell_id_values

        keys = np.fromiter(grid.keys(), dtype=cell_ids.dtype)
        values = np.fromiter(grid.values(), dtype=float)

        sort_idx = np.argsort(keys)
        sorted_keys = keys[sort_idx]
        sorted_values = values[sort_idx]

        idx = np.searchsorted(sorted_keys, cell_ids)

        if not np.all(sorted_keys[idx] == cell_ids):
            raise ValueError("Some cell_ids are missing from grid")

        cell_values = sorted_values[idx]

        field = mc.MEDCouplingFieldDouble(mc.ON_CELLS, mc.ONE_TIME)
        field.setName(name)
        field.setMesh(self.unstructured_mesh)
        field.setArray(mc.DataArrayDouble(cell_values))
        field.checkConsistencyLight()

        self.fields[name] = field

    def get_cells_values(self, name: str, cell_ids: List[int]) -> np.ndarray:
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

    # ------------------------------------------------------------------
    # 3D data
    # ------------------------------------------------------------------
    def compute_3D_data(self, options: Dict[str, Any] = None) -> Tuple["Data3D", bool]:
        """Returns the full extruded 3D mesh (with a ``cell_id`` field and any
        field previously set via ``set_values``) ready for 3D display.

        This mirrors ``MEDInterface.compute_3D_data``: the MEDCoupling mesh is
        converted to a PyVista mesh (PyVista/VTK still being the rendering
        backend for the 3D viewer) and wrapped into a ``Data3D``. Unlike
        ``MEDInterface``, this mesh is static (built once at construction), so
        the cache is only invalidated when a new field is registered via
        ``set_values``, not by a "time" option.

        Parameters
        ----------
        options : Dict[str, Any], optional
            Unused for now; kept for interface-compatibility with
            ``MEDInterface.compute_3D_data``.

        Returns
        -------
        Data3D
            3D geometry (and any field set via ``set_values``) to display
        bool
            Whether the data changed since the last call
        """
        import pyvista as pv

        from scivianna.data.data3d import Data3D

        if self._data3d is not None:
            logger.debug("Skipping 3D mesh computation (cached)")
            return self._data3d, False

        MC_TO_PV_CELLTYPE = {
            mc.NORM_POINT1: pv.CellType.VERTEX,
            mc.NORM_SEG2: pv.CellType.LINE,
            mc.NORM_SEG3: pv.CellType.QUADRATIC_EDGE,
            mc.NORM_TRI3: pv.CellType.TRIANGLE,
            mc.NORM_QUAD4: pv.CellType.QUAD,
            mc.NORM_TETRA4: pv.CellType.TETRA,
            mc.NORM_PENTA6: pv.CellType.WEDGE,
            mc.NORM_HEXA8: pv.CellType.HEXAHEDRON,
            mc.NORM_POLYGON: pv.CellType.POLYGON,
            mc.NORM_QPOLYG: pv.CellType.QUADRATIC_POLYGON,
            mc.NORM_POLYHED: pv.CellType.POLYHEDRON,
        }

        MC_DIM = {
            mc.NORM_POINT1: 0,
            mc.NORM_SEG2: 1,
            mc.NORM_SEG3: 1,
            mc.NORM_TRI3: 2,
            mc.NORM_QUAD4: 2,
            mc.NORM_TETRA4: 3,
            mc.NORM_PENTA6: 3,
            mc.NORM_HEXA8: 3,
            mc.NORM_POLYGON: 2,
            mc.NORM_QPOLYG: 3,  # only UnstructuredGrid
            mc.NORM_POLYHED: 3,
        }

        def _mc_ph_to_vtk_fast(connectivity: np.ndarray, offsets: np.ndarray) -> np.ndarray:
            if len(connectivity) == 0:
                return np.array([], dtype=int)
            cell_length = offsets[1:] - offsets[:-1]
            face_delims = connectivity == -1  # all face separators
            face_delims[offsets[:-1]] = True  # all cell separators

            face_offsets = np.r_[np.flatnonzero(face_delims), len(connectivity)]

            cell_delim_in_face_offsets = face_offsets.searchsorted(offsets)
            num_faces = cell_delim_in_face_offsets[1:] - cell_delim_in_face_offsets[:-1]

            face_length = face_offsets[1:] - face_offsets[:-1] - 1
            connectivity[face_delims] = face_length

            inds = np.empty((2 * len(cell_length),), dtype=int)
            inds[::2] = offsets[:-1]
            inds[1::2] = offsets[:-1]

            vals = np.empty((2 * len(cell_length),), dtype=int)
            vals[::2] = cell_length + 1
            vals[1::2] = num_faces

            return np.insert(connectivity, obj=inds, values=vals)

        def _to_unstructured(mesh: "mc.MEDCouplingUMesh", coords: np.ndarray) -> pv.UnstructuredGrid:
            offsets = mesh.getNodalConnectivityIndex().toNumPyArray()
            cell_length = offsets[1:] - offsets[:-1] - 1

            cell_types_idx = offsets[:-1]
            connectivity = np.array(mesh.getNodalConnectivity().toNumPyArray())
            mc_cell_types = np.array(connectivity[cell_types_idx])

            # Split off polyhedrons (assumed to have a higher type id than the
            # other cell types present).
            ind_l = np.searchsorted(mc_cell_types, mc.NORM_POLYHED, side="left")
            ind_r = np.searchsorted(mc_cell_types, mc.NORM_POLYHED, side="right")
            cell_types = np.array(mc_cell_types)

            types_idx = dict()
            for mc_type in MC_TO_PV_CELLTYPE:
                types_idx[mc_type] = cell_types == mc_type
            for mc_type, pv_type in MC_TO_PV_CELLTYPE.items():
                cell_types[types_idx[mc_type]] = pv_type

            connectivity_npl = connectivity[: offsets[ind_l]]
            connectivity_npl[cell_types_idx[:ind_l]] = cell_length[:ind_l]
            connectivity_npr = connectivity[offsets[ind_r]:]
            connectivity_npr[cell_types_idx[ind_r:]] = cell_length[ind_r:]

            connectivity_ph = _mc_ph_to_vtk_fast(
                connectivity[offsets[ind_l]: offsets[ind_r]],
                offsets=offsets[ind_l: ind_r + 1] - offsets[ind_l],
            )

            connectivity = np.r_[connectivity_npl, connectivity_npr, connectivity_ph]

            return pv.UnstructuredGrid(connectivity, cell_types, coords)

        def _to_polydata(mesh: "mc.MEDCouplingUMesh", coords: np.ndarray) -> pv.PolyData:
            offsets = mesh.getNodalConnectivityIndex().toNumPyArray()
            cell_length = offsets[1:] - offsets[:-1] - 1
            cell_types_idx = offsets[:-1]

            connectivity = np.array(mesh.getNodalConnectivity().toNumPyArray())
            any_type = connectivity[0]
            connectivity[cell_types_idx] = cell_length

            if MC_DIM[any_type] == 0:
                return pv.PolyData(coords, verts=connectivity)
            if MC_DIM[any_type] == 1:
                return pv.PolyData(coords, lines=connectivity)
            if MC_DIM[any_type] == 2:
                return pv.PolyData(coords, faces=connectivity)
            raise ValueError(f"{any_type} is not in known types: {MC_DIM=}")

        # Sort a working copy in MED file order (required before converting
        # to a single-cell-type-block PyVista mesh), and keep the resulting
        # permutation to reorder our own per-cell arrays (cell_id, fields).
        mesh = self.unstructured_mesh.deepCopyConnectivityOnly()
        permut = mesh.sortCellsInMEDFileFrmt()

        coords = np.array(mesh.getCoords().toNumPyArray())
        coords = np.c_[coords, np.zeros((coords.shape[0], 3 - coords.shape[1]))]

        offsets = mesh.getNodalConnectivityIndex().toNumPyArray()
        connectivity = np.array(mesh.getNodalConnectivity().toNumPyArray())
        last_cell_type = connectivity[offsets[-2]]
        max_type = MC_DIM[last_cell_type]

        if max_type == 3:
            pv_mesh = _to_unstructured(mesh, coords)
        elif max_type < 3:
            pv_mesh = _to_polydata(mesh, coords)
        else:
            raise ValueError(f"The coords shape is not valid: {coords.shape=}")

        if permut is not None:
            permut_np = permut.toNumPyArray()
            p = np.empty_like(permut_np)
            p[permut_np] = np.arange(permut_np.size)
        else:
            p = np.arange(pv_mesh.GetNumberOfCells())

        pv_mesh.cell_data["cell_id"] = self.cell_id_values[p]

        for name, field in self.fields.items():
            farr = field.getArray().toNumPyArray()
            pv_mesh.cell_data[name] = farr[p]

        self._data3d = Data3D.from_vtk(pv_mesh)

        return self._data3d, True

    # ------------------------------------------------------------------
    # Slicing
    # ------------------------------------------------------------------
    def compute_2D_slice(
        self,
        origin: Tuple[float, float, float],
        u: Tuple[float, float, float],
        v: Tuple[float, float, float]
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
            return self.data

        polygons = self.med_interface.compute_2D_data(u, v, origin, None, None, None, {})[0]

        cell_ids = polygons.cell_ids
        slices = (cell_ids / len(self.base_cell_polygon_id_dict.keys())).astype(int)
        base_ids = np.mod(cell_ids, len(self.base_cell_polygon_id_dict.keys()))
        base_ids = [
            self.base_cell_polygon_id_dict[e] for e in base_ids
        ]
        cell_ids = base_ids + len(self.base_polygons) * slices

        polygons.cell_ids = cell_ids
        for i, p in enumerate(polygons.polygons):
            p.cell_id = cell_ids[i]

        polygons_to_union: Dict[int, List[PolygonElement]] = {e: [] for e in cell_ids}

        for polygon in polygons.get_polygons():
            polygons_to_union[polygon.cell_id].append(polygon)

        def union_to_polygons(geometries):
            union = shapely.union_all(geometries)

            if union.geom_type == "MultiPolygon":
                return list(union.geoms)

            return [union]

        new_polygon_list = [
            PolygonElement.from_shapely(
                polygon=polygon,
                cell_id=cell_id
            )
            for cell_id, e in polygons_to_union.items()
            for polygon in union_to_polygons([p.to_shapely() for p in e])
        ]
        self.data = Data2D.from_polygon_list(
            new_polygon_list
        )

        self.past_computation = [*list(u), *list(v), *list(origin)]

        return self.data, True


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from scivianna.data.data2d import Data2D
    from scivianna.plotter_2d.polygon.matplotlib import Matplotlib2DPolygonPlotter
    from scivianna.utils.color_tools import interpolate_cmap_at_values

    outer_square = [(0, 0), (2, 0), (2, 2), (0, 2)]
    inner_hole = [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]
    inner_hole_0 = [(0.5, 0.5), (1.0, 0.5), (1.0, 1.5), (0.5, 1.5)]
    inner_hole_1 = [(1.0, 0.5), (1.5, 0.5), (1.5, 1.5), (1.0, 1.5)]

    outer_coords = PolygonCoords([e[0] for e in outer_square], [e[1] for e in outer_square])

    inner_coords = PolygonCoords([e[0] for e in inner_hole], [e[1] for e in inner_hole])

    inner_coords_0 = PolygonCoords([e[0] for e in inner_hole_0], [e[1] for e in inner_hole_0])

    inner_coords_1 = PolygonCoords([e[0] for e in inner_hole_1], [e[1] for e in inner_hole_1])

    p0 = PolygonElement(outer_coords, [inner_coords], 0)

    p1 = PolygonElement(inner_coords_0, [], 1)

    p2 = PolygonElement(inner_coords_1, [], 2)

    mesh = ExtrudedStructuredMesh([p0, p1, p2], list(range(5)))
    mesh.set_values("id", {i: i for i in range(4 * 3)})

    polygons = mesh.compute_2D_slice((1.0, 1.0, 0.5), (1, 0, 0), (0, 1, 0))

    data = Data2D.from_polygon_list(polygons)

    data.cell_values = mesh.get_cells_values("id", [p.cell_id for p in polygons])
    data.cell_colors = interpolate_cmap_at_values("viridis", np.array(data.cell_values) / (4 * 3))
    data.cell_edge_colors = get_edges_colors(data.cell_colors)

    plotter = Matplotlib2DPolygonPlotter()
    plotter.plot_2d_frame(data)
    plotter.figure.savefig("test_extruded_0.png")
    plt.close()

    polygons = mesh.compute_2D_slice((1.0, 1.0, 0.5), (1, 0, 0), (0, 0, 1))
    data = Data2D.from_polygon_list(polygons)

    data.cell_values = mesh.get_cells_values("id", [p.cell_id for p in polygons])
    data.cell_colors = interpolate_cmap_at_values("viridis", np.array(data.cell_values) / (4 * 3))
    data.cell_edge_colors = get_edges_colors(data.cell_colors)

    plotter = Matplotlib2DPolygonPlotter()
    plotter.plot_2d_frame(data)
    plotter.figure.savefig("test_extruded_1.png")
    plt.close()
