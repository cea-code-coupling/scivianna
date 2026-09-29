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
  reassemble their cut edges into one polygon boundary. ``compute_2D_data``
  below performs that logic on top of the raw slice produced by the MED
  interface: it takes the per-prism polygons returned by
  ``MEDInterface.compute_2D_data`` (one cut loop per elementary 3D cell),
  maps their cells back to the logical polygon ids, and unions the cut loops
  belonging to the same id into single polygons.
* Once the MED geometry is built, this class hands it over (mesh + ``cell_id``
  field) to an internal :class:`~scivianna.interface.med_interface.MEDInterface`
  instance, which performs all the actual Scivianna operations (2D slicing,
  3D data computation, value extraction), so no logic is duplicated between
  this class and the MED interface.

Caveats
-------
The exact Python import name (``medcoupling`` vs ``MEDCoupling``) and the
precise signature of ``buildSlice3D`` / ``buildExtrudedMesh`` can differ
slightly between MEDCoupling releases; adjust as needed for your install.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

import multiprocessing as mp

import numpy as np
from scivianna.constants import GEOMETRY, MESH
from scivianna.interface.med_interface import MEDInterface
import shapely
import shapely.coords

from scivianna.data.data2d import Data2D
from scivianna.enums import GeometryType, VisualizationMode
from scivianna.interface.generic_interface import Geometry2DPolygon, Geometry3D
from scivianna.logging_config import get_logger

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


class ExtrudedStructuredMesh(Geometry2DPolygon, Geometry3D):
    """Structured mesh build from a set of PolygonElement on the XY plane, extruded at a set of Z values.

    The base polygons, the z bins and the extrusion vector are either given at construction time or
    provided through ``read_file`` (which allows driving the interface through a ComputeSlave). The
    MEDCoupling geometry is built lazily by :meth:`build_medcoupling_geometry` before the first
    ``compute_2D_data`` / ``compute_3D_data`` call, then handed over to an internal
    :class:`~scivianna.interface.med_interface.MEDInterface`, which performs all the actual Scivianna
    operations (2D slicing, 3D data, value extraction) so that no logic is duplicated between this
    class and the MED interface.
    """

    geometry_type: GeometryType = GeometryType._3D_INFINITE

    def __init__(
        self,
        xy_mesh: List[PolygonElement] = None,
        z_coords: np.ndarray = None,
        extrusion_vector: Tuple[float, float, float] = (0, 0, 1),
    ):
        """Builds the interface from a set of base polygons on the XY plane.

        The MEDCoupling geometry is not built here; it is built lazily by
        ``build_medcoupling_geometry`` before the first ``compute_2D_data`` /
        ``compute_3D_data`` call, once all the geometry inputs are available
        (either from these arguments or from a prior ``read_file`` call).

        Parameters
        ----------
        xy_mesh : List[PolygonElement], optional
            List of PolygonElement to extrude along the z axis
        z_coords : np.ndarray, optional
            Bins on the Z axis (m)
        extrusion_vector : Tuple[float, float, float], optional
            Direction vector along which the base mesh is extruded (default: (0, 0, 1))
        """
        super().__init__()

        self.base_polygons = xy_mesh
        self.z_coords = np.asarray(z_coords, dtype=float) if z_coords is not None else None
        self.extrusion_vector = np.array(extrusion_vector, dtype=float)

        self.med_interface = MEDInterface()

        # Storing a dictionnary mapping fields names to their base mesh ids
        self.grids: Dict[str, Dict[int, Any]] = {}

    # ------------------------------------------------------------------
    # Generic interface - file handling
    # ------------------------------------------------------------------
    def read_file(self, file_path: str, file_label: str) -> None:
        """Read a geometry input and store its content in the interface.

        The geometry is described by three inputs, each sent through one call of this function:

        -   ``file_path="base_polygons"`` with ``file_label`` a list of :class:`PolygonElement`
            (the 2D base polygons on the XY plane),
        -   ``file_path="z_coords"`` with ``file_label`` a list of z bins,
        -   ``file_path="extrusion_vector"`` with ``file_label`` a 3D direction vector.

        The MEDCoupling geometry itself is not built here; it is built lazily by
        ``build_medcoupling_geometry`` before the first ``compute_2D_data`` /
        ``compute_3D_data`` call, once all the inputs are available.

        Parameters
        ----------
        file_path : str
            Geometry input label: "base_polygons", "z_coords" or "extrusion_vector"
        file_label : List[PolygonElement], list of float, or Tuple[float, float, float]
            Value associated to the given geometry input label

        Raises
        ------
        ValueError
            If the given file_path is not a supported geometry input label.
        """
        if file_path == "base_polygons":
            self.base_polygons = list(file_label)

        elif file_path == "z_coords":
            self.z_coords = np.asarray(file_label, dtype=float)

        elif file_path == "extrusion_vector":
            self.extrusion_vector = np.array(file_label, dtype=float)

        else:
            raise ValueError(
                f"Unknown geometry input label {file_path}. "
                "Expected one of 'base_polygons', 'z_coords' or 'extrusion_vector'."
            )

    def get_labels(self) -> List[str]:
        """Returns a list of fields names displayable with this interface.

        Returns
        -------
        List[str]
            List of fields names (the mesh itself plus every field set through ``set_values``).
        """
        return [MESH] + list(self.grids.keys())

    def get_label_coloring_mode(self, label: str) -> VisualizationMode:
        """Returns wheter the given field is colored based on a string value or a float.

        Parameters
        ----------
        label : str
            Field to color name

        Returns
        -------
        VisualizationMode
            Coloring mode
        """
        if label == MESH:
            return VisualizationMode.NONE

        return VisualizationMode.FROM_VALUE

    def get_file_input_list(self) -> List[Tuple[str, str]]:
        """Returns a list of file label and its description for the GUI.

        Returns
        -------
        List[Tuple[str, str]]
            List of (file label, description). This interface reads no file, so the list is empty.
        """
        return []

    def save(self, file_path: Path, include_files: bool):
        """Pickle saves the slave content to a file, allows slave state reload.

        Two modes are available:
            -   If **include_files** is at True, all loaded data are saved, the pickled file can be loaded on its own to recover last session.
            -   If **include_files** is at False, only the computed data are loaded, enabling faster first computation allowing a smaller pickle file size.

        Parameters
        ----------
        file_path : Path
            File in which save the file
        include_files : bool
            Included loaded file
        """
        raise NotImplementedError(
            f"Function save not implemented for class {self.__class__.__name__}."
        )

    def load(self, file_path: Path, include_files: bool):
        """Pickle loads the slave content to a file, allows slave state reload.

        Two modes are available:
            -   If **include_files** is at True, all loaded data are saved, the pickled file can be loaded on its own to recover last session.
            -   If **include_files** is at False, only the computed data are loaded, enabling faster first computation allowing a smaller pickle file size.

        Parameters
        ----------
        file_path : Path
            File from which load the slave
        include_files : bool
            Included loaded file
        """
        raise NotImplementedError(
            f"Function load not implemented for class {self.__class__.__name__}."
        )

    # ------------------------------------------------------------------
    # Geometry construction
    # ------------------------------------------------------------------
    def _ensure_geometry_built(self):
        """Builds the MEDCoupling geometry if it has not been built yet.

        The build is lazy so that the geometry inputs (base polygons, z bins and
        extrusion vector) can be provided either at construction time or through
        ``read_file`` calls before the first ``compute_2D_data`` /
        ``compute_3D_data`` request.
        """
        if getattr(self, "unstructured_mesh", None) is not None:
            return

        if self.base_polygons is None or len(self.base_polygons) == 0:
            raise RuntimeError(
                "No base polygons defined; provide them at construction time or through a "
                "read_file call with file_path='base_polygons' before computing data."
            )

        if self.z_coords is None or len(self.z_coords) < 2:
            raise RuntimeError(
                "Not enough z coordinates (at least two are required); provide them at "
                "construction time or through a read_file call with file_path='z_coords' "
                "before computing data."
            )

        self.build_medcoupling_geometry()

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

        self.base_cell_polygon_id_dict = dict(
            zip(range(len(base_cell_polygon_id)), base_cell_polygon_id)
        )

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

        # Hand the built geometry over to the MED interface, which then performs
        # all the actual Scivianna operations (slicing, 3D data, value extraction).
        cell_id_field = mc.MEDCouplingFieldDouble(mc.ON_CELLS, mc.ONE_TIME)
        cell_id_field.setName("cell_id")
        cell_id_field.setMesh(mesh3d)
        cell_id_field.setArray(mc.DataArrayDouble(cell_id_values.astype(float)))

        self.med_interface.read_file(cell_id_field, GEOMETRY)

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
            for name in self.med_interface.fields:
                field_np_array, _ = self.med_interface._get_field_at_time(name, 0.0)
                if field_np_array is None:
                    continue

                field = mc.MEDCouplingFieldDouble(mc.ON_CELLS, mc.ONE_TIME)
                field.setName(name)
                field.setMesh(self.unstructured_mesh)
                field.setArray(mc.DataArrayDouble(np.asarray(field_np_array, dtype=float)))
                mc.WriteFieldUsingAlreadyWrittenMesh(path_3d, field)

        logger.info("Wrote debug meshes to %s and %s", path_2d, path_3d)

        return {"2d": path_2d, "3d": path_3d}

    # ------------------------------------------------------------------
    # Field handling
    # ------------------------------------------------------------------
    def set_values(self, name: str, grid: Dict[int, Any]):
        """Setting a dict grid (cell_id -> value) to the given name.

        The values are scattered onto the extruded mesh cells following the
        ``cell_id`` mapping and stored in the internal MED interface as a
        MEDCoupling field, which then serves them for display and extraction.

        Parameters
        ----------
        name : str
            Field name
        grid : Dict[int, Any]
            Field value, keyed by cell_id

        Raises
        ------
        RuntimeError
            If the geometry has not been built yet (no base polygons or z coordinates provided).
        """
        self._ensure_geometry_built()

        cell_ids = self.cell_id_values

        keys = np.fromiter(grid.keys(), dtype=cell_ids.dtype)
        values = np.fromiter(grid.values(), dtype=float)

        sort_idx = np.argsort(keys)
        sorted_keys = keys[sort_idx]
        sorted_values = values[sort_idx]

        idx = np.searchsorted(sorted_keys, cell_ids)

        if not np.all(sorted_keys[idx] == cell_ids):
            raise ValueError("Some cell_ids are missing from grid")

        self.grids[name] = grid

    def get_value_dict(
        self,
        value_label: str,
        cells: List[Union[int, str]],
        options: Dict[str, Any],
        caller: str = "API",
    ) -> Dict[Union[int, str], str]:
        """Returns a cell name - field value map for a given field name.

        Parameters
        ----------
        value_label : str
            Field name to get values from
        cells : List[Union[int,str]]
            List of cells names
        options : Dict[str, Any]
            Additional options for frame computation.
        caller : str
            Identifier of the caller requesting the computation (default: "API")

        Returns
        -------
        Dict[Union[int,str], str]
            Field value for each requested cell names
        """
        if value_label == MESH:
            return {v: np.nan for v in cells}

        if value_label == "cell_id":
            # The cell_id field is internal (set at build time), but the slave
            # asks for it to fill Data2D.cell_values when used as a label.
            self._ensure_geometry_built()
            return dict(zip(cells, [int(c) for c in cells]))

        if value_label not in self.grids:
            raise RuntimeError(
                f"Field {value_label} is not defined. Found {list(self.grids.keys())}."
            )

        grid = self.grids[value_label]

        return dict(zip(cells, [grid[int(c)] for c in cells]))

    # ------------------------------------------------------------------
    # 3D data
    # ------------------------------------------------------------------
    def compute_3D_data(self, options: Dict[str, Any]) -> Tuple["Data3D", bool]:
        """Returns the full extruded 3D mesh (with a ``cell_id`` field and any
        field previously set via ``set_values``) ready for 3D display.

        Delegates to the internal MED interface, which converts the
        MEDCoupling mesh to a PyVista mesh and wraps it into a ``Data3D``.

        Parameters
        ----------
        options : Dict[str, Any]
            Additional options for frame computation.

        Returns
        -------
        Data3D
            3D geometry (and any field set via ``set_values``) to display
        bool
            Whether the data changed since the last call

        Raises
        ------
        RuntimeError
            If the geometry has not been built yet (no base polygons or z coordinates provided).
        """
        self._ensure_geometry_built()

        return self.med_interface.compute_3D_data(options if options is not None else {})

    def get_3d_value_dict(
        self,
        value_label: str,
        cells: List[Union[int, str]],
        options: Dict[str, Any],
        caller: str = "API",
    ) -> Dict[Union[int, str], str]:
        """Returns a cell name - field value map for a given field name on the 3D mesh.

        Parameters
        ----------
        value_label : str
            Field name to get values from
        cells : List[Union[int,str]]
            List of cells names
        options : Dict[str, Any]
            Additional options for frame computation.
        caller : str
            Identifier of the caller requesting the computation (default: "API")

        Returns
        -------
        Dict[Union[int,str], str]
            Field value for each requested cell names

        Raises
        ------
        RuntimeError
            If the geometry has not been built yet (no base polygons or z coordinates provided),
            or if the requested field is not defined.
        """
        self._ensure_geometry_built()

        if value_label == MESH:
            return {v: np.nan for v in cells}

        if value_label == "cell_id":
            # The cell_id field is internal (set at build time), but it can be
            # requested to check the mesh cells <-> logical polygon mapping.
            return dict(zip(cells, [int(c) for c in cells]))

        if value_label not in self.grids:
            raise RuntimeError(
                f"Field {value_label} is not defined. Found {list(self.grids.keys())}."
            )

        grid = self.grids[value_label]

        return dict(zip(cells, [grid[int(c)] for c in cells]))

    # ------------------------------------------------------------------
    # Slicing
    # ------------------------------------------------------------------
    def compute_2D_data(
        self,
        u: Tuple[float, float, float],
        v: Tuple[float, float, float],
        origin: Tuple[float, float, float],
        size_u: float,
        size_v: float,
        q_tasks: mp.Queue,
        options: Dict[str, Any],
        caller: str = "API",
    ) -> Tuple[Data2D, bool]:
        """Returns a list of polygons that defines the geometry in a given frame.

        Parameters
        ----------
        u : Tuple[float, float, float]
            Horizontal coordinate director vector
        v : Tuple[float, float, float]
            Vertical coordinate director vector
        origin : Tuple[float, float, float]
            Physical 3D position of the slice center
        size_u : float
            Size of the slice along the u axis
        size_v : float
            Size of the slice along the v axis
        q_tasks : mp.Queue
            Queue from which get orders from the master.
        options : Dict[str, Any]
            Additional options for frame computation.
        caller : str
            Identifier of the caller requesting the computation (default: "API")

        Returns
        -------
        Data2D
            Geometry to display
        bool
            Were the polygons updated compared to the past call

        Raises
        ------
        RuntimeError
            If the geometry has not been built yet (no base polygons or z coordinates provided).
        """
        if options is None:
            options = {}

        self._ensure_geometry_built()

        polygons = self.med_interface.compute_2D_data(
            u, v, origin, size_u, size_v, q_tasks, options, caller
        )[0]

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
        data = Data2D.from_polygon_list(new_polygon_list)

        value_dict = self.med_interface.get_value_dict(
            "cell_id", data.cell_ids.tolist(), options, caller
        )
        data.cell_values = np.array([value_dict[c] for c in data.cell_ids])

        return data, True


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from scivianna.slave import ComputeSlave
    from scivianna.plotter_2d.api import plot_frame_in_axes

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

    # Drive the interface through a ComputeSlave: the geometry inputs are sent
    # with read_file, and the MEDCoupling geometry is built lazily on the first
    # compute_2D_data call. The "cell_id" field (set at build time) is used as
    # the coloring label, so the slave fills data.cell_values itself.
    slave = ComputeSlave(ExtrudedStructuredMesh)
    slave.read_file("base_polygons", [p0, p1, p2])
    slave.read_file("z_coords", list(range(5)))
    slave.read_file("extrusion_vector", (0, 0, 1))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    plot_frame_in_axes(
        slave,
        "cell_id",
        axs[0]
    )
    plot_frame_in_axes(
        slave,
        "cell_id",
        axs[1],
        v = (0, 0, 1)
    )

    fig.savefig("test_extruded.png")