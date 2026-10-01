"""
Gmsh interface for Scivianna.

Reads meshes (and, optionally, CAD / .geo files that are meshed on the fly) with
the gmsh Python API, converts them to a PyVista ``UnstructuredGrid`` and exposes
them through the ``Geometry2DPolygon`` and ``Geometry3D`` scivianna interfaces.

Features
--------
- Read ``.msh`` files (v2 / v4), and ``.geo`` / ``.step`` / ``.brep`` / ``.stl`` ... files
  (meshed automatically if they contain no elements)
- Physical groups exposed as the ``Material`` field
- Geometrical entities exposed as the ``Entity`` field
- Cell size (length / area / volume) exposed as the ``Size`` field
- Post-processing views stored in ``.msh`` files (NodeData / ElementData) exposed as
  fields (last time step)
- 2D slices of 3D meshes converted to polygons
- Native support of planar 2D meshes (the mesh is projected on the (u, v) plane)
- 3D display through ``Data3D.from_vtk``

Notes
-----
Only the highest-dimension elements of the mesh are displayed (tetrahedra/hexahedra
for a 3D mesh, triangles/quadrangles for a 2D mesh). Higher-order elements are
reduced to their linear (corner-node) counterpart.

Example
-------
>>> from scivianna.slave import ComputeSlave
>>> from scivianna.interface.gmsh_interface import GmshInterface
>>>
>>> slave = ComputeSlave(GmshInterface)
>>> slave.read_file("mesh.msh", "Geometry")

Dependencies
------------
pip install gmsh pyvista
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scivianna.interface

try:
    import gmsh
    import pyvista as pv
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The Gmsh interface requires the 'gmsh' and 'pyvista' packages. "
        "Install them with: pip install gmsh pyvista"
    ) from e

import scivianna
from scivianna.constants import GEOMETRY, MATERIAL, MESH
from scivianna.data.data2d import Data2D
from scivianna.data.data3d import Data3D
from scivianna.enums import GeometryType, VisualizationMode
from scivianna.interface.generic_interface import Geometry2DPolygon, Geometry3D
from scivianna.logging_config import get_logger
from scivianna.utils.polygonize_tools import PolygonCoords, PolygonElement

logger = get_logger(__name__)

ENTITY = "Entity"
"""Field name of the geometrical entity a cell belongs to."""

SIZE = "Size"
"""Field name of the cell length / area / volume."""

# gmsh element type -> (vtk cell type, number of corner nodes to keep).
# Corner nodes always come first in the gmsh node ordering, so higher order
# elements are reduced to their linear counterpart by keeping the first nodes.
_GMSH_TO_VTK: Dict[int, Tuple[int, int]] = {
    1: (3, 2),  # 2-node line
    2: (5, 3),  # 3-node triangle
    3: (9, 4),  # 4-node quadrangle
    4: (10, 4),  # 4-node tetrahedron
    5: (12, 8),  # 8-node hexahedron
    6: (13, 6),  # 6-node prism
    7: (14, 5),  # 5-node pyramid
    8: (3, 2),  # 3-node line
    9: (5, 3),  # 6-node triangle
    10: (9, 4),  # 9-node quadrangle
    11: (10, 4),  # 10-node tetrahedron
    12: (12, 8),  # 27-node hexahedron
    13: (13, 6),  # 18-node prism
    14: (14, 5),  # 14-node pyramid
    15: (1, 1),  # point
    16: (9, 4),  # 8-node quadrangle
    17: (12, 8),  # 20-node hexahedron
    18: (13, 6),  # 15-node prism
    19: (14, 5),  # 13-node pyramid
}



def _init_gmsh() -> None:
    """Initializes gmsh without signal handlers (required outside of the main thread)."""
    if gmsh.isInitialized():
        return
    try:
        gmsh.initialize(interruptible=False)
    except TypeError:  # older gmsh versions
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)


class GmshInterface(Geometry2DPolygon, Geometry3D):
    """
    Gmsh mesh interface for Scivianna.

    Provides 2D slice (polygons) and 3D visualization of gmsh meshes.
    """

    geometry_type = GeometryType._3D_INFINITE

    mesh_size: Optional[float] = None
    """Maximum mesh size used when a CAD / .geo file has to be meshed (None : gmsh default)."""

    def __init__(self) -> None:
        self.mesh: Optional[pv.UnstructuredGrid] = None
        """Displayed mesh (highest-dimension elements only)."""

        self.dimension: int = 0
        """Dimension of the displayed elements."""

        self.planar_normal: Optional[np.ndarray] = None
        """Normal of the mesh plane if the displayed mesh is a planar surface mesh."""

        self.material_names: List[str] = []
        """Physical group names, indexed by the values of the material id array."""

        self.entity_names: List[str] = []
        """Entity names, indexed by the values of the entity id array."""

        self.view_names: List[str] = []
        """Names of the post-processing fields stored in the mesh cell data."""

        self.data: Dict[str, Data2D] = {}
        """Cache of computed 2D data per caller."""

        self.last_computed_frame: Dict[str, List[float]] = {}
        """Cache keys of the last computed 2D frame per caller."""

        self.last_3d_frame: Optional[Dict[str, Any]] = None
        """Cache key of the last 3D computation."""

        self.file_infos: List[Tuple[str, str]] = []
        """List of (file_path, file_label) loaded by this interface."""

    # ------------------------------------------------------------------ #
    #                            File reading                            #
    # ------------------------------------------------------------------ #
    def read_file(self, file_path: Union[str, Path], file_label: str) -> None:
        """
        Reads a gmsh compatible file and builds the PyVista mesh.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the file (.msh, .geo, .step, .brep, ...)
        file_label : str
            File label, only "Geometry" is supported.

        Raises
        ------
        FileNotFoundError
            The file does not exist
        NotImplementedError
            Unsupported file label
        ValueError
            The file contains no mesh element
        """
        if file_label != GEOMETRY:
            raise NotImplementedError(
                f"File label '{file_label}' not supported by GmshInterface. Use '{GEOMETRY}'."
            )

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Gmsh file not found: {file_path}")

        logger.info("Reading gmsh file %s", file_path)

        _init_gmsh()
        try:
            gmsh.clear()
            gmsh.open(str(file_path))

            if self._model_has_no_elements():
                self._generate_mesh()

            self._extract_mesh()
        finally:
            gmsh.finalize()

        self.data.clear()
        self.last_computed_frame.clear()
        self.last_3d_frame = None
        self.file_infos = [(str(file_path), file_label)]

    def get_options_dictionnary(self) -> dict[str, Any]:
        """Returns a current interface state option dictionnary.

        Returns
        -------
        dict[str, Any]
            Option dictionnary to provide to the interface functions
        """
        return {}

    @staticmethod
    def _model_has_no_elements() -> bool:
        """Returns whether the current gmsh model contains no mesh element."""
        types, _, _ = gmsh.model.mesh.getElements()
        return len(types) == 0

    def _generate_mesh(self) -> None:
        """Meshes the current gmsh model (3D if it has volumes, 2D otherwise)."""
        if self.mesh_size is not None:
            gmsh.option.setNumber("Mesh.MeshSizeMax", float(self.mesh_size))

        dim = 3 if len(gmsh.model.getEntities(3)) > 0 else 2
        logger.info("No element found, generating a %dD mesh", dim)
        gmsh.model.mesh.generate(dim)

    def _extract_mesh(self) -> None:
        """Converts the current gmsh model to a PyVista unstructured grid."""
        # ---- Nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        if len(node_tags) == 0:
            raise ValueError("The gmsh model contains no node.")

        node_tags = np.asarray(node_tags, dtype=np.int64)
        points = np.asarray(node_coords, dtype=float).reshape(-1, 3)
        tag_to_index = np.full(node_tags.max() + 1, -1, dtype=np.int64)
        tag_to_index[node_tags] = np.arange(len(node_tags))

        # ---- Elements, highest dimension only
        entities = [
            (dim, tag)
            for dim, tag in gmsh.model.getEntities()
            if len(gmsh.model.mesh.getElements(dim, tag)[0]) > 0
        ]
        if not entities:
            raise ValueError("The gmsh model contains no mesh element.")
        max_dim = max(dim for dim, _ in entities)
        entities = [(d, t) for d, t in entities if d == max_dim]

        # ---- Physical groups: (dim, entity tag) -> name
        entity_to_group: Dict[int, str] = {}
        for g_dim, g_tag in gmsh.model.getPhysicalGroups(max_dim):
            name = gmsh.model.getPhysicalName(g_dim, g_tag) or f"Group {g_tag}"
            for ent in gmsh.model.getEntitiesForPhysicalGroup(g_dim, g_tag):
                # Elements of several groups: the first one found is kept.
                entity_to_group.setdefault(int(ent), name)

        cells_blocks: List[np.ndarray] = []
        celltypes: List[np.ndarray] = []
        element_tags: List[np.ndarray] = []
        entity_ids: List[np.ndarray] = []
        material_ids: List[np.ndarray] = []

        material_names: List[str] = []
        entity_names: List[str] = []

        def _index(names: List[str], name: str) -> int:
            if name not in names:
                names.append(name)
            return names.index(name)

        for _, ent_tag in entities:
            types, e_tags, n_tags = gmsh.model.mesh.getElements(max_dim, ent_tag)

            ent_id = _index(entity_names, f"Entity {ent_tag}")
            mat_id = _index(material_names, entity_to_group.get(ent_tag, f"Entity {ent_tag}"))

            for e_type, e_t, n_t in zip(types, e_tags, n_tags):
                if int(e_type) not in _GMSH_TO_VTK:
                    logger.warning("Gmsh element type %d not supported, skipped", e_type)
                    continue
                vtk_type, n_keep = _GMSH_TO_VTK[int(e_type)]

                n_t = np.asarray(n_t, dtype=np.int64)
                e_t = np.asarray(e_t, dtype=np.int64)
                n_per_elem = len(n_t) // len(e_t)
                conn = tag_to_index[n_t].reshape(-1, n_per_elem)[:, :n_keep]

                cells_blocks.append(
                    np.hstack([np.full((len(conn), 1), n_keep, dtype=np.int64), conn]).ravel()
                )
                celltypes.append(np.full(len(conn), vtk_type, dtype=np.uint8))
                element_tags.append(e_t)
                entity_ids.append(np.full(len(conn), ent_id, dtype=np.int64))
                material_ids.append(np.full(len(conn), mat_id, dtype=np.int64))

        if not cells_blocks:
            raise ValueError("None of the mesh elements are supported.")

        mesh = pv.UnstructuredGrid(
            np.concatenate(cells_blocks), np.concatenate(celltypes), points
        )
        n_cells = mesh.n_cells

        mesh.cell_data["cell_id"] = np.arange(n_cells)
        mesh.cell_data["element_tag"] = np.concatenate(element_tags)
        mesh.cell_data["entity_id"] = np.concatenate(entity_ids)
        mesh.cell_data["material_id"] = np.concatenate(material_ids)

        # ---- Cell size
        sizes = mesh.compute_cell_sizes(
            length=max_dim == 1, area=max_dim == 2, volume=max_dim == 3
        )
        size_name = {1: "Length", 2: "Area", 3: "Volume"}.get(max_dim)
        if size_name is not None and size_name in sizes.cell_data:
            mesh.cell_data[SIZE] = np.asarray(sizes.cell_data[size_name])

        # ---- Post-processing views
        self.view_names = self._extract_views(mesh, node_tags, tag_to_index)

        self.mesh = mesh
        self.dimension = max_dim
        self.material_names = material_names
        self.entity_names = entity_names
        self.planar_normal = self._planar_normal(mesh) if max_dim == 2 else None

        logger.info(
            "Loaded gmsh mesh: %d cells of dimension %d, %d physical groups",
            n_cells,
            max_dim,
            len(material_names),
        )

    def _extract_views(
        self, mesh: pv.UnstructuredGrid, node_tags: np.ndarray, tag_to_index: np.ndarray
    ) -> List[str]:
        """Stores the NodeData / ElementData post-processing views as cell arrays (last time step)."""
        names: List[str] = []
        element_index = {int(t): i for i, t in enumerate(mesh.cell_data["element_tag"])}

        for position, view_tag in enumerate(gmsh.view.getTags()):
            try:
                name = gmsh.view.option.getString(view_tag, "Name") or f"View {view_tag}"
                n_steps = int(gmsh.view.option.getNumber(view_tag, "NbTimeStep"))
                data_type, tags, values, _, n_comp = gmsh.view.getModelData(
                    view_tag, max(n_steps - 1, 0)
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not read gmsh view %s: %s", view_tag, e)
                continue

            if data_type not in ("NodeData", "ElementData") or len(tags) == 0:
                logger.info("View '%s' of type '%s' skipped", name, data_type)
                continue

            arr = np.asarray([np.asarray(v, dtype=float) for v in values])
            arr = arr.reshape(len(tags), -1)
            scalar = arr[:, 0] if arr.shape[1] == 1 else np.linalg.norm(arr, axis=1)

            cell_values = np.full(mesh.n_cells, np.nan)
            if data_type == "ElementData":
                for tag, val in zip(tags, scalar):
                    idx = element_index.get(int(tag))
                    if idx is not None:
                        cell_values[idx] = val
            else:
                node_values = np.full(mesh.n_points, np.nan)
                valid = np.asarray(tags, dtype=np.int64) < len(tag_to_index)
                node_values[tag_to_index[np.asarray(tags, dtype=np.int64)[valid]]] = scalar[valid]
                mesh.point_data["_tmp"] = node_values
                cell_values = mesh.point_data_to_cell_data().cell_data["_tmp"]
                del mesh.point_data["_tmp"]

            unique = name
            while unique in mesh.cell_data or unique in names:
                unique += "_"
            mesh.cell_data[unique] = cell_values
            names.append(unique)

        return names

    @staticmethod
    def _planar_normal(mesh: pv.UnstructuredGrid, tol: float = 1e-9) -> Optional[np.ndarray]:
        """Returns the normal of the mesh plane if all the points are coplanar, None otherwise."""
        pts = np.asarray(mesh.points)
        if len(pts) > 20000:
            pts = pts[np.random.default_rng(0).choice(len(pts), 20000, replace=False)]
        centered = pts - pts.mean(axis=0)
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        if s[0] == 0 or s[-1] / s[0] > tol:
            return None
        return vt[-1] / np.linalg.norm(vt[-1])

    # ------------------------------------------------------------------ #
    #                              Labels                                #
    # ------------------------------------------------------------------ #
    def get_labels(self) -> List[str]:
        """Returns the displayable fields."""
        labels = [MESH, MATERIAL, ENTITY]
        if self.mesh is not None and SIZE in self.mesh.cell_data:
            labels.append(SIZE)
        labels.extend(self.view_names)
        return labels

    def get_label_coloring_mode(self, label: str) -> VisualizationMode:
        """Returns how the given field is colored."""
        if label == MESH:
            return VisualizationMode.NONE
        if label in (MATERIAL, ENTITY):
            return VisualizationMode.FROM_STRING
        return VisualizationMode.FROM_VALUE

    def get_file_input_list(self) -> List[Tuple[str, str]]:
        """Returns the supported file labels."""
        return [(GEOMETRY, "Gmsh file (.msh, .geo, .step, .brep, .stl, ...)")]

    # ------------------------------------------------------------------ #
    #                                2D                                  #
    # ------------------------------------------------------------------ #
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
        """
        Computes the polygons of a 2D slice of the mesh.

        - 3D mesh: the mesh is sliced by the plane defined by (u, v, origin).
        - Planar 2D mesh, viewed from its normal: the cells are directly projected
          on (u, v); the w coordinate of the origin is ignored.
        - Planar 2D mesh viewed from another direction: the slice only contains
          segments, a ValueError is raised.
        """
        if self.mesh is None:
            raise ValueError("No mesh loaded. Call read_file first.")

        cache_key = [*u, *v, *origin, size_u, size_v]
        if (
            caller in self.last_computed_frame
            and self.last_computed_frame[caller] == cache_key
            and caller in self.data
        ):
            return self.data[caller], False

        u_arr = np.array(u, dtype=float)
        v_arr = np.array(v, dtype=float)
        u_arr /= np.linalg.norm(u_arr)
        v_arr /= np.linalg.norm(v_arr)

        w_arr = np.cross(u_arr, v_arr)
        w_norm = np.linalg.norm(w_arr)
        if w_norm == 0.0:
            raise ValueError(f"Vectors u={u} and v={v} are parallel or zero.")
        w_arr /= w_norm

        origin_arr = np.array(origin, dtype=float)

        if self.planar_normal is not None and abs(np.dot(w_arr, self.planar_normal)) > 1 - 1e-6:
            polygons = self._project_cells(u_arr, v_arr)
        else:
            polygons = self._slice_to_polygons(u_arr, v_arr, w_arr, origin_arr)

        self.data[caller] = Data2D.from_polygon_list(polygons)
        self.last_computed_frame[caller] = cache_key
        return self.data[caller], True

    def _project_cells(self, u_arr: np.ndarray, v_arr: np.ndarray) -> List[PolygonElement]:
        """Projects every cell of a planar 2D mesh on the (u, v) plane."""
        mesh = self.mesh
        points = np.asarray(mesh.points)
        xs_all = points @ u_arr
        ys_all = points @ v_arr

        connectivity = np.asarray(mesh.cell_connectivity)
        offsets = np.asarray(
            mesh.cell_offsets if hasattr(mesh, "cell_offsets") else mesh.offset
        )
        cell_ids = np.asarray(mesh.cell_data["cell_id"])

        polygons: List[PolygonElement] = []
        for i in range(mesh.n_cells):
            ids = connectivity[offsets[i] : offsets[i + 1]]
            if len(ids) < 3:
                continue
            polygons.append(
                PolygonElement(
                    exterior_polygon=PolygonCoords(
                        x_coords=xs_all[ids].tolist(), y_coords=ys_all[ids].tolist()
                    ),
                    holes=[],
                    cell_id=int(cell_ids[i]),
                )
            )
        return polygons

    def _slice_to_polygons(
        self, u_arr: np.ndarray, v_arr: np.ndarray, w_arr: np.ndarray, origin_arr: np.ndarray
    ) -> List[PolygonElement]:
        """Slices the mesh with a plane and converts the result to polygons."""
        mesh_slice: pv.PolyData = self.mesh.slice(normal=w_arr, origin=origin_arr)

        if mesh_slice.n_cells == 0:
            raise ValueError(
                f"Slice at origin {origin_arr} with normal {w_arr} produced no cells. "
                f"Mesh bounds: {self.mesh.bounds}"
            )

        points = np.asarray(mesh_slice.points)
        xs_all = points @ u_arr
        ys_all = points @ v_arr
        cell_ids = np.asarray(mesh_slice.cell_data["cell_id"])

        # PolyData.faces: flat array [n, id_0, ..., id_n-1, n, ...]
        faces = np.asarray(mesh_slice.faces)
        polygons: List[PolygonElement] = []
        i = 0
        cell = 0
        while i < len(faces):
            n = int(faces[i])
            ids = faces[i + 1 : i + 1 + n]
            i += n + 1
            if n >= 3:
                polygons.append(
                    PolygonElement(
                        exterior_polygon=PolygonCoords(
                            x_coords=xs_all[ids].tolist(), y_coords=ys_all[ids].tolist()
                        ),
                        holes=[],
                        cell_id=int(cell_ids[cell]),
                    )
                )
            cell += 1

        if not polygons:
            raise ValueError(
                "The slice only contains lines or points (planar mesh viewed edge-on?)."
            )
        return polygons

    # ------------------------------------------------------------------ #
    #                              Values                                #
    # ------------------------------------------------------------------ #
    def get_value_dict(
        self,
        value_label: str,
        cells: List[Union[int, str]],
        options: Dict[str, Any],
        caller: str = "API",
    ) -> Dict[Union[int, str], Any]:
        """Returns a cell id - field value map for the requested field."""
        if self.mesh is None:
            raise ValueError("No mesh loaded. Call read_file first.")

        cell_data = self.mesh.cell_data

        if value_label == MESH:
            return {cell: np.nan for cell in cells}

        if value_label == MATERIAL:
            ids = cell_data["material_id"]
            return {c: self.material_names[int(ids[int(c)])] for c in cells}

        if value_label == ENTITY:
            ids = cell_data["entity_id"]
            return {c: self.entity_names[int(ids[int(c)])] for c in cells}

        if value_label in cell_data:
            values = cell_data[value_label]
            return {c: float(values[int(c)]) for c in cells}

        raise NotImplementedError(
            f"Field '{value_label}' not found. Available fields: {self.get_labels()}"
        )

    # ------------------------------------------------------------------ #
    #                                3D                                  #
    # ------------------------------------------------------------------ #
    def compute_3D_data(self, options: Dict[str, Any]) -> Tuple[Data3D, bool]:
        """Returns the mesh as a Data3D object, and whether it changed since the last call."""
        if self.mesh is None:
            raise ValueError("No mesh loaded. Call read_file first.")

        updated = self.last_3d_frame is None
        self.last_3d_frame = dict(options)
        return Data3D.from_vtk(self.mesh), updated

    def get_3d_value_dict(
        self,
        value_label: str,
        cells: List[Union[int, str]],
        options: Dict[str, Any],
        caller: str = "API",
    ) -> Dict[Union[int, str], Any]:
        """Returns a cell id - field value map for the requested field (3D display)."""
        return self.get_value_dict(value_label, cells, options, caller)

    # ------------------------------------------------------------------ #
    #                            Save / load                             #
    # ------------------------------------------------------------------ #
    def save(self, file_path: Path, include_files: bool) -> None:
        """
        Pickles the interface state.

        The converted mesh and the computed 2D polygons are always saved; the
        source file paths are only kept if ``include_files`` is True.
        """
        state = {
            "mesh": self.mesh,
            "dimension": self.dimension,
            "planar_normal": self.planar_normal,
            "material_names": self.material_names,
            "entity_names": self.entity_names,
            "view_names": self.view_names,
            "file_infos": self.file_infos if include_files else [],
            "data": self.data,
            "last_computed_frame": self.last_computed_frame,
        }
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(state, f)

    def load(self, file_path: Path, include_files: bool) -> None:
        """Restores a state saved with :meth:`save`."""
        with open(file_path, "rb") as f:
            state = pickle.load(f)

        self.mesh = state["mesh"]
        self.dimension = state["dimension"]
        self.planar_normal = state["planar_normal"]
        self.material_names = state["material_names"]
        self.entity_names = state["entity_names"]
        self.view_names = state["view_names"]
        self.file_infos = state["file_infos"] if include_files else []

        # Computed polygons are restored, the next identical 2D computation is skipped
        self.data = state["data"]
        self.last_computed_frame = state["last_computed_frame"]
        self.last_3d_frame = None

scivianna.interface.register_interface("GMSH", GmshInterface)
