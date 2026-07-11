import param
from panel.custom import JSComponent
import numpy as np
import pyvista as pv

# =============================================================================
# Binary helpers
# =============================================================================

def _pack(arr, dtype):
    return memoryview(arr.astype(dtype, copy=False)).tobytes()


def _pyvista_to_numpy(vtk_array):
    """Convert VTK array to numpy array (works with pyvista's underlying VTK arrays)"""
    import vtk.util.numpy_support as vtk_np
    return vtk_np.vtk_to_numpy(vtk_array)

# =============================================================================
# VTK 9.6+ SAFE CELL EXTRACTION
# =============================================================================

def extract_cell_stream(cell):
    """
    Convert VTK cell array (polys/lines/verts) into vtk.js-compatible stream.

    Uses modern VTK API:
    - GetOffsetsArray()
    - GetConnectivityArray()
    """

    if cell is None:
        return None

    offsets_vtk = cell.GetOffsetsArray()
    conn_vtk = cell.GetConnectivityArray()

    offsets = _pyvista_to_numpy(offsets_vtk)
    conn = _pyvista_to_numpy(conn_vtk)

    stream = []

    for i in range(len(offsets) - 1):

        start = offsets[i]
        end = offsets[i + 1]

        cell_pts = conn[start:end]

        stream.append(len(cell_pts))
        stream.extend(cell_pts.tolist())

    stream_np = np.array(stream, dtype=np.uint32)

    return {
        "buffer": memoryview(stream_np).tobytes(),
    }


def polydata_to_dict(poly):
    """
    Convert pyvista.PolyData → vtk.js-friendly binary structure
    
    Note: pyvista.PolyData is a subclass of vtkPolyData, so the API is compatible.
    """

    # -------------------------------------------------------------------------
    # POINTS
    # -------------------------------------------------------------------------

    pts = _pyvista_to_numpy(poly.GetPoints().GetData())

    points = {
        "buffer": _pack(pts, np.float32),
        "components": 3,
    }

    # -------------------------------------------------------------------------
    # TOPOLOGY
    # -------------------------------------------------------------------------

    def cell(cell):
        return extract_cell_stream(cell)

    polys = cell(poly.GetPolys())
    lines = cell(poly.GetLines())
    verts = cell(poly.GetVerts())
    strips = cell(poly.GetStrips())

    # -------------------------------------------------------------------------
    # POINT DATA
    # -------------------------------------------------------------------------

    point_data = {}
    pd = poly.GetPointData()

    for i in range(pd.GetNumberOfArrays()):

        arr = pd.GetArray(i)
        name = arr.GetName()

        np_arr = _pyvista_to_numpy(arr)

        point_data[name] = {
            "buffer": memoryview(np_arr).tobytes(),
            "components": arr.GetNumberOfComponents(),
            "dtype": str(np_arr.dtype),
        }

    # -------------------------------------------------------------------------
    # CELL DATA
    # -------------------------------------------------------------------------

    cell_data = {}
    cd = poly.GetCellData()

    for i in range(cd.GetNumberOfArrays()):

        arr = cd.GetArray(i)
        name = arr.GetName()

        np_arr = _pyvista_to_numpy(arr)

        cell_data[name] = {
            "buffer": _pack(np_arr, np.float32),
            "components": arr.GetNumberOfComponents(),
        }

    return {
        "points": points,

        "polys": polys,
        "lines": lines,
        "verts": verts,
        "strips": strips,

        "pointData": point_data,
        "cellData": cell_data,
    }


def _convert_cells_to_polys(poly):
    """
    Convert cell-based PolyData to polygon surface.
    
    For structured grids and other datasets that don't have explicit polys,
    we need to extract the surface or generate polygons from cells.
    """
    # Check if we already have polys
    if poly.GetPolys().GetNumberOfCells() > 0:
        return poly
    
    # Try to extract surface which will create proper polys
    try:
        return poly.extract_surface(algorithm='dataset_surface')
    except Exception:
        # If extract_surface fails, return original
        return poly


def unstructured_grid_to_dict(ugrid: pv.UnstructuredGrid):
    """
    Convert pyvista.UnstructuredGrid → vtk.js-friendly binary structure
    
    For vtk.js compatibility, we convert the unstructured grid to PolyData
    using extract_geometry() which preserves cell data.
    
    Uses vtkDataSetSurfaceFilter with original cell IDs to properly map cell data.
    """
    import vtk
    
    # Use vtkDataSetSurfaceFilter which preserves original cell IDs
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(ugrid)
    surface_filter.Update()
    
    poly = pv.PolyData(surface_filter.GetOutput())
    
    # The filter adds 'vtkOriginalCellIds' array to point data
    # We need to use it to map cell data from original to new cells
    if 'vtkOriginalCellIds' in poly.cell_data:
        original_ids = poly.cell_data['vtkOriginalCellIds'].astype(int)
        
        # Map each cell data array from original to new cells
        for name, arr in ugrid.cell_data.items():
            if len(original_ids) == len(arr):
                # Same number of cells, direct copy
                poly.cell_data[name] = arr
            else:
                # Different number of cells, use original IDs to map
                if len(original_ids) <= len(arr):
                    new_arr = arr[original_ids]
                    poly.cell_data[name] = new_arr
    
    return polydata_to_dict(poly)


class VTKPlotter(JSComponent):

    geometry = param.Dict()
    colors = param.Dict()

    info = param.Boolean(default=True, doc="Whether to show the info panel.")

    hover_cell_id = param.Integer(default=-1)
    hover_cell_value = param.Integer(default=-1)

    hover_position = param.List(default=[float("nan"), float("nan"), float("nan")])

    # Clip plane parameters
    clip_enabled = param.Boolean(default=False, doc="Enable/disable clip plane visualization")
    clip_origin = param.List(default=[0.0, 0.0, 0.0], doc="Clip plane origin [x, y, z]")
    clip_normal = param.List(default=[0.0, 0.0, 1.0], doc="Clip plane normal [x, y, z]")
    
    plane_visible = param.Boolean(default=False, doc="Plane visualization visible")

    _esm = "./dist/VTKPlotter.bundle.js"

    def __init__(self, **params):
        super().__init__(**params)

    # -------------------------------------------------------------------------
    # Clip Plane Control Methods
    # -------------------------------------------------------------------------

    def set_plane_enabled(self, enabled: bool):
        """Enable or disable plane visualization."""
        self.plane_visible = enabled

    def set_clip_enabled(self, enabled: bool):
        """Enable or disable clip plane visualization."""
        self.clip_enabled = enabled

    def set_clip_plane(self, origin=None, normal=None):
        """
        Set clip plane position and orientation.

        Parameters
        ----------
        origin : list of 3 floats, optional
            Plane origin [x, y, z]. If None, keeps current origin.
        normal : list of 3 floats, optional
            Plane normal [x, y, z]. If None, keeps current normal.
        """
        if origin is not None:
            self.clip_origin = list(origin)
        if normal is not None:
            self.clip_normal = list(normal)

    def set_clip_axis(self, axis: str, sign: int = 1):
        """
        Set clip plane normal to a cardinal direction.

        Parameters
        ----------
        axis : {'x', 'y', 'z'}
            Axis for the normal direction.
        sign : {1, -1}
            Direction sign.
        """
        normals = {
            'x': [sign, 0, 0],
            'y': [0, sign, 0],
            'z': [0, 0, sign],
        }
        self.clip_normal = normals.get(axis, [0, 0, sign])

    @property
    def clip_plane_state(self) -> dict:
        """
        Get current clip plane state.

        Returns
        -------
        dict
            Dictionary with 'enabled', 'origin', and 'normal' keys.
        """
        return {
            'enabled': self.clip_enabled,
            'origin': self.clip_origin,
            'normal': self.clip_normal,
        }

    @property
    def clip_center(self) -> list:
        """Get clip plane center (origin)."""
        return self.clip_origin

    @property
    def clip_axes(self) -> list:
        """Get clip plane normal vector."""
        return self.clip_normal

    def _convert_mesh(self, mesh):
        """
        Convert various pyvista mesh types to vtk.js format.
        
        Supports:
        - pv.PolyData
        - pv.UnstructuredGrid
        - pv.StructuredGrid (converted to PolyData)
        - pv.RectilinearGrid (converted to PolyData)
        - pv.ImageData (converted to PolyData)
        """
        if isinstance(mesh, pv.PolyData):
            # Ensure we have polys for proper rendering
            if mesh.GetPolys().GetNumberOfCells() == 0 and hasattr(mesh, 'extract_surface'):
                mesh = _convert_cells_to_polys(mesh)
            return polydata_to_dict(mesh)
        elif isinstance(mesh, pv.UnstructuredGrid):
            return unstructured_grid_to_dict(mesh)
        elif isinstance(mesh, (pv.StructuredGrid, pv.RectilinearGrid, pv.ImageData)):
            # Convert to PolyData using extract_surface for proper polygon generation
            poly = mesh.extract_surface(algorithm='dataset_surface')
            return polydata_to_dict(poly)
        else:
            raise TypeError(f"Unsupported mesh type: {type(mesh)}")

    def update_polydata(self, polydata):
        d = self._convert_mesh(polydata)

        self.geometry = {
            "points": d["points"],
            "polys": d["polys"],
            "lines": d["lines"],
            "verts": d["verts"],
            "strips": d["strips"],
        }

        self.colors = {
            "pointData": d["pointData"],
            "cellData": d["cellData"],
        }
    
    def update_colors(self, polydata):
        d = self._convert_mesh(polydata)

        self.colors = {
            "pointData": d["pointData"],
            "cellData": d["cellData"],
        }