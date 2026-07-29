from pathlib import Path

import numpy as np

import scivianna
from scivianna.constants import GEOMETRY, X, Y
from scivianna.enums import UpdateEvent
from scivianna.layout.split import SplitDirection, SplitItem, SplitLayout
from scivianna.panel.panel_2d import Panel2D
from scivianna.panel.panel_3d import Panel3D
from scivianna.panel.visualisation_panel import VisualizationPanel
from scivianna.slave import ComputeSlave
from scivianna.interface.vtk_interface import VTKInterface


def create_uniform_structured_grid(
    nx: int, ny: int, nz: int, spacing: float = 1.0, cmap: str = "viridis"
) -> "pv.UnstructuredGrid":
    """
    Create a uniform structured grid using pyvista.

    Generates a 3D structured grid with evenly spaced points and computes
    cell values based on normalized x-coordinates. The grid is colored
    using the specified colormap.

    Parameters
    ----------
    nx : int
        Number of points in the x-dimension.
    ny : int
        Number of points in the y-dimension.
    nz : int
        Number of points in the z-dimension.
    spacing : float, optional
        Total spacing/size of the grid. Default is 1.0.
    cmap : str, optional
        Colormap name for coloring cells. Default is "viridis".

    Returns
    -------
    pv.UnstructuredGrid
        A structured grid with cell_data containing 'cell_id', 'cell_value',
        and 'rgb' arrays.
    """
    try:
        import pyvista as pv
        from vtk.util import numpy_support as nps
    except ImportError:
        raise ImportError("PyVista and VTK are required. Install with: pip install scivianna[3d]")

    # Create coordinate arrays
    x = np.arange(nx, dtype=np.float64) * spacing / nx
    y = np.arange(ny, dtype=np.float64) * spacing / ny
    z = np.arange(nz, dtype=np.float64) * spacing / nz

    # Create meshgrid for the structured grid
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Create structured grid
    grid = pv.StructuredGrid(X, Y, Z)

    # Calculate cell values
    n_cells = (nx - 1) * (ny - 1) * (nz - 1)
    cell_id = np.arange(n_cells, dtype=np.float64)

    # Handle edge cases for normalization
    x_norm = np.arange(max(nx - 1, 1), dtype=np.float64)
    y_norm = np.arange(max(ny - 1, 1), dtype=np.float64)
    z_norm = np.arange(max(nz - 1, 1), dtype=np.float64)

    if nx > 1:
        x_norm = x_norm / (nx - 2) if nx > 2 else x_norm
    if ny > 1:
        y_norm = y_norm / (ny - 2) if ny > 2 else y_norm
    if nz > 1:
        z_norm = z_norm / (nz - 2) if nz > 2 else y_norm

    X_c, Y_c, Z_c = np.meshgrid(x_norm, y_norm, z_norm, indexing="ij")
    cell_value = X_c.flatten()

    # Ensure cell data arrays match the number of cells
    if len(cell_value) != n_cells:
        cell_value = cell_value[:n_cells]
        if len(cell_value) < n_cells:
            cell_value = np.pad(cell_value, (0, n_cells - len(cell_value)))

    grid.cell_data["cell_id"] = cell_id
    grid.cell_data["cell_value"] = cell_value

    # Convert to unstructured grid and apply colormap
    unstructured = grid.cast_to_unstructured_grid()

    # Apply colormap to cell data
    try:
        colormap = pv.LookupTable(cmap=cmap)
        # Normalize cell values to [0, 1]
        min_val, max_val = cell_value.min(), cell_value.max()
        if max_val > min_val:
            normalized = (cell_value - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(cell_value)

        # Get RGB colors from colormap
        colors = np.zeros((len(normalized), 4), dtype=np.uint8)
        for i, val in enumerate(normalized):
            rgb = colormap.GetColor(val)
            colors[i, :3] = [int(c * 255) for c in rgb[:3]]
            colors[i, 3] = 255  # Alpha

        unstructured.cell_data["rgb"] = colors
    except Exception as e:
        # Fallback to white color if colormap fails
        unstructured.cell_data["rgb"] = np.ones((n_cells, 4), dtype=np.uint8) * 255

    return unstructured


def get_panel(
    geo, title: str = "VTK 3D", displayed_field: str = "cell_value", *args, return_slaves=False, **kwargs
) -> VisualizationPanel:
    """
    Create a VTK 3D visualization panel.

    Parameters
    ----------
    geo : str, Path, or None
        Path to VTK file, or None to use default generated grid
    title : str, optional
        Panel title. Default is "VTK 3D".
    displayed_field : str, optional
        Field to display. Default is "cell_value".
    return_slaves : bool, optional
        If True, return the slave along with the panel. Default is False.

    Returns
    -------
    VisualizationPanel or Tuple[VisualizationPanel, list]
        Visualization panel or tuple of (panel, slaves) if return_slaves is True
    """
    slave = ComputeSlave(VTKInterface)

    if geo is None:
        # Create default structured grid
        try:
            import pyvista as pv

            grid = create_uniform_structured_grid(nx=10, ny=10, nz=10, spacing=10.0)

            # Save to temporary PVD file (collection with single timestep)
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                # Save as VTU file
                vtu_path = Path(tmpdir) / "grid.vtu"
                grid.save(str(vtu_path))

                # Create PVD collection file
                pvd_path = Path(tmpdir) / "grid.pvd"
                pvd_content = """<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1" byte_order="LittleEndian" compressor="vtkZLibDataCompressor">
  <Collection>
    <DataSet group="" part="0" file="grid.vtu" timestep="0.0"/>
  </Collection>
</VTKFile>
"""
                pvd_path.write_text(pvd_content)

                # Load the PVD file
                slave.read_file(str(pvd_path), GEOMETRY)

        except ImportError:
            raise ImportError("PyVista is required for VTK examples. Install with: pip install scivianna[3d]")

    elif isinstance(geo, (str, Path)):
        slave.read_file(geo, GEOMETRY)
    else:
        raise TypeError(f"Provided type {type(geo)} not implemented")

    # Check if requested field exists
    labels = slave.get_labels()
    if displayed_field not in labels:
        displayed_field = labels[0] if labels else None

    # Create 2D panel
    vtk_2d = Panel2D(slave, name="VTK slice", u=X, v=Y, displayed_field=displayed_field)
    vtk_2d.update_event = [UpdateEvent.CLIC, UpdateEvent.AXES_CHANGE]

    # Create 3D panel
    vtk_panel_3d = Panel3D(slave, name="VTK 3D Demo", displayed_field=displayed_field)
    vtk_panel_3d.update_event = [UpdateEvent.CLIC, UpdateEvent.AXES_CHANGE]

    # Create split layout
    split = SplitItem(vtk_panel_3d, vtk_2d, SplitDirection.VERTICAL)

    if return_slaves:
        return SplitLayout(split), [vtk_panel_3d.get_slave()]
    else:
        return SplitLayout(split)


if __name__ == "__main__":
    # Run the demo
    get_panel(None).show()
