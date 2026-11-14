"""Check that MEDCoupling 3D cells are converted to VTK with a valid node order.

MED and VTK use opposite orientation conventions for linear 3D cells, so the
conversion done in ``MEDInterface.compute_3D_data`` must permute the nodes:
a well oriented MED cell must give a positive VTK volume.
"""

import numpy as np
import pytest


def _single_cell_mesh(mc, mc_type, conn, coords):
    mesh = mc.MEDCouplingUMesh("mesh", 3)
    mesh.setCoords(mc.DataArrayDouble(np.array(coords, dtype=float)))
    mesh.allocateCells(1)
    mesh.insertNextCell(mc_type, conn)
    mesh.finishInsertingCells()
    return mesh


@pytest.mark.medcoupling
@pytest.mark.pyvista
def test_med_to_vtk_cell_orientation():
    """Each linear 3D MED cell type gives a positive volume once in VTK."""
    mc = pytest.importorskip("medcoupling")
    pv = pytest.importorskip("pyvista")
    from scivianna.constants import GEOMETRY
    from scivianna.interface.med_interface import MEDInterface

    cases = [
        (mc.NORM_TETRA4, pv.CellType.TETRA, 1.0 / 6.0,
         [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        (mc.NORM_PYRA5, pv.CellType.PYRAMID, 1.0 / 3.0,
         [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]]),
        (mc.NORM_PENTA6, pv.CellType.WEDGE, 0.5,
         [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]]),
        (mc.NORM_HEXA8, pv.CellType.HEXAHEDRON, 1.0,
         [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
          [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]),
    ]

    for mc_type, vtk_type, volume, coords in cases:
        mesh = _single_cell_mesh(mc, mc_type, list(range(len(coords))), coords)
        if mesh.getMeasureField(False).getArray().getIJ(0, 0) < 0:
            # Reverse the cell so that it is well oriented for MED
            conn = {
                mc.NORM_TETRA4: [0, 2, 1, 3],
                mc.NORM_PYRA5: [0, 3, 2, 1, 4],
                mc.NORM_PENTA6: [0, 2, 1, 3, 5, 4],
                mc.NORM_HEXA8: [0, 3, 2, 1, 4, 7, 6, 5],
            }[mc_type]
            mesh = _single_cell_mesh(mc, mc_type, conn, coords)
        assert mesh.getMeasureField(False).getArray().getIJ(0, 0) > 0

        field = mc.MEDCouplingFieldDouble(mc.ON_CELLS, mc.ONE_TIME)
        field.setName("cell_id")
        field.setMesh(mesh)
        field.setArray(mc.DataArrayDouble([0.0]))

        interface = MEDInterface()
        interface.read_file(field, GEOMETRY)
        data3d, _ = interface.compute_3D_data({})
        grid = data3d.polydata

        assert grid.celltypes[0] == vtk_type
        vtk_volume = grid.compute_cell_sizes()["Volume"][0]
        assert vtk_volume == pytest.approx(volume)


@pytest.mark.medcoupling
@pytest.mark.pyvista
def test_extruded_mesh_vtk_orientation():
    """The extruded mesh is made of VTK wedges, all with a positive volume."""
    pytest.importorskip("medcoupling")
    pv = pytest.importorskip("pyvista")
    from scivianna.utils.extruded_mesh import ExtrudedStructuredMesh
    from scivianna.utils.polygonize_tools import PolygonCoords, PolygonElement

    outer = PolygonCoords([0, 2, 2, 0], [0, 0, 2, 2])
    hole = PolygonCoords([0.5, 1.5, 1.5, 0.5], [0.5, 0.5, 1.5, 1.5])
    inner = PolygonCoords([0.5, 1.5, 1.5, 0.5], [0.5, 0.5, 1.5, 1.5])

    interface = ExtrudedStructuredMesh()
    interface.read_file("base_polygons", [
        PolygonElement(outer, [hole], 0),
        PolygonElement(inner, [], 1),
    ])
    interface.read_file("z_coords", [0.0, 1.0, 3.0])
    interface.read_file("extrusion_vector", (0, 0, 1))

    data3d, _ = interface.compute_3D_data({})
    grid = data3d.polydata

    assert np.all(grid.celltypes == pv.CellType.WEDGE)
    volumes = grid.compute_cell_sizes()["Volume"]
    assert np.all(volumes > 0)
    assert volumes.sum() == pytest.approx(2 * 2 * 3)
