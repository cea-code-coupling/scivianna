import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scivianna.constants import GEOMETRY, MATERIAL, X, Y
from scivianna.plotter_2d.api import plot_frame_in_axes
from scivianna.slave import ComputeSlave
from scivianna.utils.file_cleaner import mark_for_deletion

gmsh = pytest.importorskip("gmsh")
pytest.importorskip("pyvista")

from scivianna.interface.gmsh_interface import GmshInterface  # noqa: E402


@pytest.fixture(scope="module")
def msh_3d(tmp_path_factory):
    """Two touching boxes, each in its own physical group, meshed in 3D."""
    path = tmp_path_factory.mktemp("gmsh") / "boxes_3d.msh"

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("boxes")
    box_1 = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    box_2 = gmsh.model.occ.addBox(1, 0, 0, 1, 1, 1)
    gmsh.model.occ.fragment([(3, box_1)], [(3, box_2)])
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [box_1], name="fuel")
    gmsh.model.addPhysicalGroup(3, [box_2], name="water")
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.4)
    gmsh.model.mesh.generate(3)
    gmsh.write(str(path))
    gmsh.finalize()

    return path


@pytest.fixture(scope="module")
def msh_2d(tmp_path_factory):
    """A rectangle meshed with quadrangles."""
    path = tmp_path_factory.mktemp("gmsh") / "plate_2d.msh"

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("plate")
    surface = gmsh.model.occ.addRectangle(0, 0, 0, 2, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [surface], name="plate")
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.2)
    gmsh.model.mesh.generate(2)
    gmsh.write(str(path))
    gmsh.finalize()

    return path


def test_plot_gmsh_3d(msh_3d, plot=False):
    """Test plotting a slice of a 3D gmsh mesh"""
    slave = ComputeSlave(GmshInterface)
    slave.read_file(msh_3d, GEOMETRY)

    fig, axes = plt.subplots(1, 1, figsize=(8, 5))
    plot_frame_in_axes(
        slave,
        u=X,
        v=Y,
        origin=(1.0, 0.5, 0.4),
        size_u=2.0,
        size_v=1.0,
        coloring_label=MATERIAL,
        axes=axes,
    )

    if plot:
        fig.savefig("gmsh_3d_xy.png")

    slave.terminate()
    plt.close()

    assert True


def test_plot_gmsh_2d(msh_2d, plot=False):
    """Test plotting a planar 2D gmsh mesh"""
    slave = ComputeSlave(GmshInterface)
    slave.read_file(msh_2d, GEOMETRY)

    fig, axes = plt.subplots(1, 1, figsize=(8, 4))
    plot_frame_in_axes(
        slave,
        u=X,
        v=Y,
        origin=(1.0, 0.5, 0.0),
        size_u=2.0,
        size_v=1.0,
        coloring_label="Size",
        color_map="viridis",
        display_colorbar=True,
        axes=axes,
    )

    if plot:
        fig.savefig("gmsh_2d_xy.png")

    slave.terminate()
    plt.close()

    assert True


def test_save_load_gmsh(msh_3d, tmp_path):
    """Test saving a gmsh slave and loading it in a new one"""
    slave = ComputeSlave(GmshInterface)
    slave.read_file(msh_3d, GEOMETRY)

    u, v = X, Y
    origin = (1.0, 0.5, 0.4)
    size_u, size_v = 2.0, 1.0

    data, computed = slave.compute_2D_data(
        u, v, origin, size_u, size_v, None, MATERIAL, {}, caller="Test"
    )
    assert computed, "First compute_2d_data should have been computed"

    dict1 = slave.get_value_dict(MATERIAL, data.cell_ids, {}, caller="Test")

    # Saving the current state (loaded mesh, computed polygons...)
    save_path = str(tmp_path / "gmsh_test.pkl")
    slave.save(save_path, True)
    mark_for_deletion(save_path)

    # Creating a new slave and loading the file
    slave2 = ComputeSlave(GmshInterface)
    slave2.load(save_path, True)

    # New compute_2D_data is now instant as the polygons were saved
    data2, computed = slave2.compute_2D_data(
        u, v, origin, size_u, size_v, None, MATERIAL, {}, caller="Test"
    )
    assert not computed, "Loaded compute_2d_data should have been skipped"

    dict2 = slave2.get_value_dict(MATERIAL, data2.cell_ids, {}, caller="Test")
    assert dict1 == dict2, "Returned cell value dictionnary doesn't match the first"

    slave.terminate()
    slave2.terminate()