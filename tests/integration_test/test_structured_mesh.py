
import math
import matplotlib.pyplot as plt
import numpy as np
import pytest

from scivianna.constants import X, Y
from scivianna.slave import ComputeSlave
from scivianna.plotter_2d.api import plot_frame_in_axes

try:
    from scivianna.utils.structured_mesh import CarthesianStructuredMesh, CylindricalStructuredMesh, SphericalStructuredMesh
    from scivianna.interface.structured_mesh_interface import StructuredMeshInterface
    
    class CarthesianInterface(StructuredMeshInterface):
        def read_file(self, file_path: str, file_label: str):
            """Read a file and store its content in the interface

            Parameters
            ----------
            file_path : str
                File to read
            file_label : str
                Label to define the file type
            """
            size = 7
            self.mesh = CarthesianStructuredMesh(
                np.linspace(0, 4, size),
                np.linspace(0, 4, size),
                np.linspace(0, 4, size),
            )
            self.mesh.set_values("id", np.arange(size*size*size).reshape(size, size, size))

    class SphericalInterface(StructuredMeshInterface):
        def read_file(self, file_path: str, file_label: str):
            """Read a file and store its content in the interface

            Parameters
            ----------
            file_path : str
                File to read
            file_label : str
                Label to define the file type
            """
            size = 7
            self.mesh = SphericalStructuredMesh(
                np.linspace(0, 4, size),
                np.linspace(0, math.pi*2, size),
                np.linspace(0, math.pi, size),
            )
            self.mesh.set_values("id", np.arange(size*size*size).reshape(size, size, size))

    class CylindricalInterface(StructuredMeshInterface):
        def read_file(self, file_path: str, file_label: str):
            """Read a file and store its content in the interface

            Parameters
            ----------
            file_path : str
                File to read
            file_label : str
                Label to define the file type
            """
            size = 7
            self.mesh = CylindricalStructuredMesh(
                np.linspace(0, 4, size),
                np.linspace(0, math.pi*2, size),
                np.linspace(0, 4, size),
            )
            self.mesh.set_values("id", np.arange(size*size*size).reshape(size, size, size))

except ImportError:
    class CarthesianInterface:
        pass
    class SphericalInterface:
        pass
    class CylindricalInterface:
        pass

@pytest.mark.pyvista
def test_plot_carthesian():
    """Test plotting a carthesian structured mesh
    """
    # Field example
    slave = ComputeSlave(CarthesianInterface)
    slave.read_file(
        None, None,
    )

    fig, axes = plt.subplots(1, 1, figsize=(8, 7))

    plot_frame_in_axes(
        slave,
        u=X,
        v=Y,
        origin=(2.0, 2.0, 2.0),
        size_u=4.0,
        size_v=4.0,
        coloring_label="id",
        color_map="viridis",
        display_colorbar=True,
        axes=axes,
    )

    slave.terminate()
    plt.close()

    assert True
@pytest.mark.pyvista
def test_plot_cylindrical():
    """Test plotting a cylindrical structured mesh
    """
    # Field example
    slave = ComputeSlave(CylindricalInterface)
    slave.read_file(
        None, None,
    )

    fig, axes = plt.subplots(1, 1, figsize=(8, 7))

    plot_frame_in_axes(
        slave,
        u=X,
        v=Y,
        origin=(2.0, 2.0, 2.0),
        size_u=4.0,
        size_v=4.0,
        coloring_label="id",
        color_map="viridis",
        display_colorbar=True,
        axes=axes,
    )

    slave.terminate()
    plt.close()

    assert True

@pytest.mark.pyvista
def test_plot_spherical():
    """Test plotting a spherical structured mesh
    """
    # Field example
    slave = ComputeSlave(SphericalInterface)
    slave.read_file(
        None, None,
    )

    fig, axes = plt.subplots(1, 1, figsize=(8, 7))

    plot_frame_in_axes(
        slave,
        u=X,
        v=Y,
        origin=(2.0, 2.0, 2.0),
        size_u=4.0,
        size_v=4.0,
        coloring_label="id",
        color_map="viridis",
        display_colorbar=True,
        axes=axes,
    )

    slave.terminate()
    plt.close()

    assert True

if __name__ == "__main__":
    print("Testing carthesian")
    test_plot_carthesian()

    print("Testing cylindrical")
    test_plot_cylindrical()

    print("Testing spherical")
    test_plot_spherical()
