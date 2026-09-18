import matplotlib.pyplot as plt
import numpy as np
import pytest

from scivianna.data.data2d import Data2D
from scivianna.plotter_2d.api import plot_frame_in_axes
from scivianna.slave import ComputeSlave
from scivianna.utils.polygonize_tools import PolygonCoords, PolygonElement

try:
    from scivianna.utils.extruded_mesh import ExtrudedStructuredMesh

except ImportError:
    from scivianna.interface.generic_interface import Geometry2D
    class ExtrudedStructuredMesh(Geometry2D):
        pass

@pytest.mark.medcoupling
def test_extruded_mesh(plot=False):
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

    # fig.savefig("test_extruded.png")

if __name__ == "__main__":
    test_extruded_mesh(plot=True)