import matplotlib.pyplot as plt
import numpy as np
import pytest

from scivianna.data.data2d import Data2D
from scivianna.plotter_2d.polygon.matplotlib import Matplotlib2DPolygonPlotter
from scivianna.utils.color_tools import get_edges_colors, interpolate_cmap_at_values
from scivianna.utils.polygonize_tools import PolygonCoords, PolygonElement

try:
    from scivianna.utils.extruded_mesh import ExtrudedStructuredMesh
    
except ImportError:
    from scivianna.interface.generic_interface import Geometry2D
    class ExtrudedStructuredMesh(Geometry2D):
        pass

@pytest.mark.pyvista
def test_extruded_mesh():
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


    mesh = ExtrudedStructuredMesh(
        [p0, p1, p2], list(range(5))
    )
    mesh.set_values("id", {i:i for i in range(4*3)})

    polygons = mesh.compute_2D_slice((1., 1., 0.5), (1, 0, 0), (0, 1, 0))

    data = Data2D.from_polygon_list(polygons)

    data.cell_values = mesh.get_cells_values("id", [p.cell_id for p in polygons])
    data.cell_colors = interpolate_cmap_at_values("viridis", np.array(data.cell_values)/(4*3))
    data.cell_edge_colors = get_edges_colors(data.cell_colors)
    
    plotter = Matplotlib2DPolygonPlotter()
    plotter.plot_2d_frame(data)
    # plotter.figure.savefig("test_extruded_0.png")
    plt.close()

    polygons = mesh.compute_2D_slice((1., 1., 0.5), (1, 0, 0), (0, 0, 1))
    data = Data2D.from_polygon_list(polygons)

    data.cell_values = mesh.get_cells_values("id", [p.cell_id for p in polygons])
    data.cell_colors = interpolate_cmap_at_values("viridis", np.array(data.cell_values)/(4*3))
    data.cell_edge_colors = get_edges_colors(data.cell_colors)

    plotter = Matplotlib2DPolygonPlotter()
    plotter.plot_2d_frame(data)
    # plotter.figure.savefig("test_extruded_1.png")
    plt.close()
