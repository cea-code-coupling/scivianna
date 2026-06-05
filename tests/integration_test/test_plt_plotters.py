
# Field example
from pathlib import Path
import pytest
import scivianna
from scivianna.constants import GEOMETRY, MATERIAL, X, Y
from scivianna.interface.med_interface import MEDInterface
from scivianna.data.data2d import Data2D
from scivianna.extension.field_selector import set_colors_list
from scivianna.plotter_2d.polygon.matplotlib import Matplotlib2DPolygonPlotter
from scivianna.plotter_2d.grid.matplotlib import Matplotlib2DGridPlotter

from scivianna_example.mandelbrot.mandelbrot import MandelBrotInterface

def build_data_polygon() -> Data2D:
    med = MEDInterface()
    med.read_file(
        str(Path(scivianna.__file__).parent / "input_file" / "power.med"),
        GEOMETRY,
    )
    data_2d:Data2D
    data_2d, _ = med.compute_2D_data(
        X,
        Y, 
        0, 1, 0, 1, # values not used
        0., 
        None,
        {"u_steps": 50, "v_steps": 50},
        caller="Test"
    )
    set_colors_list(data_2d, med, "INTEGRATED_POWER", "viridis", False, {})

    return data_2d

def build_data_grid() -> Data2D:
    mandelbrot = MandelBrotInterface()
    data_2d:Data2D
    data_2d, _ = mandelbrot.compute_2D_data(
        X,
        Y, 
        0, 1, 
        0, 1, 
        0., 
        None,
        {"u_steps": 50, "v_steps": 50},
        caller="Test"
    )
    set_colors_list(data_2d, mandelbrot, MATERIAL, "viridis", False, {})

    return data_2d

@pytest.mark.default
def test_polygon_plt_from_polygons():
    data_2d = build_data_polygon()

    plotter = Matplotlib2DPolygonPlotter()
    plotter.plot_2d_frame(data_2d)
    plotter.figure.savefig("p2p.png")

@pytest.mark.default
def test_grid_plt_from_polygons():
    data_2d = build_data_grid()

    plotter = Matplotlib2DPolygonPlotter()
    plotter.plot_2d_frame(data_2d)
    plotter.figure.savefig("g2p.png")

@pytest.mark.default
def test_grid_plt_from_grid():
    data_2d = build_data_grid()

    plotter = Matplotlib2DGridPlotter()
    plotter.plot_2d_frame(data_2d)
    plotter.figure.savefig("g2g.png")
