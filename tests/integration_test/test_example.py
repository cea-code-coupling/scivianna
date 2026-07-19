import math
from typing import TYPE_CHECKING

import pytest

import numpy as np

from scivianna.constants import XS, YS

from scivianna_example import demo
from scivianna_example.mandelbrot import mandelbrot
from scivianna_example.med import split_item_example

if TYPE_CHECKING:
    from scivianna.panel.panel_2d import Panel2D
    from scivianna.panel.panel_3d import Panel3D

import scivianna.utils

scivianna.utils._testing = True


@pytest.mark.default
def test_demo():
    _, slaves = demo.make_demo(return_slaves = True)

    for slave in slaves:
        slave.terminate()

# Unmarked as default because it might fail on github but still works locally
# @pytest.mark.default
def test_mandelbrot():
    """Ensuring that the mandelbrot example works as expected, and that the range update works correctly."""
    def get_polygon_bounds(data):
        xs = np.concatenate([e for sublist in data[XS] for item in sublist for e in item])
        ys = np.concatenate([e for sublist in data[YS] for item in sublist for e in item])

        return (xs.min(), xs.max()), (ys.min(), ys.max())

    from bokeh.events import RangesUpdate
    layout, slaves = mandelbrot.make_panel(None, return_slaves = True)

    polygon_panel: Panel2D = layout.visualisation_panels["Mandelbrot polygons"]
    grid_panel: Panel2D = layout.visualisation_panels["Mandelbrot grid"]

    np.testing.assert_almost_equal(
        get_polygon_bounds(polygon_panel.plotter.source_polygons.data),
        ((-0.491, 0.509), (-0.491, 0.509))
    )
    polygon_panel.plotter.update_range(RangesUpdate(model = None, x0 = 1., y0 = 1., x1 = 2.5, y1 = 2.5))
    np.testing.assert_almost_equal(
        get_polygon_bounds(polygon_panel.plotter.source_polygons.data),
        ((0.9985, 2.4985), (0.9985, 2.4985))
    )

    assert grid_panel.plotter.image.glyph.x == -.49
    assert grid_panel.plotter.image.glyph.y == -.49
    assert grid_panel.plotter.image.glyph.dw == 1.
    assert grid_panel.plotter.image.glyph.dh == 1.

    grid_panel.plotter.update_range(RangesUpdate(model = None, x0 = 1., y0 = 1., x1 = 2.5, y1 = 2.5))

    assert grid_panel.plotter.image.glyph.x == 1.
    assert grid_panel.plotter.image.glyph.y == 1.
    assert grid_panel.plotter.image.glyph.dw == 1.5
    assert grid_panel.plotter.image.glyph.dh == 1.5

    for slave in slaves:
        slave.terminate()

# Unmarked as default because it might fail on github but still works locally
# @pytest.mark.default
def test_split_item():
    """Ensuring that the split item example works as expected, every plot update on tap."""
    layout, slaves = split_item_example.get_panel(None, return_slaves = True)

    xy_panel: Panel2D = layout.visualisation_panels["MEDCoupling visualizer XY"]
    xz_panel: Panel2D = layout.visualisation_panels["MEDCoupling visualizer XZ"]
    yz_panel: Panel2D = layout.visualisation_panels["MEDCoupling visualizer YZ"]

    xy_panel.plotter.source_mouse.data.update(x=[.5], y=[.5], z=[.5])

    for callback in xy_panel.plotter.figure._event_callbacks["tap"]:
        callback()

    assert xz_panel.plotter.source_coordinates.data["w"][0] == pytest.approx(-.5, rel = 1e-2), \
        f"Expected {xz_panel.plotter.source_coordinates.data['w'][0]} to be approximately 0.5"
    assert yz_panel.plotter.source_coordinates.data["w"][0] == pytest.approx(.5, rel = 1e-2), \
        f"Expected {yz_panel.plotter.source_coordinates.data['w'][0]} to be approximately 0.5"

    xz_panel.plotter.source_mouse.data.update(x=[-.3], y=[-.3], z=[-.3])

    for callback in xz_panel.plotter.figure._event_callbacks["tap"]:
        callback()

    assert xy_panel.plotter.source_coordinates.data["w"][0] == pytest.approx(-.3, rel = 1e-2), \
        f"Expected {xy_panel.plotter.source_coordinates.data['w'][0]} to be approximately -0.3"
    assert yz_panel.plotter.source_coordinates.data["w"][0] == pytest.approx(-.3, rel = 1e-2), \
        f"Expected {yz_panel.plotter.source_coordinates.data['w'][0]} to be approximately 0.3"

    for slave in slaves:
        slave.terminate()

@pytest.mark.pyvista
def test_demo_3d():
    """Ensuring that 2D and 3D plots interact as expected."""
    from scivianna_example.med import demo_3d
    layout, slaves = demo_3d.get_panel(None, return_slaves = True)

    panel_3d: Panel3D = layout.visualisation_panels["3D Demo"]
    panel_2d: Panel2D = layout.visualisation_panels["MEDCoupling slice"]

    panel_3d.plotter.move_slice_to(
        origin=(0.5, 0.5, 0.5),
    )

    assert panel_2d.plotter.source_coordinates.data["w"][0] == pytest.approx(0.5, rel = 1e-2), \
        f"Expected {panel_2d.plotter.source_coordinates.data['w'][0]} to be approximately 0.5"

    panel_3d.plotter.move_slice_to(
        u=(math.sqrt(2)/2, 0, math.sqrt(2)/2),
        v=(-math.sqrt(2)/2, math.sqrt(2)/2, 0)
    )

    for slave in slaves:
        slave.terminate()


if __name__ == "__main__":
    test_mandelbrot()
    # test_mandelbrot()
    # test_demo_3d()
