import pytest

import numpy as np

from scivianna.constants import XS, YS
from scivianna.panel.panel_2d import Panel2D

from scivianna_example import demo
from scivianna_example.mandelbrot import mandelbrot

import scivianna.utils

scivianna.utils._testing = True


@pytest.mark.default
def test_demo():
    _, slaves = demo.make_demo(return_slaves = True)

    for slave in slaves:
        slave.terminate()

@pytest.mark.default
def test_mandelbrot():
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


if __name__ == "__main__":
    test_mandelbrot()
