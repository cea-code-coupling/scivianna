
import matplotlib.pyplot as plt

import pytest
from scivianna.constants import MATERIAL, X, Y
from scivianna.slave import ComputeSlave
from scivianna.plotter_2d.api import plot_frame_in_axes

from scivianna_example.mandelbrot.mandelbrot import MandelBrotInterface


@pytest.mark.grid
def test_plot_grid():
    """Simple test to make sure things happen before more tests are actually implemented
    """

    # Field example
    slave = ComputeSlave(MandelBrotInterface)

    fig, axes = plt.subplots(1, 1, figsize=(8, 7))

    plot_frame_in_axes(
        slave,
        u=X,
        v=Y,
        origin=(0.0, 0.0, 0.0),
        size_u=2.0,
        size_v=2.0,
        coloring_label=MATERIAL,
        color_map="viridis",
        axes=axes,
        options = {
            "Max iter":20,
            "u_steps":50,
            "v_steps":50
        }
    )

    slave.terminate()

if __name__ == "__main__":
    test_plot_grid()