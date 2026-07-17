from pathlib import Path
import matplotlib.pyplot as plt

import pytest
import scivianna
from scivianna.constants import GEOMETRY, X, Y
from scivianna.slave import ComputeSlave
from scivianna.plotter_2d.api import plot_frame_in_axes

from scivianna.interface.med_interface import MEDInterface


@pytest.mark.default
def test_plot_polygons():
    """Simple test to make sure things happen before more tests are actually implemented
    """

    # Field example
    slave = ComputeSlave(MEDInterface)
    slave.read_file(
        Path(scivianna.__file__).parent / "input_file" / "power.med",
        GEOMETRY,
    )

    fig, axes = plt.subplots(1, 1, figsize=(8, 7))

    plot_frame_in_axes(
        slave,
        u=X,
        v=Y,
        origin=(0.0, 0.0, 0.0),
        size_u=20.0,
        size_v=20.0,
        coloring_label="INTEGRATED_POWER",
        color_map="viridis",
        display_colorbar=True,
        axes=axes,
    )

    slave.terminate()
