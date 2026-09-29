from pathlib import Path

import matplotlib.pyplot as plt
import pytest

import scivianna
from scivianna.constants import GEOMETRY, X, Y
from scivianna.plotter_2d.api import plot_frame_in_axes
from scivianna.slave import ComputeSlave

from scivianna.logging_config import get_logger

@pytest.mark.medcoupling
def test_plot_medcoupling_from_memory():
    import medcoupling
    from scivianna.interface.med_interface import MEDInterface

    file_path = str(Path(scivianna.__file__).parent / "input_file" / "power.med")

    logger = get_logger("test_med")

    meshnames = medcoupling.GetMeshNames(file_path)
    fieldnames = medcoupling.GetAllFieldNamesOnMesh(file_path, meshnames[0])

    logger.info(meshnames)
    logger.info(fieldnames)

    field: medcoupling.MEDCouplingFieldDouble = medcoupling.ReadField(
        medcoupling.ON_CELLS,
        file_path,
        meshnames[0],
        0,
        fieldnames[0],
        -1,
        -1,
    )

    # Field example
    slave = ComputeSlave(MEDInterface)
    slave.read_file(
        field,
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

    if False:
        med_path = Path(".") / "test_med.png"
        fig.savefig(med_path)
        print(f"Saved med at {med_path}")

    slave.terminate()

if __name__ == "__main__":
    test_plot_medcoupling_from_memory()