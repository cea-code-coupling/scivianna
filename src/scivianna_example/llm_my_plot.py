from pathlib import Path

# Field example
import scivianna
from scivianna.agent import data_2d_worker
from scivianna.constants import GEOMETRY, X, Y
from scivianna.interface.med_interface import MEDInterface
from scivianna.plotter_2d.polygon.matplotlib import Matplotlib2DPolygonPlotter
from scivianna.extension.field_selector import set_colors_list
from scivianna.data.data2d import Data2D
from scivianna.agent.data_2d_worker import Data2DWorker

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
    {}
)
set_colors_list(data_2d, med, "INTEGRATED_POWER", "viridis", False, {})


worker = Data2DWorker(data_2d)

worker("hide zeros")

data_2d = worker.data2d


"""     data_2d est maintenant parfait
"""


plotter = Matplotlib2DPolygonPlotter()
plotter.plot_2d_frame(data_2d)
plotter.figure.savefig("my_plot.png")
