from pathlib import Path
from scivianna.layout.split import SplitLayout
from scivianna.panel.visualisation_panel import (
    VisualizationPanel
)
import scivianna
from scivianna.panel.panel_3d import Panel3D
from scivianna.slave import ComputeSlave
from scivianna.constants import GEOMETRY
from scivianna.interface.med_interface import MEDInterface
from scivianna.notebook_tools import _serve_panel


def get_panel(geo, title: str = "", *args, return_slaves=False, **kwargs) -> VisualizationPanel:
    slave = ComputeSlave(MEDInterface)
    if geo is None:
        slave.read_file(Path(scivianna.__file__).parent / "input_file" / "power.med", GEOMETRY)
    elif isinstance(geo, (str, Path)):
        slave.read_file(geo, GEOMETRY)
    else:
        raise TypeError(f"Provided type {type(geo)} not implemented")

    med_panel = Panel3D(slave)
    if return_slaves:
        return SplitLayout(med_panel), [med_panel.get_slave()]
    else:
        return SplitLayout(med_panel)


if __name__ == "__main__":
    _serve_panel(get_panel_function=get_panel)
