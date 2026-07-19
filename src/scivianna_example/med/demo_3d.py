from pathlib import Path
from scivianna.layout.split import SplitLayout
from scivianna.panel.visualisation_panel import (
    VisualizationPanel
)
import scivianna
from scivianna.panel.panel_2d import Panel2D
from scivianna.panel.panel_3d import Panel3D
from scivianna.slave import ComputeSlave
from scivianna.constants import GEOMETRY
from scivianna.interface.med_interface import MEDInterface
from scivianna.notebook_tools import _serve_panel, get_med_panel
from scivianna.enums import UpdateEvent
from scivianna.layout.split import (
    SplitItem,
    SplitDirection,
    SplitLayout,
)
from scivianna.constants import X, Y


def get_panel(geo, title: str = "3D", *args, return_slaves=False, **kwargs) -> VisualizationPanel:
    slave = ComputeSlave(MEDInterface)
    if geo is None:
        slave.read_file(Path(scivianna.__file__).parent / "input_file" / "power.med", GEOMETRY)
    elif isinstance(geo, (str, Path)):
        slave.read_file(geo, GEOMETRY)
    else:
        raise TypeError(f"Provided type {type(geo)} not implemented")
    
    med_2 = Panel2D(slave, name="MEDCoupling slice", u=X, v=Y, displayed_field = "INTEGRATED_POWER")
    med_2.update_event = [UpdateEvent.CLIC, UpdateEvent.AXES_CHANGE]

    med_panel_3d = Panel3D(slave, name="3D Demo", displayed_field = "INTEGRATED_POWER")
    med_panel_3d.update_event = [UpdateEvent.CLIC, UpdateEvent.AXES_CHANGE]
    
    split = SplitItem(med_panel_3d, med_2, SplitDirection.VERTICAL)

    if return_slaves:
        return SplitLayout(split), [med_panel_3d.get_slave()]
    else:
        return SplitLayout(split)


if __name__ == "__main__":
    if True:
        get_panel(None).show()
        # _serve_panel(get_panel_function=get_panel)
    else:
        slave = ComputeSlave(MEDInterface)
        slave.read_file(Path(scivianna.__file__).parent / "input_file" / "power.med", GEOMETRY)

        data, _ = slave.compute_3D_data({}, "INTEGRATED_POWER")

        data.update_cell_data()