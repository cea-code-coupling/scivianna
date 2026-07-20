from pathlib import Path

import scivianna
from scivianna.constants import X, Y, Z
from scivianna.enums import UpdateEvent
from scivianna.layout.split import (
    SplitItem,
    SplitDirection,
    SplitLayout,
)
from scivianna.notebook_tools import get_med_panel, _serve_panel
from scivianna.panel.panel_2d import Panel2D
from scivianna.slave import ComputeSlave
from scivianna.constants import GEOMETRY


def get_panel(_, return_slaves=False) -> SplitLayout:
    from scivianna.interface.med_interface import MEDInterface
    slave = ComputeSlave(MEDInterface)
    slave.read_file(Path(scivianna.__file__).parent / "input_file" / "power.med", GEOMETRY)

    med_1 = Panel2D(
        slave, 
        name="MEDCoupling visualizer XY",
        displayed_field = "INTEGRATED_POWER",
        u=X,
        v=Y
    )
    med_2 = Panel2D(
        slave, 
        name="MEDCoupling visualizer XZ",
        displayed_field = "INTEGRATED_POWER",
        u=X,
        v=Z
    )
    med_3 = Panel2D(
        slave, 
        name="MEDCoupling visualizer YZ",
        displayed_field = "INTEGRATED_POWER",
        u=Y,
        v=Z
    )
    med_1.update_event = UpdateEvent.CLIC
    med_2.update_event = UpdateEvent.CLIC
    med_3.update_event = UpdateEvent.CLIC

    split = SplitItem(med_1, med_2, SplitDirection.VERTICAL)
    split = SplitItem(split, med_3, SplitDirection.HORIZONTAL)

    if return_slaves:
        return SplitLayout(split), [med_1.get_slave(), med_2.get_slave(), med_3.get_slave()]
    else:   
        return SplitLayout(split)


if __name__ == "__main__":
    _serve_panel(get_panel_function=get_panel)
