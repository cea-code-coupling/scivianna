from scivianna.constants import X, Y, Z
from scivianna.enums import UpdateEvent
from scivianna.layout.split import (
    SplitItem,
    SplitDirection,
    SplitLayout,
)
from scivianna.notebook_tools import get_med_panel, _serve_panel


def get_panel(_, return_slaves=False) -> SplitLayout:

    med_1 = get_med_panel(geo=None, title="MEDCoupling visualizer XY")
    med_2 = get_med_panel(geo=None, title="MEDCoupling visualizer XZ")
    med_3 = get_med_panel(geo=None, title="MEDCoupling visualizer YZ")

    med_1.set_field("INTEGRATED_POWER")
    med_2.set_field("INTEGRATED_POWER")
    med_3.set_field("INTEGRATED_POWER")

    med_1.update_event = UpdateEvent.CLIC

    med_2.update_event = UpdateEvent.CLIC
    med_2.set_coordinates(u=X, v=Z)

    med_3.update_event = UpdateEvent.CLIC
    med_3.set_coordinates(u=Y, v=Z)

    split = SplitItem(med_1, med_2, SplitDirection.VERTICAL)
    split = SplitItem(split, med_3, SplitDirection.HORIZONTAL)

    if return_slaves:
        return SplitLayout(split), [med_1.get_slave(), med_2.get_slave(), med_3.get_slave()]
    else:   
        return SplitLayout(split)


if __name__ == "__main__":
    _serve_panel(get_panel_function=get_panel)
