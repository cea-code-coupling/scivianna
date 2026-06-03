from typing import Dict
from scivianna.constants import X, Y, Z
from scivianna.enums import UpdateEvent
from scivianna.layout.gridstack import GridStackLayout
from scivianna.panel.visualisation_panel import VisualizationPanel
from scivianna.notebook_tools import (
    get_med_panel
)

import panel as pn


def get_panel(_, return_slaves = False):
    visualisation_panels: Dict[str, VisualizationPanel] = {}

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

    for m in [med_1, med_2, med_3]:
        visualisation_panels[m.panel_name] = m

    bounds_x = {
        "MEDCoupling visualizer XY": (0, 5),
        "MEDCoupling visualizer YZ": (0, 5),
        "MEDCoupling visualizer XZ": (5, 10),
    }

    bounds_y = {
        "MEDCoupling visualizer XY": (0, 5),
        "MEDCoupling visualizer YZ": (5, 10),
        "MEDCoupling visualizer XZ": (0, 10),
    }

    if return_slaves:
        return GridStackLayout(visualisation_panels, bounds_y, bounds_x), [med_1.get_slave(), med_2.get_slave(), med_3.get_slave()]
    else:   
        return GridStackLayout(visualisation_panels, bounds_y, bounds_x)



def get_template():
    panel = get_panel(None)
    return panel.main_frame


if __name__ == "__main__":
    #   Serving panel as main, file executed with a command : "python my_file.py"
    import socket

    ip_adress = socket.gethostbyname(socket.gethostname())

    """
        Catching a free port to provide to pn.serve
    """
    sock = socket.socket()
    sock.bind((ip_adress, 0))
    port = sock.getsockname()[1]
    sock.close()

    server = pn.serve(
        get_template,
        address=ip_adress,
        websocket_origin=f"{ip_adress}:{port}",
        port=port,
        threaded=True,
    )
