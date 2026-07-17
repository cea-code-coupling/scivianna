
import scivianna
from scivianna.panel.demo import Demonstrator

import scivianna_example

from scivianna_example.europe_grid.europe_grid import (
    make_europe_panel as europe_example,
)
import scivianna_example.europe_grid.europe_grid as europe_grid

from scivianna_example.med.split_item_example import (
    get_panel as medcoupling_example,
)
import scivianna_example.med.split_item_example as split_item_example

from scivianna_example.med.demo_3d import (
    get_panel as medcoupling_3d_example,
)

from scivianna_example.mandelbrot.mandelbrot import (
    make_panel as mandelbrot_example,
)
import scivianna_example.mandelbrot.mandelbrot as mandelbrot

from scivianna_example.c3po_coupling.coupling import (
    get_panel as coupling_example,
)
import scivianna_example.c3po_coupling.coupling as coupling


from pathlib import Path

import panel as pn
import panel_material_ui as pmui


def get_icon(icon_name: str):
    with open(str(Path(scivianna.__file__).parent / "icon" / icon_name) + ".svg", "r") as f:
        return f.read()

# download at https://composables.com/icons/icon-libraries/material-symbols?q=dashboard
def make_demo(return_slaves=False) -> pmui.Page:
    icons = {
        "Help": get_icon("question_mark"),
        "Europe example": get_icon("line_axis"),
        "Medcoupling example": get_icon("dashboard"),
        "Medcoupling 3D example": get_icon("view_in_ar"),
        "Mandelbrot example": get_icon("grid_4x4"),
        "Coupling example": get_icon("player-play"),
    }

    if return_slaves:
        europe_panel, slaves_europe = europe_example(None, return_slaves)
        medcoupling_panel, slaves_medcoupling = medcoupling_example(None, return_slaves)
        medcoupling_3d_panel, slaves_medcoupling_3d = medcoupling_3d_example(None, return_slaves)
        mandelbrot_panel, slaves_mandelbrot = mandelbrot_example(None, return_slaves)
        coupling_panel, slaves_coupling = coupling_example(computation_time = .01, return_slaves=return_slaves, start = False, use_server=False)
    else:
        europe_panel = europe_example(None)
        medcoupling_panel = medcoupling_example(None)
        medcoupling_3d_panel = medcoupling_3d_example(None)
        mandelbrot_panel = mandelbrot_example(None)
        coupling_panel = coupling_example(computation_time = .01, start = False, use_server=False)

    with open(Path(europe_grid.__file__).parent / "description.md", "r") as f:
        europe_with_description = pmui.Row(
            europe_panel.main_frame, pmui.Typography(f.read(), width=300)
        )

    with open(Path(split_item_example.__file__).parent / "description.md", "r") as f:
        medcoupling_with_description = pmui.Row(
            medcoupling_panel.main_frame, pmui.Typography(f.read(), width=300)
        )

    with open(Path(split_item_example.__file__).parent / "description_3d.md", "r") as f:
        medcoupling_3d_with_description = pmui.Row(
            medcoupling_3d_panel.main_frame, pmui.Typography(f.read(), width=300)
        )

    with open(Path(mandelbrot.__file__).parent / "description.md", "r") as f:
        mandelbrot_with_description = pmui.Row(
            mandelbrot_panel.main_frame, pmui.Typography(f.read(), width=300)
        )

    with open(Path(coupling.__file__).parent / "description.md", "r") as f:
        coupling_with_description = pmui.Row(
            coupling_panel.main_frame, pmui.Typography(f.read(), width=300)
        )

    description_file = Path(scivianna_example.__file__).parent / "demo_description.md"

    image = pn.pane.Image(Path(scivianna_example.__file__).parent / "image/tuto_visu_serma.png", sizing_mode = "stretch_both")

    with open(description_file, 'r') as f:
        help = pn.Column(pmui.Typography(f.read()), image, sizing_mode = "stretch_both")

    guis = {
        "Help": help,
        "Europe example": europe_with_description,
        "Medcoupling example": medcoupling_with_description,
        "Medcoupling 3D example": medcoupling_3d_with_description,
        "Mandelbrot example": mandelbrot_with_description,
        "Coupling example": coupling_with_description,
    }

    demo = Demonstrator(guis, icons)

    if return_slaves:
        return demo, slaves_medcoupling + slaves_europe + slaves_mandelbrot + slaves_coupling + slaves_medcoupling_3d
    else:
        return demo


if __name__ == "__main__":
    import panel as pn
    import socket

    ip_adress = socket.gethostbyname(socket.gethostname())

    """
        Catching a free port to provide to pn.serve
    """
    sock = socket.socket()
    sock.bind((ip_adress, 0))
    port = sock.getsockname()[1]
    sock.close()

    pn.serve(
        make_demo,
        address=ip_adress,
        websocket_origin=f"{ip_adress}:{port}",
        port=port,
        threaded=True,
    )
