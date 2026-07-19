from scivianna.icon import get_icon
from scivianna.panel.demo import Demonstrator

import scivianna_example

from scivianna_example.europe_grid.europe_grid import (
    make_europe_panel as europe_example,
)
import scivianna_example.europe_grid.europe_grid as europe_grid

try:
    import medcoupling
    from scivianna_example.med.split_item_example import (
        get_panel as medcoupling_example,
    )
    import scivianna_example.med.split_item_example as split_item_example
    has_med = True
except ImportError as e:
    print(f"Could not import medcoupling app, {e}, skipping demo medcoupling build")
    has_med = False

try:
    from scivianna_example.med.demo_3d import (
        get_panel as medcoupling_3d_example,
    )
    has_3d = True
except ImportError as e:
    print(f"Could not import 3D app, {e}, skipping demo 3D build")
    has_3d = False

from scivianna_example.mandelbrot.mandelbrot import (
    make_panel as mandelbrot_example,
)
import scivianna_example.mandelbrot.mandelbrot as mandelbrot

try:
    from scivianna_example.c3po_coupling.coupling import (
        get_panel as coupling_example,
    )
    import scivianna_example.c3po_coupling.coupling as coupling
    has_coupling = True
except ImportError as e:
    print(f"Could not import coupling app, {e}, skipping demo coupling build")
    has_coupling = False


from pathlib import Path

import panel as pn
import panel_material_ui as pmui


# download at https://composables.com/icons/icon-libraries/material-symbols?q=dashboard
def make_demo(return_slaves=False) -> pmui.Page:
    icons = {
        "Help": get_icon("question_mark"),
        "Europe example": get_icon("line_axis"),
        "Medcoupling example": get_icon("dashboard"),
        "Mandelbrot example": get_icon("grid_4x4"),
        "Coupling example": get_icon("player-play"),
        "Medcoupling 3D example": get_icon("view_in_ar"),
    }

    if return_slaves:
        europe_panel, slaves_europe = europe_example(None, return_slaves = return_slaves)

        if has_med:
            medcoupling_panel, slaves_medcoupling = medcoupling_example(None, return_slaves = return_slaves)
        else:
            slaves_medcoupling = []

        if has_3d:
            medcoupling_3d_panel, slaves_medcoupling_3d = medcoupling_3d_example(None, return_slaves = return_slaves)
        else:
            slaves_medcoupling_3d = []

        mandelbrot_panel, slaves_mandelbrot = mandelbrot_example(None, return_slaves = return_slaves)

        if has_coupling:
            coupling_panel, slaves_coupling = coupling_example(computation_time = .01, return_slaves=return_slaves, start = False, use_server=False)
        else:
            slaves_coupling = []
    else:
        europe_panel = europe_example(None)

        if has_med:
            medcoupling_panel = medcoupling_example(None)

        if has_3d:
            medcoupling_3d_panel = medcoupling_3d_example(None)

        mandelbrot_panel = mandelbrot_example(None)

        if has_coupling:
            coupling_panel = coupling_example(computation_time = .01, start = False, use_server=False)

    with open(Path(europe_grid.__file__).parent / "description.md", "r") as f:
        europe_with_description = pmui.Row(
            europe_panel.main_frame, pmui.Typography(f.read(), width=300)
        )

    if has_med:
        with open(Path(split_item_example.__file__).parent / "description.md", "r") as f:
            medcoupling_with_description = pmui.Row(
                medcoupling_panel.main_frame, pmui.Typography(f.read(), width=300)
            )

    if has_3d:
        with open(Path(split_item_example.__file__).parent / "description_3d.md", "r") as f:
            medcoupling_3d_with_description = pmui.Row(
                medcoupling_3d_panel.main_frame, pmui.Typography(f.read(), width=300)
            )

    with open(Path(mandelbrot.__file__).parent / "description.md", "r") as f:
        mandelbrot_with_description = pmui.Row(
            mandelbrot_panel.main_frame, pmui.Typography(f.read(), width=300)
        )

    if has_coupling:
        with open(Path(coupling.__file__).parent / "description.md", "r") as f:
            coupling_with_description = pmui.Row(
                coupling_panel.main_frame, pmui.Typography(f.read(), width=300)
            )

    description_file = Path(scivianna_example.__file__).parent / "demo_description.md"

    image = pn.pane.Image(Path(scivianna_example.__file__).parent / "image/main_page.png", sizing_mode = "stretch_both")

    with open(description_file, 'r') as f:
        help = pn.Column(pmui.Typography(f.read()), image, sizing_mode = "stretch_both")

    guis = {
        "Help": help,
        "Europe example": europe_with_description,
        "Mandelbrot example": mandelbrot_with_description,
    }

    if has_med:
        guis["Medcoupling example"] = medcoupling_with_description
    if has_coupling:
        guis["Coupling example"] = coupling_with_description
    if has_3d:
        guis["Medcoupling 3D example"] = medcoupling_3d_with_description

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
