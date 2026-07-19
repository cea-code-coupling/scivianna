from pathlib import Path
import shutil
import pytest

from scivianna.plotter_1d.bokeh_1d_plotter import BokehPlotter1D
from scivianna.plotter_2d.polygon.bokeh import Bokeh2DPolygonPlotter
import scivianna.utils
from scivianna.utils.serialization import load_gridstack_from_zip, load_layout_from_zip
from scivianna.notebook_tools import _serve_panel

working_directory = Path(__file__).parent / "working_dir"
working_directory.mkdir(exist_ok=True, parents=True)

@pytest.fixture
def cleanup():
    """Cleanup working directory after test."""
    yield
    shutil.rmtree(working_directory, ignore_errors=True)

# 1. Define the fixture with params
@pytest.fixture(params=[True, False])
def grid(request):
    # Return True or False based on the parameter
    return request.param
 
@pytest.mark.coupling
def test_run_coupling(cleanup, grid):
    from scivianna_example.c3po_coupling.coupling import get_panel
    get_panel(computation_time = 0.001, use_server = False, grid = grid)


@pytest.mark.coupling
def test_run_coupling_with_server(cleanup, grid):
    from scivianna_example.c3po_coupling.coupling import get_panel
    get_panel(computation_time = 0.001, grid = grid)


@pytest.mark.coupling
def test_reload_coupling(cleanup, grid):
    from scivianna_example.c3po_coupling.coupling import get_panel
    scivianna.utils._testing = True
    get_panel(
        computation_time = 0.001, 
        use_server = False, 
        working_directory=working_directory, 
        grid = grid
    )

    if grid:
        layout = load_gridstack_from_zip(
            working_directory / "save_layout.zip"
        )
    else:
        layout = load_layout_from_zip(
            working_directory / "save_layout.zip"
        )

    assert layout.time_widget is not None

    assert layout.time_widget.time_slider.options == [0., 0.0005]

    layout.time_widget.time_slider.value = 0.
    assert set(list(layout.visualisation_panels.keys())) == set(["Field", "Temperature"])

    assert isinstance(layout.visualisation_panels["Temperature"].plotter, BokehPlotter1D)
    assert isinstance(layout.visualisation_panels["Field"].plotter, Bokeh2DPolygonPlotter)

    for panel in layout.visualisation_panels.values():
        panel.get_slave().terminate()

if __name__ == "__main__":
    from scivianna_example.c3po_coupling.coupling import get_panel
    get_panel(
        computation_time = 0.01,
        use_server = False,
        working_directory=working_directory
    )

    def load_layout(*args, **kwargs):
        layout = load_gridstack_from_zip(
            working_directory / "save_layout.zip"
        )
        return layout
    _serve_panel(get_panel_function = load_layout)
