from pathlib import Path
import shutil
import pytest

from scivianna_example.c3po_coupling.coupling import get_panel
from scivianna.utils.serialization import load_gridstack_from_zip
from scivianna.notebook_tools import _serve_panel

working_directory = Path(__file__).parent / "working_dir"


@pytest.fixture
def cleanup():
    """Cleanup working directory after test."""
    yield
    shutil.rmtree(working_directory, ignore_errors=True)


def test_run_coupling(cleanup):
    get_panel(computation_time = 0.001, use_server = False)


def test_run_coupling_with_server(cleanup):
    get_panel(computation_time = 0.001)


def test_reload_coupling(cleanup):
    get_panel(computation_time = 0.001, use_server = False, working_directory=working_directory)

    layout = load_gridstack_from_zip(
        working_directory / "save_layout.zip",
        add_run_button=True
    )

    for panel in layout.visualisation_panels.values():
        panel.get_slave().terminate()

if __name__ == "__main__":
    get_panel(
        computation_time = 0.001, 
        use_server = False, 
        working_directory=working_directory
    )

    layout = load_gridstack_from_zip(
        working_directory / "save_layout.zip",
        add_run_button = True
    )
    _serve_panel(layout)
