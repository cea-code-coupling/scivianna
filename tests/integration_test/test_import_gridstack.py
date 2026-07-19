from typing import Dict
import pytest
from scivianna.layout.gridstack import GridStackLayout
from scivianna.notebook_tools import get_med_panel
from scivianna.panel.visualisation_panel import VisualizationPanel


def get_panel():
    visualisation_panels: Dict[str, VisualizationPanel] = {}

    visualisation_panels["MEDCoupling visualizer 1"] = get_med_panel(
        geo=None, title="MEDCoupling visualizer 1"
    )
    visualisation_panels["MEDCoupling visualizer 2"] = get_med_panel(
        geo=None, title="MEDCoupling visualizer 2"
    )
    
    visualisation_panels["MEDCoupling visualizer 3"] = get_med_panel(
        geo=None, title="MEDCoupling visualizer 3"
    )

    bounds_x = {
        "MEDCoupling visualizer 1": (0, 5),
        "MEDCoupling visualizer 2": (0, 5),
        "MEDCoupling visualizer 3": (5, 10),
    }

    bounds_y = {
        "MEDCoupling visualizer 1": (0, 5),
        "MEDCoupling visualizer 2": (5, 10),
        "MEDCoupling visualizer 3": (0, 10),
    }

    visualisation_panels["MEDCoupling visualizer 1"].get_slave().terminate()
    visualisation_panels["MEDCoupling visualizer 2"].get_slave().terminate()
    visualisation_panels["MEDCoupling visualizer 3"].get_slave().terminate()

    return GridStackLayout(visualisation_panels, bounds_y, bounds_x)

def get_template():
    panel = get_panel()
    return panel

@pytest.mark.medcoupling
def test_import_gridstack():
    """Test importing the gridstack layout and make the panel without opening it
    """
    get_template()
    assert True

if __name__ == "__main__":
    print("Starting")
    test_import_gridstack()