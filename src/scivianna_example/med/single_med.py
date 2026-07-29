from scivianna.layout.split import SplitLayout
from scivianna.notebook_tools import _serve_panel, get_med_panel
from scivianna.panel.visualisation_panel import VisualizationPanel


def get_panel(*args, return_slaves=False, **kwargs) -> VisualizationPanel:
    med_1 = get_med_panel(geo=None, title="MEDCoupling visualizer XY")

    if return_slaves:
        return SplitLayout(med_1), [med_1.get_slave()]
    else:
        return SplitLayout(med_1)


if __name__ == "__main__":
    _serve_panel(get_panel_function=get_panel)
