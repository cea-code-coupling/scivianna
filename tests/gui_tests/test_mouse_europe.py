import pytest

from scivianna.layout.split import SplitLayout
from scivianna.panel.panel_1d import Panel1D
from scivianna.panel.panel_2d import Panel2D

from scivianna_example.europe_grid.europe_grid import (
    make_europe_panel as europe_example,
)

from scivianna.constants import XS, YS, CELL_NAMES
from scivianna.enums import UpdateEvent
import scivianna.utils

scivianna.utils._testing = True

def move_to_country(panel_2d, country):
    index = list(panel_2d.plotter.source_polygons.data[CELL_NAMES]).index(country)
    panel_2d.plotter.source_mouse.data = {
        "sx": [0.],
        "sy": [0.],
        "x": [0.],
        "y": [0.],
        "z": [0.0],
        "index": [index],
    }
    panel_2d.plotter.send_event(panel_2d.plotter.on_mouse_move_callback)
    return panel_2d.plotter.source_polygons.data[CELL_NAMES][index]

def clic_to_country(panel_2d, country):
    index = list(panel_2d.plotter.source_polygons.data[CELL_NAMES]).index(country)
    panel_2d.plotter.source_mouse.data = {
        "sx": [0.],
        "sy": [0.],
        "x": [0.],
        "y": [0.],
        "z": [0.0],
        "index": [index],
    }
    panel_2d.plotter.send_event(panel_2d.plotter.on_clic_callback)
    return panel_2d.plotter.source_polygons.data[CELL_NAMES][index]


# @pytest.mark.default
def test_europe_mouse_move():
    split_panel: SplitLayout
    split_panel, [slave, slave_result] = europe_example(None, return_slaves=True)

    panel_1d = split_panel.split_item.panel_2
    panel_2d = split_panel.split_item.panel_1

    current_field = panel_1d.visible_fields_list[0]

    country_name = move_to_country(panel_2d, "FR")
    assert f"{country_name.lower()}_{current_field.lower()}" in panel_1d.plotter.source_data_dict
    assert panel_1d.plotter.visible == [f"{country_name.lower()}_{current_field.lower()}"]

    country_name = move_to_country(panel_2d, "DE")
    assert f"{country_name.lower()}_{current_field.lower()}" in panel_1d.plotter.source_data_dict
    assert panel_1d.plotter.visible == [f"{country_name.lower()}_{current_field.lower()}"]

    panel_1d.visible_fields_list = [panel_1d.fields_list[1]]
    current_field = panel_1d.visible_fields_list[0]

    country_name = move_to_country(panel_2d, "ES")
    assert f"{country_name.lower()}_{current_field.lower()}" in panel_1d.plotter.source_data_dict
    assert panel_1d.plotter.visible == [f"{country_name.lower()}_{current_field.lower()}"]

    country_name = move_to_country(panel_2d, "UK")
    assert f"{country_name.lower()}_{current_field.lower()}" in panel_1d.plotter.source_data_dict
    assert panel_1d.plotter.visible == [f"{country_name.lower()}_{current_field.lower()}"]

    clic_to_country(panel_2d, "IT")
    assert f"{country_name.lower()}_{current_field.lower()}" in panel_1d.plotter.source_data_dict
    assert panel_1d.plotter.visible == [f"{country_name.lower()}_{current_field.lower()}"]

# @pytest.mark.default
def test_europe_mouse_clic():
    split_panel: SplitLayout
    split_panel, [slave, slave_result] = europe_example(None, return_slaves=True)

    panel_1d = split_panel.split_item.panel_2
    panel_2d = split_panel.split_item.panel_1

    panel_1d.update_event = UpdateEvent.CLIC

    current_field = panel_1d.visible_fields_list[0]

    country_name = clic_to_country(panel_2d, "FR")
    assert f"{country_name.lower()}_{current_field.lower()}" in panel_1d.plotter.source_data_dict
    assert panel_1d.plotter.visible == [f"{country_name.lower()}_{current_field.lower()}"]

    country_name = clic_to_country(panel_2d, "DE")
    assert f"{country_name.lower()}_{current_field.lower()}" in panel_1d.plotter.source_data_dict
    assert panel_1d.plotter.visible == [f"{country_name.lower()}_{current_field.lower()}"]

    panel_1d.visible_fields_list = [panel_1d.fields_list[1]]
    current_field = panel_1d.visible_fields_list[0]

    country_name = clic_to_country(panel_2d, "ES")
    assert f"{country_name.lower()}_{current_field.lower()}" in panel_1d.plotter.source_data_dict
    assert panel_1d.plotter.visible == [f"{country_name.lower()}_{current_field.lower()}"]

    country_name = clic_to_country(panel_2d, "UK")
    assert f"{country_name.lower()}_{current_field.lower()}" in panel_1d.plotter.source_data_dict
    assert panel_1d.plotter.visible == [f"{country_name.lower()}_{current_field.lower()}"]

    move_to_country(panel_2d, "IT")
    assert f"{country_name.lower()}_{current_field.lower()}" in panel_1d.plotter.source_data_dict
    assert panel_1d.plotter.visible == [f"{country_name.lower()}_{current_field.lower()}"]


if __name__ == "__main__":
    test_europe_mouse_clic()
