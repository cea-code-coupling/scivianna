import pytest

import scivianna.utils
from scivianna.constants import X, Y, Z, MESH
from scivianna.utils.serialization import (
    save_slave_to_file, 
    load_slave_from_file, 
    save_panel2d_to_file, 
    load_panel2d_from_file
)

from scivianna.layout.split import SplitLayout
from scivianna.layout.gridstack import GridStackLayout
from scivianna_example.europe_grid.europe_grid import make_europe_panel, EuropeGridInterface, CountryTimeSeriesInterface

scivianna.utils._testing = True

@pytest.mark.medcoupling
def test_serialize_slave():
    from scivianna.interface.med_interface import MEDInterface
    from scivianna_example.med.split_item_example import get_panel, get_med_panel
    slave = None
    slave2 = None
    try:
        med_panel = get_med_panel(None)

        slave = med_panel.get_slave()
        
        # First compute with caller="Test" to populate cache
        import numpy as np
        origin = np.array(X) * 0.5 + np.array(Y) * 0.5 + 0.5 * np.cross(X, Y)
        slave.compute_2D_data(
            X, Y, tuple(origin), 1.0, 1.0, None, MESH, {}, caller="Test"
        )
        
        save_slave_to_file(
            slave,
            "slave.pkl",
            True
        )
        slave.terminate()

        slave2 = load_slave_from_file(
            "slave.pkl",
            MEDInterface,
            True
        )
        data, recomputed, = slave2.compute_2D_data(
            X, Y, tuple(origin), 1.0, 1.0, None, MESH, {}, caller="Test"
        )
        assert not recomputed, "Slave2 should not have recomputed the polygons."
    finally:
        if slave is not None:
            slave.terminate()
        if slave2 is not None:
            slave2.terminate()

@pytest.mark.medcoupling
def test_serialize_panel():
    from scivianna_example.med.split_item_example import get_panel, get_med_panel
    slave = None
    slave2 = None
    try:
        med_panel = get_med_panel(None)
        slave = med_panel.get_slave()
        med_panel.set_field("INTEGRATED_POWER")
        med_panel.set_colormap("viridis")
        med_panel.set_coordinates(u=X, v=Z)

        import time
        time.sleep(1)

        # First compute with caller="Test" to populate cache
        import numpy as np
        u = med_panel.u
        v = med_panel.v
        w = np.cross(u, v)
        origin = med_panel.origin  # tuple (origin_x, origin_y, origin_z)
        size_u = med_panel.size_u
        size_v = med_panel.size_v
        slave.compute_2D_data(
            u, 
            v,
            origin,
            size_u,
            size_v,
            None,
            MESH,
            {},
            caller="Test"
        )

        save_panel2d_to_file(
            med_panel,
            "panel2d_test"
        )
        slave.terminate()

        panel_2 = load_panel2d_from_file(
            "panel2d_test",
        )
        slave_2 = panel_2.get_slave()
        _, recomputed, = slave_2.compute_2D_data(
            panel_2.u, 
            panel_2.v,
            panel_2.origin,  # tuple (origin_x, origin_y, origin_z)
            panel_2.size_u,
            panel_2.size_v,
            None,
            MESH,
            {},
            caller="Test"
        )
        print(
            panel_2.u, 
            panel_2.v, 
            panel_2.origin,
            panel_2.size_u, panel_2.size_v,
            _, recomputed
        )
        assert not recomputed, "Slave should not have recomputed the polygons."

        # panel_2.show()
    finally:
        if slave is not None:
            print("Terminating slave 0")
            slave.terminate()
        if slave2 is not None:
            print("Terminating slave 1")
            slave2.terminate()

@pytest.mark.medcoupling
def test_serialize_split():
    from scivianna_example.med.split_item_example import get_panel, get_med_panel
    panel, slaves = get_panel(None, True)

    panel.save_to_zip("test.zip")

    for slave in slaves:
        slave.terminate()

    try:
        new_layout = SplitLayout.restore_from_zip("test.zip")
    except Exception as e:
        print(e)
    finally:
        for panel in new_layout.visualisation_panels.values():
            panel.get_slave().terminate()

@pytest.mark.medcoupling
def test_serialize_gridstack():
    from scivianna_example.med.grid_stack_example import get_panel as get_gridstack_panel
    panel, slaves = get_gridstack_panel(None, True)

    panel.save_to_zip("test_gridstack.zip")

    for slave in slaves:
        slave.terminate()

    try:
        new_layout = GridStackLayout.restore_from_zip("test_gridstack.zip")
    except Exception as e:
        print("Received exception ", e)
    finally:
        for panel in new_layout.visualisation_panels.values():
            panel.get_slave().terminate()

@pytest.mark.default
def test_restore_europe():
    layout, slaves = make_europe_panel(None, True)

    # Save the SplitLayout to a zip file
    layout.visualisation_panels["Map"].set_field("load")
    layout.visualisation_panels["Plot"].set_field("load")
    layout.save_to_zip("test.zip")

    for slave in slaves:
        slave.terminate()

    try:
        new_layout = SplitLayout.restore_from_zip(
            "test.zip",
            additional_interfaces={
                "EuropeGridInterface": EuropeGridInterface,
                "CountryTimeSeriesInterface": CountryTimeSeriesInterface,
            }
        )
    except Exception as e:
        print(e)
    finally:
        for panel in new_layout.visualisation_panels.values():
            panel.get_slave().terminate()

if __name__ == "__main__":
    # test_serialize_panel()
    test_serialize_split()
    # test_serialize_gridstack()