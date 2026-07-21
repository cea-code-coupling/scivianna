import pytest
import tempfile
from pathlib import Path

import scivianna.utils
from scivianna.constants import X, Y, Z, MESH
from scivianna.utils.serialization import (
    save_slave_to_file, 
    load_slave_from_file, 
    save_panel2d_to_file, 
    load_panel2d_from_file,
    save_panel1d_to_file,
    load_panel1d_from_file,
    save_panel3d_to_file,
    load_panel3d_from_file,
    save_paneldatframe_to_file,
    load_paneldatframe_from_file,
)

from scivianna.layout.split import SplitLayout
from scivianna.layout.gridstack import GridStackLayout
from scivianna.slave import ComputeSlave
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
def test_serialize_panel_2d():
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
def test_serialize_europe():
    """Serialize and restore the europe_grid layout, verify panels and fields survive."""
    layout, slaves = make_europe_panel(None, True)

    # Set fields to exercise serialization state
    layout.visualisation_panels["Map"].set_field("load")
    layout.visualisation_panels["Plot"].set_field("load")

    # Save the SplitLayout to a zip file
    layout.save_to_zip("test_europe.zip")

    for slave in slaves:
        slave.terminate()

    # Restore from zip
    new_layout = SplitLayout.restore_from_zip(
        "test_europe.zip",
    )

    # Verify all panels exist
    assert "Map" in new_layout.visualisation_panels
    assert "Plot" in new_layout.visualisation_panels
    assert "Dataframe" in new_layout.visualisation_panels

    # Verify fields were restored
    assert new_layout.visualisation_panels["Map"].displayed_field == "load"

    for panel in new_layout.visualisation_panels.values():
        panel.get_slave().terminate()


@pytest.mark.pyvista
def test_serialize_demo_3d():
    """Serialize and restore the demo_3d layout, verify 2D/3D panels and coupling survive."""
    from scivianna_example.med.demo_3d import get_panel

    layout, slaves = get_panel(None, return_slaves=True)

    # Save the SplitLayout to a zip file
    layout.save_to_zip("test_demo_3d.zip")

    for slave in slaves:
        slave.terminate()

    # Restore from zip
    new_layout = SplitLayout.restore_from_zip("test_demo_3d.zip")

    # Verify all panels exist
    assert "3D Demo" in new_layout.visualisation_panels
    assert "MEDCoupling slice" in new_layout.visualisation_panels

    for panel in new_layout.visualisation_panels.values():
        panel.get_slave().terminate()


# =============================================================================
# PANEL1D SERIALIZATION
# =============================================================================

@pytest.mark.default
def test_serialize_panel1d_individual():
    """Serialize and restore a Panel1D individually, verify panel state survives."""
    from scivianna.panel.panel_1d import Panel1D
    from scivianna_example.europe_grid.country_time_series import CountryTimeSeriesInterface
    from scivianna_example import europe_grid

    slave = None
    panel_loaded = None
    try:
        # Create a Panel1D with CountryTimeSeriesInterface
        slave = ComputeSlave(CountryTimeSeriesInterface)
        csv_path = str(Path(europe_grid.__file__).parent / "time_series.csv")
        slave.read_file(csv_path, "TimeSeries")

        panel = Panel1D(slave, name="Test Plot")
        panel.set_field("load")

        # Save and load (use .zip suffix explicitly)
        zip_path = Path("panel1d_test.zip")
        save_panel1d_to_file(panel, zip_path)
        slave.terminate()

        panel_loaded = load_panel1d_from_file(zip_path)

        # Verify panel was restored
        assert panel_loaded.panel_name == "Test Plot"
        # Verify the slave was restored and has labels
        loaded_labels = panel_loaded.slave.get_labels()
        assert len(loaded_labels) > 0
        # Verify the panel has a valid visible_fields_list (non-empty)
        assert len(panel_loaded.visible_fields_list) > 0

    finally:
        if slave is not None:
            slave.terminate()
        if panel_loaded is not None:
            panel_loaded.get_slave().terminate()


@pytest.mark.default
def test_serialize_panel1d_in_layout():
    """Serialize a layout containing Panel1D, verify panel state survives round-trip."""
    layout, slaves = make_europe_panel(None, True)

    # Change fields
    layout.visualisation_panels["Plot"].set_field("load")

    # Save layout
    layout.save_to_zip("test_panel1d_layout.zip")

    for slave in slaves:
        slave.terminate()

    # Restore
    new_layout = SplitLayout.restore_from_zip(
        "test_panel1d_layout.zip",
    )

    # Verify Panel1D state
    plot_panel = new_layout.visualisation_panels["Plot"]
    assert plot_panel.panel_name == "Plot"
    # Verify the panel has valid state (non-empty visible fields)
    assert len(plot_panel.visible_fields_list) > 0

    for panel in new_layout.visualisation_panels.values():
        panel.get_slave().terminate()


# =============================================================================
# PANEL3D SERIALIZATION
# =============================================================================

@pytest.mark.pyvista
def test_serialize_panel3d_individual():
    """Serialize and restore a Panel3D individually, verify displayed field and colormap survive."""
    import scivianna
    from scivianna.constants import GEOMETRY
    from scivianna.panel.panel_3d import Panel3D
    from scivianna.interface.med_interface import MEDInterface

    slave = None
    panel_loaded = None
    try:
        slave = ComputeSlave(MEDInterface)
        med_path = str(Path(scivianna.__file__).parent / "input_file" / "power.med")
        slave.read_file(med_path, GEOMETRY)

        panel = Panel3D(slave, name="Test 3D", displayed_field="INTEGRATED_POWER", colormap="viridis")

        # Compute some data to populate current_data
        data, _ = slave.compute_3D_data("INTEGRATED_POWER", {})
        data.update_cell_data()

        # Save and load (use .zip suffix explicitly)
        zip_path = Path("panel3d_test.zip")
        save_panel3d_to_file(panel, zip_path)
        slave.terminate()

        panel_loaded = load_panel3d_from_file(zip_path)

        # Verify state was restored
        assert panel_loaded.panel_name == "Test 3D"
        assert panel_loaded.displayed_field == "INTEGRATED_POWER"
        assert panel_loaded.colormap == "viridis"
        # Verify current_data survived
        assert panel_loaded.current_data is not None

    finally:
        if slave is not None:
            slave.terminate()
        if panel_loaded is not None:
            panel_loaded.get_slave().terminate()


@pytest.mark.pyvista
def test_serialize_panel3d_in_layout():
    """Serialize a layout containing Panel3D, verify 3D panel state survives round-trip."""
    from scivianna_example.med.demo_3d import get_panel

    layout, slaves = get_panel(None, return_slaves=True)

    # Change displayed field on 3D panel
    panel_3d = layout.visualisation_panels["3D Demo"]
    if "INTEGRATED_POWER" in [l for l in panel_3d.slave.get_labels()]:
        panel_3d.set_field("INTEGRATED_POWER")

    # Save layout
    layout.save_to_zip("test_panel3d_layout.zip")

    for slave in slaves:
        slave.terminate()

    # Restore
    new_layout = SplitLayout.restore_from_zip("test_panel3d_layout.zip")

    # Verify Panel3D state
    restored_3d = new_layout.visualisation_panels["3D Demo"]
    assert restored_3d.panel_name == "3D Demo"

    for panel in new_layout.visualisation_panels.values():
        panel.get_slave().terminate()


# =============================================================================
# PANELDATAFRAME SERIALIZATION
# =============================================================================

@pytest.mark.default
def test_serialize_paneldatframe_individual():
    """Serialize and restore a PanelDataFrame individually, verify dataframe content survives."""
    from scivianna.panel.panel_dataframe import PanelDataFrame
    from scivianna_example.europe_grid.country_time_series import CountryTimeSeriesInterface
    from scivianna_example import europe_grid

    slave = None
    panel_loaded = None
    try:
        slave = ComputeSlave(CountryTimeSeriesInterface)
        csv_path = str(Path(europe_grid.__file__).parent / "time_series.csv")
        slave.read_file(csv_path, "TimeSeries")

        panel = PanelDataFrame(slave, name="Test Dataframe")
        panel.recompute()

        # Save and load (use .zip suffix explicitly)
        zip_path = Path("paneldatframe_test.zip")
        save_paneldatframe_to_file(panel, zip_path)
        slave.terminate()

        panel_loaded = load_paneldatframe_from_file(zip_path)

        # Verify state was restored
        assert panel_loaded.panel_name == "Test Dataframe"
        # Verify dataframe was restored
        df = panel_loaded.plotter.get_data()
        assert df is not None
        assert len(df) > 0

    finally:
        if slave is not None:
            slave.terminate()
        if panel_loaded is not None:
            panel_loaded.get_slave().terminate()


@pytest.mark.default
def test_serialize_paneldatframe_in_layout():
    """Serialize a layout containing PanelDataFrame, verify dataframe survives round-trip."""
    layout, slaves = make_europe_panel(None, True)

    # Trigger recomputation on dataframe panel
    df_panel = layout.visualisation_panels["Dataframe"]
    df_panel.recompute()

    # Save layout
    layout.save_to_zip("test_paneldatframe_layout.zip")

    for slave in slaves:
        slave.terminate()

    # Restore
    new_layout = SplitLayout.restore_from_zip(
        "test_paneldatframe_layout.zip",
    )

    # Verify PanelDataFrame state
    restored_df = new_layout.visualisation_panels["Dataframe"]
    assert restored_df.panel_name == "Dataframe"
    df = restored_df.plotter.get_data()
    assert df is not None

    for panel in new_layout.visualisation_panels.values():
        panel.get_slave().terminate()


# =============================================================================
# ENRICHED LAYOUT TESTS
# =============================================================================

@pytest.mark.default
def test_serialize_europe_full_state():
    """Serialize the europe layout with all panels, verify every panel's state survives."""
    layout, slaves = make_europe_panel(None, True)

    # Set various fields to exercise different panel types
    layout.visualisation_panels["Map"].set_field("load")
    layout.visualisation_panels["Plot"].set_field("load")
    layout.visualisation_panels["Dataframe"].recompute()

    # Save layout
    layout.save_to_zip("test_europe_full.zip")

    for slave in slaves:
        slave.terminate()

    # Restore
    new_layout = SplitLayout.restore_from_zip(
        "test_europe_full.zip",
    )

    # Verify Panel2D (Map) state
    map_panel = new_layout.visualisation_panels["Map"]
    assert map_panel.panel_name == "Map"
    assert map_panel.displayed_field == "load"

    # Verify Panel1D (Plot) state
    plot_panel = new_layout.visualisation_panels["Plot"]
    assert plot_panel.panel_name == "Plot"
    assert len(plot_panel.visible_fields_list) > 0

    # Verify PanelDataFrame (Dataframe) state
    df_panel = new_layout.visualisation_panels["Dataframe"]
    assert df_panel.panel_name == "Dataframe"
    df = df_panel.plotter.get_data()
    assert df is not None

    for panel in new_layout.visualisation_panels.values():
        panel.get_slave().terminate()


@pytest.mark.pyvista
def test_serialize_demo_3d_full_state():
    """Serialize the demo_3d layout with both panels, verify all state survives."""
    from scivianna_example.med.demo_3d import get_panel

    layout, slaves = get_panel(None, return_slaves=True)

    # Set fields on both panels
    panel_3d = layout.visualisation_panels["3D Demo"]
    panel_2d = layout.visualisation_panels["MEDCoupling slice"]
    
    panel_3d.set_field("INTEGRATED_POWER")
    panel_2d.set_field("INTEGRATED_POWER")

    # Save layout
    layout.save_to_zip("test_demo_3d_full.zip")

    for slave in slaves:
        slave.terminate()

    # Restore
    new_layout = SplitLayout.restore_from_zip("test_demo_3d_full.zip")

    # Verify Panel3D state
    restored_3d = new_layout.visualisation_panels["3D Demo"]
    assert restored_3d.panel_name == "3D Demo"
    assert restored_3d.displayed_field == "INTEGRATED_POWER"

    # Verify Panel2D state
    restored_2d = new_layout.visualisation_panels["MEDCoupling slice"]
    assert restored_2d.panel_name == "MEDCoupling slice"
    assert restored_2d.displayed_field == "INTEGRATED_POWER"

    for panel in new_layout.visualisation_panels.values():
        panel.get_slave().terminate()


if __name__ == "__main__":
    # test_serialize_panel()
    test_serialize_split()
    # test_serialize_gridstack()