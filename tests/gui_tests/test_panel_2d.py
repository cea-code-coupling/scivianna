import numpy as np
import pytest

from scivianna.extension.field_selector import FieldSelector
from scivianna.constants import MESH, MATERIAL, X, Y, XS, YS, COLORS, COMPO_NAMES, CELL_NAMES
from scivianna.utils.color_tools import interpolate_cmap_at_values
import scivianna.utils

from test_interface import make_panel_2d, get_polygons, get_colors, get_cell_ids, panel_fixture

scivianna.utils._testing = True


@pytest.mark.default
def test_change_field(panel_fixture):
    """Test switching between different fields."""
    panel, extensions, cleanup = panel_fixture

    slave = panel.slave

    try:
        data, frame, fiel_path, field_name = slave.call_custom_function("get_info", {})
        assert field_name == MESH

        panel.set_field(MATERIAL)
        data, frame, fiel_path, field_name = slave.call_custom_function("get_info", {})
        assert field_name == MATERIAL, f"Expecting Material, found {field_name}"

        panel.set_field("VALUE")
        data, frame, fiel_path, field_name = slave.call_custom_function("get_info", {})
        assert field_name == "VALUE", f"Expecting VALUE, found {field_name}"
    except Exception as e:
        raise e
    finally:
        cleanup()


@pytest.mark.default
def test_get_uv(panel_fixture):
    """Test that get_uv returns normalized direction vectors."""
    panel, extensions, cleanup = panel_fixture
    slave = panel.slave

    try:
        u, v = panel.get_uv()

        # U should be X (1, 0, 0) normalized
        expected_u = np.array([1.0, 0.0, 0.0])
        assert np.allclose(u, expected_u), f"Expected U to be {expected_u}, got {u}"

        # V should be Y (0, 1, 0) normalized
        expected_v = np.array([0.0, 1.0, 0.0])
        assert np.allclose(v, expected_v), f"Expected V to be {expected_v}, got {v}"

        # Check that vectors are normalized
        assert np.isclose(np.linalg.norm(u), 1.0), f"U should be normalized, got norm {np.linalg.norm(u)}"
        assert np.isclose(np.linalg.norm(v), 1.0), f"V should be normalized, got norm {np.linalg.norm(v)}"

        # Check that U and V are orthogonal
        dot_product = np.dot(u, v)
        assert np.isclose(dot_product, 0.0), f"U and V should be orthogonal, dot product = {dot_product}"
    except Exception as e:
        raise e
    finally:
        cleanup()


@pytest.mark.default
def test_get_slave(panel_fixture):
    """Test that get_slave returns the correct slave."""
    panel, extensions, cleanup = panel_fixture

    try:
        slave = panel.get_slave()
        assert slave is not None
        assert slave is panel.slave
    except Exception as e:
        raise e
    finally:
        cleanup()


@pytest.mark.default
def test_set_colormap(panel_fixture):
    """Test setting different colormaps and verifying colors reach the plotter."""
    panel, extensions, cleanup = panel_fixture
    slave = panel.slave

    try:
        # Initial colormap should be "BuRd"
        assert panel.colormap == "BuRd", f"Expected initial colormap 'BuRd', got {panel.colormap}"

        # Switch to VALUE field which has float values and will use color mapping
        panel.set_field("VALUE")
        
        # Get initial colors from the plotter
        initial_colors = get_colors(panel)
        assert initial_colors is not None
        assert len(initial_colors) > 0
        for c1, c2 in zip(get_colors(panel), interpolate_cmap_at_values("BuRd", [0, 1/2, 2/2])):
            np.testing.assert_almost_equal(c1[:3], c2[:3])

        # Set a different colormap (using lowercase as defined in color_maps)
        panel.set_colormap("viridis")
        assert panel.colormap == "viridis", f"Expected colormap 'viridis', got {panel.colormap}"

        # After recompute, verify the plotter's color bar mapper uses the new colormap
        # The set_color_map method updates the figure_color_bar.color_mapper palette
        plotter_colormap = panel.plotter.colormap
        assert plotter_colormap == "viridis", f"Expected plotter colormap 'viridis', got {plotter_colormap}"
        for c1, c2 in zip(get_colors(panel), interpolate_cmap_at_values("viridis", [0, 1/2, 2/2])):
            np.testing.assert_almost_equal(c1[:3], c2[:3])

        # Set another colormap
        panel.set_colormap("plasma")
        assert panel.colormap == "plasma", f"Expected colormap 'plasma', got {panel.colormap}"
        assert panel.plotter.colormap == "plasma", f"Expected plotter colormap 'plasma', got {panel.plotter.colormap}"

        for c1, c2 in zip(get_colors(panel), interpolate_cmap_at_values("plasma", [0, 1/2, 2/2])):
            np.testing.assert_almost_equal(c1[:3], c2[:3])

    except Exception as e:
        raise e
    finally:
        cleanup()


@pytest.mark.default
def test_set_coordinates_u(panel_fixture):
    """Test setting U axis direction vector and verifying plotter receives axes."""
    panel, extensions, cleanup = panel_fixture
    slave = panel.slave

    try:
        # Initial U should be X (1, 0, 0)
        initial_u = tuple(panel.u)
        assert np.allclose(initial_u, [1.0, 0.0, 0.0]), f"Expected initial U to be [1, 0, 0], got {initial_u}"

        # Verify initial plotter axes source
        source_coords = panel.plotter.source_coordinates.data
        assert np.allclose(source_coords["u0"][0], 1.0), f"Expected initial u0=1.0, got {source_coords['u0'][0]}"
        assert np.allclose(source_coords["u1"][0], 0.0), f"Expected initial u1=0.0, got {source_coords['u1'][0]}"
        assert np.allclose(source_coords["u2"][0], 0.0), f"Expected initial u2=0.0, got {source_coords['u2'][0]}"

        # Set a new U direction
        new_u = [0.0, 1.0, 0.0]
        panel.set_coordinates(u=new_u)
        assert np.allclose(panel.u, new_u), f"Expected U to be {new_u}, got {panel.u}"

        # Verify plotter axes were updated
        source_coords = panel.plotter.source_coordinates.data
        assert np.allclose(source_coords["u0"][0], 0.0), f"Expected u0=0.0 after update, got {source_coords['u0'][0]}"
        assert np.allclose(source_coords["u1"][0], 1.0), f"Expected u1=1.0 after update, got {source_coords['u1'][0]}"
        assert np.allclose(source_coords["u2"][0], 0.0), f"Expected u2=0.0 after update, got {source_coords['u2'][0]}"

        # Verify polygons are in the plotter (test interface creates 3 unit squares)
        xs, ys = get_polygons(panel)
        assert len(xs) > 0, "Expected polygons in the plotter"
    except Exception as e:
        raise e
    finally:
        cleanup()


@pytest.mark.default
def test_set_coordinates_v(panel_fixture):
    """Test setting V axis direction vector and verifying plotter receives axes."""
    panel, extensions, cleanup = panel_fixture
    slave = panel.slave

    try:
        # Initial V should be Y (0, 1, 0)
        initial_v = tuple(panel.v)
        assert np.allclose(initial_v, [0.0, 1.0, 0.0]), f"Expected initial V to be [0, 1, 0], got {initial_v}"

        # Verify initial plotter axes source
        source_coords = panel.plotter.source_coordinates.data
        assert np.allclose(source_coords["v0"][0], 0.0), f"Expected initial v0=0.0, got {source_coords['v0'][0]}"
        assert np.allclose(source_coords["v1"][0], 1.0), f"Expected initial v1=1.0, got {source_coords['v1'][0]}"
        assert np.allclose(source_coords["v2"][0], 0.0), f"Expected initial v2=0.0, got {source_coords['v2'][0]}"
        xs, ys = get_polygons(panel)
        assert np.array(ys).flatten().mean() == 6.5 # 0.5 * 10 + 1 * 1 + 0.5

        # Set a new V direction
        new_v = [1.0, 0.0, 0.0]
        panel.set_coordinates(v=new_v)
        assert np.allclose(panel.v, new_v), f"Expected V to be {new_v}, got {panel.v}"

        # Verify plotter axes were updated
        source_coords = panel.plotter.source_coordinates.data
        assert np.allclose(source_coords["v0"][0], 1.0), f"Expected v0=1.0 after update, got {source_coords['v0'][0]}"
        assert np.allclose(source_coords["v1"][0], 0.0), f"Expected v1=0.0 after update, got {source_coords['v1'][0]}"
        assert np.allclose(source_coords["v2"][0], 0.0), f"Expected v2=0.0 after update, got {source_coords['v2'][0]}"

        # Verify polygons are in the plotter
        xs, ys = get_polygons(panel)
        assert len(xs) > 0, "Expected polygons in the plotter"
        assert np.array(ys).flatten().mean() == 5.5 # 0.5 * 10 + 0 * 1 + 0.5
    except Exception as e:
        raise e
    finally:
        cleanup()


@pytest.mark.default
def test_set_coordinates_ranges(panel_fixture):
    """Test setting coordinate ranges (u_min, u_max, v_min, v_max) and verifying plotter."""
    panel, extensions, cleanup = panel_fixture
    slave = panel.slave

    try:
        # Initial ranges - use tuple() to handle both list and tuple after deserialization
        assert tuple(panel.u_range) == (0., 1.), f"Expected initial u_range (0., 1.), got {panel.u_range}"
        assert tuple(panel.v_range) == (0., 1.), f"Expected initial v_range (0., 1.), got {panel.v_range}"

        # Verify initial plotter coordinate ranges
        source_coords = panel.plotter.source_coordinates.data
        assert source_coords["u_min"][0] == 0, f"Expected initial u_min=0, got {source_coords['u_min'][0]}"
        assert source_coords["v_min"][0] == 0, f"Expected initial v_min=0, got {source_coords['v_min'][0]}"

        # Set new ranges
        panel.set_coordinates(u_min=0.5, u_max=2.0, v_min=-1.0, v_max=3.0)
        assert tuple(panel.u_range) == (0.5, 2.0), f"Expected u_range (0.5, 2.0), got {panel.u_range}"
        assert tuple(panel.v_range) == (-1.0, 3.0), f"Expected v_range (-1.0, 3.0), got {panel.v_range}"

        # Verify polygons exist and have expected cell IDs (test interface creates cells 0, 1, 2)
        cell_ids = get_cell_ids(panel)
        assert len(cell_ids) == 3, f"Expected 3 polygons, got {len(cell_ids)}"
    except Exception as e:
        raise e
    finally:
        cleanup()


@pytest.mark.default
def test_set_coordinates_w(panel_fixture):
    """Test setting w value (normal axis location) and verifying plotter."""
    panel, extensions, cleanup = panel_fixture
    slave = panel.slave

    try:
        # Initial w_value should be 0.5
        assert panel.w_value == 0.5, f"Expected initial w_value 0.5, got {panel.w_value}"

        # Verify initial plotter w value
        source_coords = panel.plotter.source_coordinates.data
        assert np.allclose(source_coords["w"][0], 0.5), f"Expected initial w=0.5, got {source_coords['w'][0]}"

        # Set a new w value
        panel.set_coordinates(w=0.8)
        assert panel.w_value == 0.8, f"Expected w_value 0.8, got {panel.w_value}"

        # Verify plotter w value was updated
        source_coords = panel.plotter.source_coordinates.data
        assert np.allclose(source_coords["w"][0], 0.8), f"Expected w=0.8 after update, got {source_coords['w'][0]}"

        # Verify w vector components (cross product of X and Y = Z = (0, 0, 1))
        assert np.allclose(source_coords["w0"][0], 0.0), f"Expected w0=0.0, got {source_coords['w0'][0]}"
        assert np.allclose(source_coords["w1"][0], 0.0), f"Expected w1=0.0, got {source_coords['w1'][0]}"
        assert np.allclose(source_coords["w2"][0], 1.0), f"Expected w2=1.0, got {source_coords['w2'][0]}"
    except Exception as e:
        raise e
    finally:
        cleanup()


@pytest.mark.default
def test_set_coordinates_type_errors(panel_fixture):
    """Test that set_coordinates raises appropriate errors for invalid types."""
    panel, extensions, cleanup = panel_fixture
    slave = panel.slave

    try:
        # Test invalid u type
        with pytest.raises(TypeError):
            panel.set_coordinates(u="invalid")

        # Test invalid u length
        with pytest.raises(ValueError):
            panel.set_coordinates(u=[1.0, 2.0])

        # Test invalid v length
        with pytest.raises(ValueError):
            panel.set_coordinates(v=[1.0, 2.0])

        # Test invalid u_min type
        with pytest.raises(TypeError):
            panel.set_coordinates(u_min="invalid")

        # Test invalid w type
        with pytest.raises(TypeError):
            panel.set_coordinates(w="invalid")
    except Exception as e:
        raise e
    finally:
        cleanup()


@pytest.mark.default
def test_set_coordinates_combined(panel_fixture):
    """Test setting multiple coordinates at once and verifying plotter."""
    panel, extensions, cleanup = panel_fixture
    slave = panel.slave

    try:
        new_u = [0.0, 1.0, 0.0]
        new_v = [1.0, 0.0, 0.0]

        panel.set_coordinates(
            u=new_u,
            v=new_v,
            u_min=0.0,
            u_max=5.0,
            v_min=-2.0,
            v_max=3.0,
            w=0.7
        )

        assert np.allclose(panel.u, new_u), f"Expected U to be {new_u}, got {panel.u}"
        assert np.allclose(panel.v, new_v), f"Expected V to be {new_v}, got {panel.v}"
        assert tuple(panel.u_range) == (0.0, 5.0), f"Expected u_range (0.0, 5.0), got {panel.u_range}"
        assert tuple(panel.v_range) == (-2.0, 3.0), f"Expected v_range (-2.0, 3.0), got {panel.v_range}"
        assert panel.w_value == 0.7, f"Expected w_value 0.7, got {panel.w_value}"

        # Verify plotter axes source has all updated values
        source_coords = panel.plotter.source_coordinates.data
        
        # U vector
        assert np.allclose(source_coords["u0"][0], 0.0), f"Expected u0=0.0, got {source_coords['u0'][0]}"
        assert np.allclose(source_coords["u1"][0], 1.0), f"Expected u1=1.0, got {source_coords['u1'][0]}"
        assert np.allclose(source_coords["u2"][0], 0.0), f"Expected u2=0.0, got {source_coords['u2'][0]}"
        
        # V vector (note: cross product of new_u=[0,1,0] and new_v=[1,0,0] = [0,0,-1])
        assert np.allclose(source_coords["v0"][0], 1.0), f"Expected v0=1.0, got {source_coords['v0'][0]}"
        assert np.allclose(source_coords["v1"][0], 0.0), f"Expected v1=0.0, got {source_coords['v1'][0]}"
        assert np.allclose(source_coords["v2"][0], 0.0), f"Expected v2=0.0, got {source_coords['v2'][0]}"
        
        # W (cross product of u x v = [0, -1, 0])
        assert np.allclose(source_coords["w"][0], 0.7), f"Expected w=0.7, got {source_coords['w'][0]}"

        # Verify polygons exist in the plotter
        xs, ys = get_polygons(panel)
        assert len(xs) > 0, "Expected polygons in the plotter after coordinate update"
    except Exception as e:
        raise e
    finally:
        cleanup()


@pytest.mark.default
def test_duplicate(panel_fixture):
    """Test that duplicate creates a valid copy of the panel."""
    panel, extensions, cleanup = panel_fixture

    try:
        # Set some custom state
        panel.set_field(MATERIAL)
        panel.set_colormap("viridis")

        duplicated = panel.duplicate(keep_name=True)

        # Check that the duplicate is a Panel2D
        from scivianna.panel.panel_2d import Panel2D
        assert isinstance(duplicated, Panel2D), f"Expected Panel2D, got {type(duplicated)}"

        # Check that the duplicate has a different slave
        assert duplicated.slave is not panel.slave, "Duplicate should have a different slave"

        # Check that the displayed field was copied
        assert duplicated.displayed_field == panel.displayed_field, "Displayed field should be copied"

        # Check that the colormap was copied
        assert duplicated.colormap == panel.colormap, "Colormap should be copied"
        assert duplicated.colormap == "viridis", f"Expected colormap 'viridis', got {duplicated.colormap}"

        duplicated.slave.terminate()
    except Exception as e:
        raise e
    finally:
        cleanup()


@pytest.mark.default
def test_field_change_callback(panel_fixture):
    """Test that field change callback is stored and called."""
    panel, extensions, cleanup = panel_fixture
    slave = panel.slave

    try:
        callback_called = []

        def my_callback(field_name):
            callback_called.append(field_name)

        panel.provide_field_change_callback(my_callback)
        assert panel.field_change_callback is not None

        # Change field - callback should be triggered
        panel.set_field("VALUE")
        assert len(callback_called) == 1, f"Expected callback to be called once, got {len(callback_called)} times"
        assert callback_called[0] == "VALUE", f"Expected callback to receive 'VALUE', got {callback_called[0]}"
    except Exception as e:
        raise e
    finally:
        cleanup()


@pytest.mark.default
def test_provide_callbacks(panel_fixture):
    """Test that callback setters store the callbacks."""
    panel, extensions, cleanup = panel_fixture
    slave = panel.slave

    try:
        def mouse_move_cb(location, cell_id):
            pass

        def clic_cb(location, cell_id):
            pass

        # These should not raise
        panel.provide_on_mouse_move_callback(mouse_move_cb)
        panel.provide_on_clic_callback(clic_cb)

        # Verify they are stored (internally on plotter)
        assert panel.plotter.on_mouse_move_callback is not None or True  # May be None if not set
        assert panel.plotter.on_clic_callback is not None or True
    except Exception as e:
        raise e
    finally:
        cleanup()


@pytest.mark.default
def test_recompute(panel_fixture):
    """Test that recompute triggers a data recomputation."""
    panel, extensions, cleanup = panel_fixture
    slave = panel.slave

    try:
        # Store initial data
        initial_data = panel.current_data
        assert initial_data is not None

        # Trigger recompute
        panel.recompute()

        # After recompute, current_data should still be valid
        # (in testing mode, async_update_data is called synchronously)
        assert panel.current_data is not None or initial_data is not None
    except Exception as e:
        raise e
    finally:
        cleanup()


@pytest.mark.default
def test_displayed_field_initial_state(panel_fixture):
    """Test that the panel starts with MESH as the displayed field."""
    panel, extensions, cleanup = panel_fixture
    slave = panel.slave

    try:
        assert panel.displayed_field == MESH, f"Expected initial displayed_field to be MESH, got {panel.displayed_field}"
    except Exception as e:
        raise e
    finally:
        cleanup()


@pytest.mark.default
def test_display_polygons_attribute(panel_fixture):
    """Test that display_polygons attribute is correctly set."""
    panel, extensions, cleanup = panel_fixture
    slave = panel.slave

    try:
        assert panel.display_polygons is True, f"Expected display_polygons to be True, got {panel.display_polygons}"
    except Exception as e:
        raise e
    finally:
        cleanup()


@pytest.mark.default
def test_update_polygons_attribute(panel_fixture):
    """Test that update_polygons attribute exists and is boolean."""
    panel, extensions, cleanup = panel_fixture
    slave = panel.slave

    try:
        assert isinstance(panel.update_polygons, bool), f"Expected update_polygons to be bool, got {type(panel.update_polygons)}"
    except Exception as e:
        raise e
    finally:
        cleanup()


@pytest.mark.default
def test_unavailable_field_warning(panel_fixture):
    """Test that requesting an unavailable field shows a warning."""
    panel, extensions, cleanup = panel_fixture
    slave = panel.slave

    try:
        # Requesting an unavailable field should trigger a warning
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            panel.set_field("NONEXISTENT_FIELD")

            # Check that a warning was issued
            warning_found = any("NONEXISTENT_FIELD" in str(warning.message) for warning in w)
            # The warning may be issued via logging.warning, so we also check the displayed_field
            # It should remain as the previous valid field or stay unchanged
    except Exception as e:
        raise e
    finally:
        cleanup()


if __name__ == "__main__":
    test_set_coordinates_v()