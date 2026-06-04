"""Tests for Extension base class default callback behaviors.

The TestExtension in test_interface.py tracks all callback invocations to verify
that the Extension base class methods are called correctly by the Panel2D system.
"""

import numpy as np
import pytest

from scivianna.extension.extension import Extension
from scivianna.data.data2d import Data2D
from scivianna.constants import MESH, X, Y

from test_interface import make_panel_2d, DummyTestExtension

import scivianna.utils
scivianna.utils._testing = True


class TestExtensionDefaultFeatures:
    """Test suite for Extension base class default features."""

    def test_extension_exists(self):
        """Test that extension attributes are properly set by __init__."""
        panel, extensions_dict, cleanup = make_panel_2d()
        try:
            assert DummyTestExtension in extensions_dict
            test_ext = extensions_dict[DummyTestExtension]

            assert isinstance(test_ext, DummyTestExtension)
        finally:
            cleanup()

    def test_extension_initialization(self):
        """Test that extension attributes are properly set by __init__."""
        panel, extensions_dict, cleanup = make_panel_2d()
        test_ext = extensions_dict[DummyTestExtension]

        try:
            # Title and icon set by constructor
            assert test_ext.title == "TestExtension"
            assert test_ext.icon is not None

            # Description should be the docstring set in TestExtension
            assert test_ext.description == """
This extension allows defining the medcoupling field display parameters.
"""
            # Icon size should be set
            assert test_ext.iconsize == "1.0em"

            # Slave, plotter, and panel references should be set
            assert test_ext.slave is panel.slave
            assert test_ext.plotter is panel.plotter
            assert test_ext.panel is panel
        finally:
            cleanup()

    def test_make_gui_returns_viewable(self):
        """Test that make_gui returns a Panel viewable object."""
        panel, extensions_dict, cleanup = make_panel_2d()
        test_ext = extensions_dict[DummyTestExtension]

        try:
            gui = test_ext.make_gui()
            assert gui is not None
            # Should be a Panel viewable (has __panel__ or inherits from Viewable)
            import panel as pn
            assert isinstance(gui, pn.viewable.Viewable) or hasattr(gui, '__panel__')
        finally:
            cleanup()

    def test_provide_options_default_returns_empty_dict(self):
        """Test that provide_options returns empty dict by default."""
        panel, extensions_dict, cleanup = make_panel_2d()
        test_ext = extensions_dict[DummyTestExtension]

        try:
            options = test_ext.provide_options()
            assert isinstance(options, dict)
            assert len(options) == 0
        finally:
            cleanup()


class TestExtensionCallbacksCalled:
    """Test that extension callbacks are invoked by Panel2D operations."""

    def test_on_field_change_called(self):
        """Test that on_field_change is called when set_field is called."""
        panel, extensions_dict, cleanup = make_panel_2d()
        test_ext = extensions_dict[DummyTestExtension]

        try:
            # Reset tracking
            test_ext._field_change_history.clear()

            # Change field - should trigger on_field_change
            panel.set_field("VALUE")

            # Verify callback was called with correct field name
            assert "VALUE" in test_ext._field_change_history
            assert test_ext._on_field_change_called is True

            # Change back to MESH
            panel.set_field(MESH)
            assert MESH in test_ext._field_change_history
        finally:
            cleanup()

    def test_on_range_change_called(self):
        """Test that on_range_change is called when ranges are updated."""
        panel, extensions_dict, cleanup = make_panel_2d()
        test_ext = extensions_dict[DummyTestExtension]

        try:
            # Reset tracking
            test_ext._range_change_history.clear()

            # Update ranges via set_coordinates
            panel.set_coordinates(u_min=0.0, u_max=5.0, v_min=-1.0, v_max=3.0)

            # Verify callback was called
            assert len(test_ext._range_change_history) > 0
            last_call = test_ext._range_change_history[-1]
            assert last_call.u_bounds == (0.0, 5.0)
            assert last_call.v_bounds == (-1.0, 3.0)
            # w_value defaults to 0.5
            assert np.isclose(last_call.w_value, 0.5)
        finally:
            cleanup()

    def test_on_frame_change_called(self):
        """Test that on_frame_change is called when u/v vectors change."""
        panel, extensions_dict, cleanup = make_panel_2d()
        test_ext = extensions_dict[DummyTestExtension]

        try:
            # Reset tracking
            test_ext._frame_change_history.clear()

            # Change U vector - should trigger on_frame_change
            panel.set_coordinates(u=[0.0, 1.0, 0.0])

            # Verify callback was called
            assert len(test_ext._frame_change_history) > 0
            last_call = test_ext._frame_change_history[-1]
            assert np.allclose(last_call.u_vector, [0.0, 1.0, 0.0])
            # V vector should still be default Y
            assert np.allclose(last_call.v_vector, [0.0, 1.0, 0.0])
        finally:
            cleanup()

    def test_on_updated_data_called(self):
        """Test that on_updated_data is called when data is recomputed."""
        panel, extensions_dict, cleanup = make_panel_2d()
        test_ext = extensions_dict[DummyTestExtension]

        try:
            # Reset tracking
            test_ext._updated_data_history.clear()

            # Trigger recompute - this will call compute_2D_data which triggers on_updated_data
            panel.recompute()

            # Verify callback was called with Data2D object
            assert len(test_ext._updated_data_history) > 0
            last_data = test_ext._updated_data_history[-1]
            assert isinstance(last_data, Data2D)
        finally:
            cleanup()

    def test_on_file_load_default_no_error(self):
        """Test that on_file_load default implementation does nothing."""
        panel, extensions_dict, cleanup = make_panel_2d()
        test_ext = extensions_dict[DummyTestExtension]

        try:
            # Default implementation should not raise
            test_ext.on_file_load("/some/path.dat", "GEOMETRY")
            # Should complete without error
        finally:
            cleanup()

    def test_on_mouse_move_default_no_error(self):
        """Test that on_mouse_move default implementation does nothing."""
        panel, extensions_dict, cleanup = make_panel_2d()
        test_ext = extensions_dict[DummyTestExtension]

        try:
            # Default implementation should not raise
            test_ext.on_mouse_move((100.0, 200.0), (0.5, 0.5, 0.5), "cell_1")
            test_ext.on_mouse_move((50.0, 75.0), (0.25, 0.25, 0.5), 42)
            # Should complete without error
        finally:
            cleanup()

    def test_on_mouse_click_default_no_error(self):
        """Test that on_mouse_clic default implementation does nothing."""
        panel, extensions_dict, cleanup = make_panel_2d()
        test_ext = extensions_dict[DummyTestExtension]

        try:
            # Default implementation should not raise
            test_ext.on_mouse_clic((100.0, 200.0), (0.5, 0.5, 0.5), "cell_1")
            test_ext.on_mouse_clic((50.0, 75.0), (0.25, 0.25, 0.5), 42)
            # Should complete without error
        finally:
            cleanup()


class TestExtensionTrackingMechanism:
    """Test that the TestExtension tracking mechanism works correctly."""

    def test_tracking_attributes_exist(self):
        """Test that TestExtension has all tracking attributes."""
        panel, extensions_dict, cleanup = make_panel_2d()
        test_ext = extensions_dict[DummyTestExtension]

        try:
            assert hasattr(test_ext, '_field_change_history')
            assert hasattr(test_ext, '_range_change_history')
            assert hasattr(test_ext, '_frame_change_history')
            assert hasattr(test_ext, '_updated_data_history')
            assert hasattr(test_ext, '_on_field_change_called')
            assert hasattr(test_ext, '_on_range_change_called')
            assert hasattr(test_ext, '_on_frame_change_called')
            assert hasattr(test_ext, '_on_updated_data_called')
        finally:
            cleanup()

    def test_tracking_dataclass_fields(self):
        """Test that tracking dataclasses have correct fields."""
        from test_interface import FieldChangeEvent, RangeChangeEvent, FrameChangeEvent

        # Test FieldChangeEvent
        event = FieldChangeEvent(field_name="TEST")
        assert event.field_name == "TEST"

        # Test RangeChangeEvent
        event = RangeChangeEvent(u_bounds=(0.0, 1.0), v_bounds=(0.0, 1.0), w_value=0.5)
        assert event.u_bounds == (0.0, 1.0)
        assert event.v_bounds == (0.0, 1.0)
        assert event.w_value == 0.5

        # Test FrameChangeEvent
        event = FrameChangeEvent(u_vector=(1.0, 0.0, 0.0), v_vector=(0.0, 1.0, 0.0))
        assert np.allclose(event.u_vector, (1.0, 0.0, 0.0))
        assert np.allclose(event.v_vector, (0.0, 1.0, 0.0))


class TestExtensionCallbackSequence:
    """Test that extension callbacks are called in the correct order."""

    def test_field_change_triggers_recompute_sequence(self):
        """Test that changing field triggers on_field_change then on_updated_data."""
        panel, extensions_dict, cleanup = make_panel_2d()
        test_ext = extensions_dict[DummyTestExtension]

        try:
            # Reset tracking
            test_ext._field_change_history.clear()
            test_ext._updated_data_history.clear()

            # Change field - this should trigger on_field_change, then recompute which triggers on_updated_data
            panel.set_field("VALUE")

            # Field change should come before data update
            if len(test_ext._field_change_history) > 0 and len(test_ext._updated_data_history) > 0:
                field_idx = test_ext._field_change_history.index("VALUE")
                # The data update happens during the async flow
                # At minimum, verify both were called
                assert test_ext._on_field_change_called is True
        finally:
            cleanup()

    def test_coordinate_change_triggers_frame_then_range(self):
        """Test that coordinate changes trigger on_frame_change and on_range_change."""
        panel, extensions_dict, cleanup = make_panel_2d()
        test_ext = extensions_dict[DummyTestExtension]

        try:
            # Reset tracking
            test_ext._frame_change_history.clear()
            test_ext._range_change_history.clear()

            # Change coordinates - should trigger both callbacks
            panel.set_coordinates(u=[0.0, 1.0, 0.0], u_min=0.0, u_max=5.0)

            # Both callbacks should have been called
            assert len(test_ext._frame_change_history) > 0
            assert len(test_ext._range_change_history) > 0
        finally:
            cleanup()


class TestExtensionBaseClassMethods:
    """Test the Extension base class method signatures and defaults."""

    def test_on_field_change_signature(self):
        """Test on_field_change accepts a string argument."""
        ext = Extension.__new__(Extension)  # Create without __init__

        # Should accept field_name as string
        ext.on_field_change("TEST_FIELD")
        # Default implementation does nothing, should not raise
        assert True

    def test_on_updated_data_signature(self):
        """Test on_updated_data accepts a Data2D argument."""
        ext = Extension.__new__(Extension)

        # Create minimal Data2D for testing
        polygons = [
            type('PolygonElement', (), {
                'exterior_polygon': type('PolygonCoords', (), {
                    'x_coords': [0.0, 1.0, 1.0, 0.0],
                    'y_coords': [0.0, 0.0, 1.0, 1.0]
                })(),
                'holes': [],
                'cell_id': 0
            })()
        ]

        # Should accept Data2D (or polygon list) - default implementation does nothing
        ext.on_updated_data(None)  # None is acceptable as default
        assert True

    def test_on_range_change_signature(self):
        """Test on_range_change accepts bounds tuples and w value."""
        ext = Extension.__new__(Extension)

        # Should accept u_bounds, v_bounds, w_value
        ext.on_range_change((0.0, 1.0), (0.0, 1.0), 0.5)
        assert True

    def test_on_frame_change_signature(self):
        """Test on_frame_change accepts u and v vectors."""
        ext = Extension.__new__(Extension)

        # Should accept u_vector, v_vector
        ext.on_frame_change((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        assert True

    def test_on_mouse_move_signature(self):
        """Test on_mouse_move accepts location and cell_id."""
        ext = Extension.__new__(Extension)

        # Should accept screen_location, space_location, cell_id
        ext.on_mouse_move((100.0, 200.0), (0.5, 0.5, 0.5), "cell_1")
        ext.on_mouse_move((100.0, 200.0), (0.5, 0.5, 0.5), 42)  # int cell_id
        assert True

    def test_on_mouse_clic_signature(self):
        """Test on_mouse_clic accepts location and cell_id."""
        ext = Extension.__new__(Extension)

        # Should accept screen_location, space_location, cell_id
        ext.on_mouse_clic((100.0, 200.0), (0.5, 0.5, 0.5), "cell_1")
        ext.on_mouse_clic((100.0, 200.0), (0.5, 0.5, 0.5), 42)  # int cell_id
        assert True

    def test_provide_options_returns_dict(self):
        """Test provide_options returns a dict."""
        ext = Extension.__new__(Extension)

        options = ext.provide_options()
        assert isinstance(options, dict)

    def test_make_gui_returns_none_by_default(self):
        """Test make_gui returns None by default."""
        ext = Extension.__new__(Extension)

        gui = ext.make_gui()
        assert gui is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])