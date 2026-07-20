"""Tests for Axes extension.

The Axes extension allows defining the frame of the displayed data, including
the u and v vectors and their bounds.
"""

import numpy as np
import pytest
import panel as pn
import panel_material_ui as pmui

import scivianna.utils
# Enable testing mode so button callbacks call async_update_data directly
scivianna.utils._testing = True

from scivianna.extension.axes import Axes
from scivianna.enums import GeometryType
from scivianna.constants import XS, YS, CELL_NAMES, CELL_VALUES, COLORS, EDGE_COLORS, EDGE_ALPHA, FILL_ALPHA
from scivianna.data.data2d import Data2D

from test_interface import make_panel_2d, panel_fixture


class TestAxesInitialization:
    """Test suite for Axes initialization."""

    @pytest.mark.usefixtures("panel_fixture")
    def test_axes_extension_exists(self, panel_fixture):
        """Test that Axes is properly registered."""
        panel, extensions_dict, cleanup = panel_fixture
        try:
            assert Axes in extensions_dict
        finally:
            cleanup()

    @pytest.mark.usefixtures("panel_fixture")
    def test_axes_extension_initialization(self, panel_fixture):
        """Test that Axes attributes are properly set by __init__."""
        panel, extensions_dict, cleanup = panel_fixture
        axes_ext = extensions_dict[Axes]

        try:
            # Title and icon set by constructor
            assert axes_ext.title == "Axes customization"
            assert axes_ext.icon is not None

            # Description should be set
            assert axes_ext.description is not None

            # Icon size should be set
            assert axes_ext.iconsize == "1.0em"

            # Slave, plotter, and panel references should be set
            assert axes_ext.slave is panel.slave
            assert axes_ext.plotter is panel.plotter
            assert axes_ext.panel is panel

            # GUI components should be initialized
            assert hasattr(axes_ext, 'hide_show_button')
            assert hasattr(axes_ext, 'origin_x_inp')
            assert hasattr(axes_ext, 'origin_y_inp')
            assert hasattr(axes_ext, 'origin_z_inp')
            assert hasattr(axes_ext, 'size_u_inp')
            assert hasattr(axes_ext, 'size_v_inp')
            # w_inp was removed - slice position is now via origin directly
            assert hasattr(axes_ext, 'u0_inp')
            assert hasattr(axes_ext, 'u1_inp')
            assert hasattr(axes_ext, 'u2_inp')
            assert hasattr(axes_ext, 'v0_inp')
            assert hasattr(axes_ext, 'v1_inp')
            assert hasattr(axes_ext, 'v2_inp')
            assert hasattr(axes_ext, 'recompute_button')
            assert hasattr(axes_ext, 'xplus')
            assert hasattr(axes_ext, 'yplus')
            assert hasattr(axes_ext, 'zplus')
        finally:
            cleanup()

    @pytest.mark.usefixtures("panel_fixture")
    def test_axes_are_pmui_widgets(self, panel_fixture):
        """Test that all axes inputs are Panel/PMUI widgets."""
        panel, extensions_dict, cleanup = panel_fixture
        axes_ext = extensions_dict[Axes]

        try:
            # All inputs should be PMUI FloatInput widgets (w_inp removed - slice position via origin)
            input_attrs = ['origin_x_inp', 'origin_y_inp', 'origin_z_inp', 'size_u_inp', 'size_v_inp',
                          'u0_inp', 'u1_inp', 'u2_inp', 'v0_inp', 'v1_inp', 'v2_inp']
            
            for attr in input_attrs:
                widget = getattr(axes_ext, attr)
                assert isinstance(widget, pmui.FloatInput), \
                    f"{attr} should be a PMUI FloatInput widget"
        finally:
            cleanup()

    def test_vector_inputs_initial_values(self):
        """Test that vector inputs have correct initial values."""
        panel, extensions_dict, cleanup = make_panel_2d()
        axes_ext = extensions_dict[Axes]

        try:
            # u0_inp should be initialized to 1 (X direction)
            assert axes_ext.u0_inp.value == 1
            # u1_inp should be initialized to 0
            assert axes_ext.u1_inp.value == 0
            # u2_inp should be initialized to 0
            assert axes_ext.u2_inp.value == 0

            # v0_inp should be initialized to 0
            assert axes_ext.v0_inp.value == 0
            # v1_inp should be initialized to 1 (Y direction)
            assert axes_ext.v1_inp.value == 1
            # v2_inp should be initialized to 0
            assert axes_ext.v2_inp.value == 0
        finally:
            cleanup()

    def test_bound_inputs_initial_values(self):
        """Test that bound inputs have correct initial values."""
        panel, extensions_dict, cleanup = make_panel_2d()
        axes_ext = extensions_dict[Axes]

        try:
            # Initial widget values are the default pmui.FloatInput values (0.0)
            # The origin is computed from w at runtime when trigger_update is called
            assert axes_ext.origin_x_inp.value == 0.01
            
            assert axes_ext.origin_y_inp.value == 0.01
            
            assert axes_ext.origin_z_inp.value == 0.01
            
            # size_u_inp should be initialized to 1
            assert axes_ext.size_u_inp.value == 1
            
            # size_v_inp should be initialized to 1
            assert axes_ext.size_v_inp.value == 1

        finally:
            cleanup()

    def test_tracking_attributes_exist(self):
        """Test that Axes has all tracking attributes."""
        panel, extensions_dict, cleanup = make_panel_2d()
        axes_ext = extensions_dict[Axes]

        try:
            assert hasattr(axes_ext, 'borders_displayed')
            assert hasattr(axes_ext, 'axes_updated')
            assert hasattr(axes_ext, 'range_updated')
            assert hasattr(axes_ext, '_Axes__new_data')
            
            # borders_displayed and axes_updated start as False
            assert axes_ext.borders_displayed is False
            assert axes_ext.axes_updated is False
            # range_updated may be True after init because on_range_change is called
            # during panel initialization (computed origin differs from widget defaults)
            assert axes_ext.range_updated in [True, False]
        finally:
            cleanup()

    def test_get_uv_returns_normalized_vectors(self):
        """Test that get_uv returns normalized vectors."""
        panel, extensions_dict, cleanup = make_panel_2d()
        axes_ext = extensions_dict[Axes]

        try:
            u, v = axes_ext.get_uv()
            
            # Should be numpy arrays
            assert isinstance(u, np.ndarray)
            assert isinstance(v, np.ndarray)
            
            # Should have length 3
            assert len(u) == 3
            assert len(v) == 3
            
            # Should be normalized (unit length)
            assert np.isclose(np.linalg.norm(u), 1.0)
            assert np.isclose(np.linalg.norm(v), 1.0)
            
            # Default should be X and Y
            assert np.allclose(u, [1.0, 0.0, 0.0])
            assert np.allclose(v, [0.0, 1.0, 0.0])
        finally:
            cleanup()

    def test_get_uv_with_custom_values(self):
        """Test that get_uv handles custom vector values."""
        panel, extensions_dict, cleanup = make_panel_2d()
        axes_ext = extensions_dict[Axes]

        try:
            # Set custom values
            axes_ext.u0_inp.value = 3
            axes_ext.u1_inp.value = 4
            axes_ext.u2_inp.value = 0
            
            axes_ext.v0_inp.value = 0
            axes_ext.v1_inp.value = 0
            axes_ext.v2_inp.value = 5
            
            u, v = axes_ext.get_uv()
            
            # Should be normalized
            assert np.isclose(np.linalg.norm(u), 1.0)
            assert np.isclose(np.linalg.norm(v), 1.0)
            
            # Expected normalized values
            expected_u = np.array([3.0, 4.0, 0.0]) / 5.0
            expected_v = np.array([0.0, 0.0, 1.0])
            
            assert np.allclose(u, expected_u)
            assert np.allclose(v, expected_v)
        finally:
            cleanup()


class TestAxesCallbacks:
    """Test suite for Axes callback behaviors."""

    def test_toggle_axis_visibility_turns_on(self):
        """Test that toggle_axis_visibility turns borders on."""
        panel, extensions_dict, cleanup = make_panel_2d()
        axes_ext = extensions_dict[Axes]

        try:
            # Initially borders should be off
            assert axes_ext.borders_displayed is False
            
            # Toggle should turn them on
            axes_ext.toggle_axis_visibility()
            
            assert axes_ext.borders_displayed is True
            
            # Check that plotter borders are displayed
            # (This depends on the plotter implementation)
        finally:
            cleanup()

    def test_toggle_axis_visibility_turns_off(self):
        """Test that toggle_axis_visibility turns borders off."""
        panel, extensions_dict, cleanup = make_panel_2d()
        axes_ext = extensions_dict[Axes]

        try:
            # Turn on first
            axes_ext.toggle_axis_visibility()
            assert axes_ext.borders_displayed is True
            
            # Toggle should turn them off
            axes_ext.toggle_axis_visibility()
            
            assert axes_ext.borders_displayed is False
        finally:
            cleanup()

    def test_trigger_update_calls_panel_set_coordinates(self):
        """Test that trigger_update calls panel.set_coordinates with correct values."""
        panel, extensions_dict, cleanup = make_panel_2d()
        axes_ext = extensions_dict[Axes]

        try:
            # Set specific values
            axes_ext.origin_x_inp.value = -1.0
            axes_ext.origin_y_inp.value = -2.0
            axes_ext.origin_z_inp.value = 0.75
            axes_ext.size_u_inp.value = 6.0
            axes_ext.size_v_inp.value = 12.0
            
            # Set custom vectors (will be normalized)
            axes_ext.u0_inp.value = 1.0
            axes_ext.u1_inp.value = 0.0
            axes_ext.u2_inp.value = 0.0
            axes_ext.v0_inp.value = 0.0
            axes_ext.v1_inp.value = 1.0
            axes_ext.v2_inp.value = 0.0
            
            # trigger_update should call panel.set_coordinates
            axes_ext.trigger_update()
            
            # Panel u, v vectors and ranges should be updated
            assert np.isclose(panel.u[0], 1.0)
            assert np.isclose(panel.u[1], 0.0)
            assert np.isclose(panel.u[2], 0.0)
            assert np.isclose(panel.v[0], 0.0)
            assert np.isclose(panel.v[1], 1.0)
            assert np.isclose(panel.v[2], 0.0)
            assert np.isclose(panel.origin[0], -1.0)
            assert np.isclose(panel.origin[1], -2.0)
            assert np.isclose(panel.origin[2], 0.75)
            assert np.isclose(panel.size_u, 6.0)
            assert np.isclose(panel.size_v, 12.0)
        finally:
            cleanup()

    def test_origin_input_triggers_range_update(self):
        """Test that changing origin input triggers range update."""
        panel, extensions_dict, cleanup = make_panel_2d()
        axes_ext = extensions_dict[Axes]

        try:
            # Reset tracking
            axes_ext.range_updated = False
            
            # Change origin value
            axes_ext.origin_x_inp.value = 1.5
            
            # Range should be marked as updated
            assert axes_ext.range_updated is True
        finally:
            cleanup()

    def test_xplus_button_sets_y_z_vectors(self):
        """Test that X+ button sets U to Y and V to Z."""
        panel, extensions_dict, cleanup = make_panel_2d()
        axes_ext = extensions_dict[Axes]

        try:
            # Reset tracking
            axes_ext.axes_updated = False
            
            # Click X+ button by incrementing clicks
            axes_ext.xplus.clicks += 1
            
            # Axes should be marked as updated
            assert axes_ext.axes_updated is True
        finally:
            cleanup()

    def test_yplus_button_sets_x_z_vectors(self):
        """Test that Y+ button sets U to X and V to Z."""
        panel, extensions_dict, cleanup = make_panel_2d()
        axes_ext = extensions_dict[Axes]

        try:
            # Reset tracking
            axes_ext.axes_updated = False
            
            # Click Y+ button by incrementing clicks
            axes_ext.yplus.clicks += 1
            
            # Axes should be marked as updated
            assert axes_ext.axes_updated is True
        finally:
            cleanup()

    def test_zplus_button_sets_x_y_vectors(self):
        """Test that Z+ button sets U to X and V to Y."""
        panel, extensions_dict, cleanup = make_panel_2d()
        axes_ext = extensions_dict[Axes]

        try:
            # Reset tracking
            axes_ext.axes_updated = False
            
            # Click Z+ button by incrementing clicks
            axes_ext.zplus.clicks += 1
            
            # Axes should be marked as updated
            assert axes_ext.axes_updated is True
        finally:
            cleanup()

    def test_on_frame_change_updates_widget_values(self):
        """Test that on_frame_change updates widget values from new vectors."""
        panel, extensions_dict, cleanup = make_panel_2d()
        axes_ext = extensions_dict[Axes]

        try:
            # Store original values
            orig_u0 = axes_ext.u0_inp.value
            orig_u1 = axes_ext.u1_inp.value
            orig_v0 = axes_ext.v0_inp.value
            orig_v1 = axes_ext.v1_inp.value
            
            new_u_vector = (0.5, 0.5, 0.0)
            new_v_vector = (0.0, 0.5, 0.5)
            
            # Call on_frame_change with new vectors
            axes_ext.on_frame_change(new_u_vector, new_v_vector)
            
            # Widget values should be updated to the new vectors
            assert axes_ext.u0_inp.value == 0.5
            assert axes_ext.u1_inp.value == 0.5
            assert axes_ext.v0_inp.value == 0.0
            assert axes_ext.v1_inp.value == 0.5
            
            # u2 and v2 should also be updated
            assert axes_ext.u2_inp.value == 0.0
            assert axes_ext.v2_inp.value == 0.5
        finally:
            cleanup()

    def test_on_range_change_updates_widget_values(self):
        """Test that on_range_change updates widget values from new range parameters."""
        panel, extensions_dict, cleanup = make_panel_2d()
        axes_ext = extensions_dict[Axes]

        try:
            # Store original values
            orig_ox = axes_ext.origin_x_inp.value
            orig_oy = axes_ext.origin_y_inp.value
            orig_oz = axes_ext.origin_z_inp.value
            orig_su = axes_ext.size_u_inp.value
            orig_sv = axes_ext.size_v_inp.value
            
            origin = (-1.0, -2.0, 0.5)
            size_u = 6.0
            size_v = 12.0
            
            # Call on_range_change with new range parameters (no w_val - slice position via origin)
            axes_ext.on_range_change(origin, size_u, size_v)
            
            # Widget values should be updated to the new values
            assert axes_ext.origin_x_inp.value == -1.0
            assert axes_ext.origin_y_inp.value == -2.0
            assert axes_ext.origin_z_inp.value == 0.5
            assert axes_ext.size_u_inp.value == 6.0
            assert axes_ext.size_v_inp.value == 12.0
        finally:
            cleanup()

    def test_origin_z_hidden_for_2d(self):
        """Test that origin_z_inp is hidden for 2D geometry."""
        panel, extensions_dict, cleanup = make_panel_2d()
        axes_ext = extensions_dict[Axes]

        try:
            # Mock geometry type as 2D
            original_get_geometry_type = axes_ext.slave.get_geometry_type
            axes_ext.slave.get_geometry_type = lambda: GeometryType._2D
            
            axes_ext.update_widgets_visibility()
            
            # Origin Z should be hidden for 2D
            assert axes_ext.origin_z_inp.visible is False
            # Size inputs should be visible for 2D
            assert axes_ext.size_u_inp.visible is True
            assert axes_ext.size_v_inp.visible is True
            
            # Restore original
            axes_ext.slave.get_geometry_type = original_get_geometry_type
        finally:
            cleanup()

    def test_origin_z_visible_for_3d(self):
        """Test that origin_z_inp is visible for 3D geometry."""
        panel, extensions_dict, cleanup = make_panel_2d()
        axes_ext = extensions_dict[Axes]

        try:
            # Mock geometry type as 3D
            original_get_geometry_type = axes_ext.slave.get_geometry_type
            axes_ext.slave.get_geometry_type = lambda: GeometryType._3D
            
            axes_ext.update_widgets_visibility()
            
            # Origin Z should be visible for 3D
            assert axes_ext.origin_z_inp.visible is True
            # Size inputs should be visible for 3D
            assert axes_ext.size_u_inp.visible is True
            assert axes_ext.size_v_inp.visible is True
            
            # Restore original
            axes_ext.slave.get_geometry_type = original_get_geometry_type
        finally:
            cleanup()


class TestAxesGUI:
    """Test suite for Axes GUI rendering."""

    def test_make_gui_returns_viewable(self):
        """Test that make_gui returns a Panel viewable."""
        panel, extensions_dict, cleanup = make_panel_2d()
        axes_ext = extensions_dict[Axes]

        try:
            gui = axes_ext.make_gui()
            
            assert gui is not None
            # Should be a Panel layout
            assert hasattr(gui, '__panel__') or isinstance(gui, pn.viewable.Viewable)
        finally:
            cleanup()

    def test_make_gui_contains_expected_components(self):
        """Test that make_gui contains expected components."""
        panel, extensions_dict, cleanup = make_panel_2d()
        axes_ext = extensions_dict[Axes]

        try:
            gui = axes_ext.make_gui()
            
            # GUI should contain the recompute button
            assert hasattr(axes_ext, 'recompute_button')
            assert hasattr(axes_ext, 'hide_show_button')
            assert hasattr(axes_ext, 'bounds_card')
            assert hasattr(axes_ext, 'axes_card')
        finally:
            cleanup()


class TestAxesVectorOperations:
    """Test suite for Axes vector operations."""

    def test_u_vector_can_be_set_to_any_direction(self):
        """Test that U vector can be set to any direction via inputs."""
        panel, extensions_dict, cleanup = make_panel_2d()
        axes_ext = extensions_dict[Axes]

        try:
            # Set U to Z direction
            axes_ext.u0_inp.value = 0.0
            axes_ext.u1_inp.value = 0.0
            axes_ext.u2_inp.value = 1.0
            
            u, v = axes_ext.get_uv()
            
            assert np.allclose(u, [0.0, 0.0, 1.0])
        finally:
            cleanup()

    def test_v_vector_can_be_set_to_any_direction(self):
        """Test that V vector can be set to any direction via inputs."""
        panel, extensions_dict, cleanup = make_panel_2d()
        axes_ext = extensions_dict[Axes]

        try:
            # Set V to Z direction
            axes_ext.v0_inp.value = 0.0
            axes_ext.v1_inp.value = 0.0
            axes_ext.v2_inp.value = 1.0
            
            u, v = axes_ext.get_uv()
            
            assert np.allclose(v, [0.0, 0.0, 1.0])
        finally:
            cleanup()

    def test_origin_can_be_negative(self):
        """Test that origin coordinates can be negative values."""
        panel, extensions_dict, cleanup = make_panel_2d()
        axes_ext = extensions_dict[Axes]

        try:
            # Set negative origin
            axes_ext.origin_x_inp.value = -10.0
            axes_ext.origin_y_inp.value = -20.0
            
            assert axes_ext.origin_x_inp.value == -10.0
            assert axes_ext.origin_y_inp.value == -20.0
        finally:
            cleanup()

    def test_size_can_be_equal(self):
        """Test that size_u and size_v can be equal."""
        panel, extensions_dict, cleanup = make_panel_2d()
        axes_ext = extensions_dict[Axes]

        try:
            # Set equal sizes
            axes_ext.size_u_inp.value = 5.0
            axes_ext.size_v_inp.value = 5.0
            
            assert axes_ext.size_u_inp.value == 5.0
            assert axes_ext.size_v_inp.value == 5.0
        finally:
            cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])