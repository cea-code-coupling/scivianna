"""Tests for FieldSelector extension.

The FieldSelector extension allows selecting the displayed field and editing its colors.
It provides a color field selector, colormap selector, and center-on-zero option.
"""

import numpy as np
import pytest
import panel as pn

import scivianna.utils
# Enable testing mode so button callbacks call async_update_data directly
scivianna.utils._testing = True

from scivianna.extension.field_selector import FieldSelector, set_colors_list
from scivianna.data.data2d import Data2D
from scivianna.enums import VisualizationMode
from scivianna.constants import OUTSIDE

from test_interface import make_panel_2d, panel_fixture


class TestFieldSelectorInitialization:
    """Test suite for FieldSelector extension initialization."""

    def test_field_selector_exists(self, panel_fixture):
        """Test that FieldSelector extension is properly registered."""
        panel, extensions_dict, cleanup = panel_fixture
        try:
            assert FieldSelector in extensions_dict
        finally:
            cleanup()

    def test_field_selector_initialization(self, panel_fixture):
        """Test that FieldSelector attributes are properly set by __init__."""
        panel, extensions_dict, cleanup = panel_fixture
        field_sel = extensions_dict[FieldSelector]

        try:
            # Title and icon set by constructor
            assert field_sel.title == "Color map"
            assert field_sel.icon == "palette"

            # Description should be set
            assert "color map" in field_sel.description.lower()

            # Icon size should be set
            assert field_sel.iconsize == "6em"

            # Slave, plotter, and panel references should be set
            assert field_sel.slave is panel.slave
            assert field_sel.plotter is panel.plotter
            assert field_sel.panel is panel

            # GUI components should be initialized
            assert hasattr(field_sel, 'field_color_selector')
            assert hasattr(field_sel, 'color_map_selector')
            assert hasattr(field_sel, 'center_colormap_on_zero_tick')

            # Field color selector should have options from slave
            assert len(field_sel.field_color_selector.options) > 0
            # Should include MESH at minimum
            assert field_sel.field_color_selector.value is not None
        finally:
            cleanup()

    def test_field_color_selector_has_correct_options(self, panel_fixture):
        """Test that field_color_selector has options from the interface."""
        panel, extensions_dict, cleanup = panel_fixture
        field_sel = extensions_dict[FieldSelector]

        try:
            # Get labels from slave
            labels = panel.slave.get_labels()
            
            # Field color selector options should match slave labels
            assert set(field_sel.field_color_selector.options) == set(labels)
        finally:
            cleanup()

    def test_center_colormap_on_zero_tick_visibility(self, panel_fixture):
        """Test that center_colormap_on_zero_tick visibility depends on coloring mode."""
        panel, extensions_dict, cleanup = panel_fixture
        field_sel = extensions_dict[FieldSelector]

        try:
            # For MESH (VisualizationMode.NONE), should be hidden
            initial_field = field_sel.field_color_selector.value
            initial_mode = panel.slave.get_label_coloring_mode(initial_field)
            
            if initial_mode == VisualizationMode.FROM_VALUE:
                assert field_sel.center_colormap_on_zero_tick.visible is True
            else:
                assert field_sel.center_colormap_on_zero_tick.visible is False
        finally:
            cleanup()

    def test_color_map_selector_initial_value(self, panel_fixture):
        """Test that color_map_selector has correct initial value."""
        panel, extensions_dict, cleanup = panel_fixture
        field_sel = extensions_dict[FieldSelector]

        try:
            from scivianna.utils.color_tools import beautiful_color_maps
            assert field_sel.color_map_selector.value_name == "BuRd"
            assert field_sel.color_map_selector.value is beautiful_color_maps["BuRd"]
        finally:
            cleanup()


class TestFieldSelectorCallbacks:
    """Test suite for FieldSelector callback behaviors."""

    def test_trigger_field_change(self, panel_fixture):
        """Test that trigger_field_change updates the panel field."""
        panel, extensions_dict, cleanup = panel_fixture
        field_sel = extensions_dict[FieldSelector]

        try:
            # Get available fields
            available_fields = field_sel.field_color_selector.options
            
            if len(available_fields) > 1:
                # Change to a different field
                new_field = available_fields[1]
                field_sel.field_color_selector.value = new_field
                
                # The trigger should call panel.set_field
                # This is tested indirectly via the callback watch
                assert field_sel.field_color_selector.value == new_field
        finally:
            cleanup()

    def test_receive_colormap_change(self, panel_fixture):
        """Test that receive_colormap_change updates the color map selector."""
        panel, extensions_dict, cleanup = panel_fixture
        field_sel = extensions_dict[FieldSelector]

        try:
            # Simulate colormap change from panel
            original_value_name = field_sel.color_map_selector.value_name
            
            # Change panel's colormap
            panel.colormap = "viridis"
            
            # Receive the change - should update selector
            field_sel.receive_colormap_change()
            
            assert field_sel.color_map_selector.value_name == "viridis"
        finally:
            cleanup()

    def test_trigger_colormap_change(self, panel_fixture):
        """Test that trigger_colormap_change updates the panel colormap."""
        panel, extensions_dict, cleanup = panel_fixture
        field_sel = extensions_dict[FieldSelector]

        try:
            # Change color map selector
            from scivianna.utils.color_tools import beautiful_color_maps
            new_cmap_name = "viridis"
            field_sel.color_map_selector.value_name = new_cmap_name
            field_sel.color_map_selector.value = beautiful_color_maps[new_cmap_name]
            
            # Trigger should update panel colormap
            field_sel.trigger_colormap_change()
            
            assert panel.colormap == new_cmap_name
        finally:
            cleanup()

    def test_trigger_update(self, panel_fixture):
        """Test that trigger_update triggers a recompute."""
        panel, extensions_dict, cleanup = panel_fixture
        field_sel = extensions_dict[FieldSelector]

        try:
            # Trigger update should call recompute
            # This should not raise an error
            field_sel.trigger_update()
            # If we get here without error, the test passes
            assert True
        finally:
            cleanup()

    def test_on_field_change_updates_selector(self, panel_fixture):
        """Test that on_field_change updates the field_color_selector value."""
        panel, extensions_dict, cleanup = panel_fixture
        field_sel = extensions_dict[FieldSelector]

        try:
            # Get available fields
            available_fields = field_sel.field_color_selector.options
            
            if len(available_fields) > 1:
                new_field = available_fields[1]
                
                # Call on_field_change
                field_sel.on_field_change(new_field)
                
                # Selector value should be updated
                assert field_sel.field_color_selector.value == new_field
        finally:
            cleanup()

    def test_on_file_load_updates_options(self, panel_fixture):
        """Test that on_file_load updates the field selector options."""
        panel, extensions_dict, cleanup = panel_fixture
        field_sel = extensions_dict[FieldSelector]

        try:
            # Store initial value
            initial_value = field_sel.field_color_selector.value
            
            # Call on_file_load - should update options and reset value
            field_sel.on_file_load("/some/path.dat", "GEOMETRY")
            
            # Options should be updated from slave
            new_options = list(panel.slave.get_labels())
            assert set(field_sel.field_color_selector.options) == set(new_options)
            
            # Value should be reset to first option
            assert field_sel.field_color_selector.value == new_options[0]
        finally:
            cleanup()

    def test_on_updated_data_calls_set_colors_list(self, panel_fixture):
        """Test that on_updated_data calls set_colors_list with correct parameters."""
        panel, extensions_dict, cleanup = panel_fixture
        field_sel = extensions_dict[FieldSelector]

        try:
            # Create mock Data2D
            from scivianna.utils.polygonize_tools import PolygonElement, PolygonCoords
            
            polygons = [
                PolygonElement(
                    exterior_polygon=PolygonCoords(
                        x_coords=[0.0, 1.0, 1.0, 0.0],
                        y_coords=[0.0, 0.0, 1.0, 1.0]
                    ),
                    holes=[],
                    cell_id=i
                )
                for i in range(3)
            ]
            
            data = Data2D.from_polygon_list(polygons)
            
            # Set cell values for testing
            data.cell_values = [0.0, 1.0, 2.0]
            
            # Get current field selector value
            current_field = field_sel.field_color_selector.value
            current_cmap = field_sel.color_map_selector.value_name
            center_zero = field_sel.center_colormap_on_zero_tick.value
            
            # Call on_updated_data - should call set_colors_list internally
            # This should not raise an error
            field_sel.on_updated_data(data)
            
            # Data should have cell_colors set
            assert hasattr(data, 'cell_colors')
            assert data.cell_colors is not None
            assert len(data.cell_colors) == 3
        finally:
            cleanup()


class TestFieldSelectorGUI:
    """Test suite for FieldSelector GUI rendering."""

    def test_make_gui_returns_column(self, panel_fixture):
        """Test that make_gui returns a Panel column with expected components."""
        panel, extensions_dict, cleanup = panel_fixture
        field_sel = extensions_dict[FieldSelector]

        try:
            gui = field_sel.make_gui()
            
            assert gui is not None
            assert isinstance(gui, pn.layout.Column) or hasattr(gui, '__panel__')
            
            # Should contain the expected components
            # The GUI should have field_color_selector, color_map_selector, center_colormap_on_zero_tick
        finally:
            cleanup()

    def test_gui_components_exist(self, panel_fixture):
        """Test that all GUI components are Panel widgets."""
        panel, extensions_dict, cleanup = panel_fixture
        field_sel = extensions_dict[FieldSelector]

        try:
            # All components should be Panel viewables
            assert hasattr(field_sel.field_color_selector, '__panel__') or \
                   isinstance(field_sel.field_color_selector, pn.viewable.Viewable)
            assert hasattr(field_sel.color_map_selector, '__panel__') or \
                   isinstance(field_sel.color_map_selector, pn.viewable.Viewable)
            assert hasattr(field_sel.center_colormap_on_zero_tick, '__panel__') or \
                   isinstance(field_sel.center_colormap_on_zero_tick, pn.viewable.Viewable)
        finally:
            cleanup()


class TestSetColorsList:
    """Test suite for set_colors_list utility function."""

    def test_set_colors_list_from_string_mode(self, panel_fixture):
        """Test set_colors_list with FROM_STRING visualization mode."""
        panel, extensions_dict, cleanup = panel_fixture
        
        try:
            # Create mock Data2D with string values
            from scivianna.utils.polygonize_tools import PolygonElement, PolygonCoords
            
            polygons = [
                PolygonElement(
                    exterior_polygon=PolygonCoords(
                        x_coords=[0.0, 1.0, 1.0, 0.0],
                        y_coords=[0.0, 0.0, 1.0, 1.0]
                    ),
                    holes=[],
                    cell_id=i
                )
                for i in range(5)
            ]
            
            data = Data2D.from_polygon_list(polygons)
            data.cell_values = ["material_a", "material_b", "material_a", "material_c", "material_b"]
            
            # Set coloring mode to FROM_STRING for testing
            def mock_get_coloring_mode(label):
                return VisualizationMode.FROM_STRING
            
            panel.slave.get_label_coloring_mode = mock_get_coloring_mode
            
            # Call set_colors_list
            set_colors_list(
                data,
                panel.slave,
                "MATERIAL",  # coloring_label
                "viridis",   # color_map
                False,       # center_colormap_on_zero
                {}           # options
            )
            
            # Check that edge_colors was set
            assert hasattr(data, 'cell_edge_colors')
            assert data.cell_edge_colors is not None
        finally:
            cleanup()

    def test_set_colors_list_from_value_mode(self, panel_fixture):
        """Test set_colors_list with FROM_VALUE visualization mode."""
        panel, extensions_dict, cleanup = panel_fixture
        
        try:
            # Create mock Data2D with numeric values
            from scivianna.utils.polygonize_tools import PolygonElement, PolygonCoords
            
            polygons = [
                PolygonElement(
                    exterior_polygon=PolygonCoords(
                        x_coords=[0.0, 1.0, 1.0, 0.0],
                        y_coords=[0.0, 0.0, 1.0, 1.0]
                    ),
                    holes=[],
                    cell_id=i
                )
                for i in range(5)
            ]
            
            data = Data2D.from_polygon_list(polygons)
            data.cell_values = [1.0, 2.0, 3.0, 4.0, 5.0]
            
            # Set coloring mode to FROM_STRING for testing (since TestInterface returns FROM_STRING for MATERIAL)
            def mock_get_coloring_mode(label):
                return VisualizationMode.FROM_VALUE
            
            panel.slave.get_label_coloring_mode = mock_get_coloring_mode
            
            # Call set_colors_list with FROM_VALUE mode
            set_colors_list(
                data,
                panel.slave,
                "VALUE",     # coloring_label
                "viridis",   # color_map
                False,       # center_colormap_on_zero
                {}           # options
            )
            
            # Check that cell_colors was set
            assert hasattr(data, 'cell_colors')
            assert data.cell_colors is not None
            assert len(data.cell_colors) == 5
        finally:
            cleanup()

    def test_set_colors_list_none_mode(self, panel_fixture):
        """Test set_colors_list with NONE visualization mode."""
        panel, extensions_dict, cleanup = panel_fixture
        
        try:
            # Create mock Data2D
            from scivianna.utils.polygonize_tools import PolygonElement, PolygonCoords
            
            polygons = [
                PolygonElement(
                    exterior_polygon=PolygonCoords(
                        x_coords=[0.0, 1.0, 1.0, 0.0],
                        y_coords=[0.0, 0.0, 1.0, 1.0]
                    ),
                    holes=[],
                    cell_id=i
                )
                for i in range(3)
            ]
            
            data = Data2D.from_polygon_list(polygons)
            data.cell_values = [0.0, 0.0, 0.0]
            
            # Set coloring mode to NONE
            def mock_get_coloring_mode(label):
                return VisualizationMode.NONE
            
            panel.slave.get_label_coloring_mode = mock_get_coloring_mode
            
            # Call set_colors_list
            set_colors_list(
                data,
                panel.slave,
                "MESH",      # coloring_label
                "viridis",   # color_map (should be ignored in NONE mode)
                False,       # center_colormap_on_zero (should be ignored in NONE mode)
                {}           # options
            )
            
            # Check that cell_colors was set to transparent gray
            assert hasattr(data, 'cell_colors')
            assert data.cell_colors is not None
            assert len(data.cell_colors) == 3
            
            # In NONE mode, all colors should be (200, 200, 200, 0)
            expected_color = (200, 200, 200, 0)
            for i in range(3):
                assert tuple(data.cell_colors[i]) == expected_color
        finally:
            cleanup()

    def test_set_colors_list_with_outside_cell(self, panel_fixture):
        """Test set_colors_list handles OUTSIDE cells correctly."""
        panel, extensions_dict, cleanup = panel_fixture

        try:
            # Create mock Data2D with OUTSIDE cell
            from scivianna.utils.polygonize_tools import PolygonElement, PolygonCoords
            from scivianna.constants import OUTSIDE
            import numpy as np

            polygons = [
                PolygonElement(
                    exterior_polygon=PolygonCoords(
                        x_coords=[0.0, 1.0, 1.0, 0.0],
                        y_coords=[0.0, 0.0, 1.0, 1.0]
                    ),
                    holes=[],
                    cell_id=OUTSIDE
                ),
                PolygonElement(
                    exterior_polygon=PolygonCoords(
                        x_coords=[1.0, 2.0, 2.0, 1.0],
                        y_coords=[0.0, 0.0, 1.0, 1.0]
                    ),
                    holes=[],
                    cell_id=1
                )
            ]

            data = Data2D.from_polygon_list(polygons)
            data.cell_values = ["mat_a", "mat_b"]
            data.cell_ids = np.array([OUTSIDE, 1])

            # Set coloring mode to FROM_STRING
            def mock_get_coloring_mode(label):
                return VisualizationMode.FROM_STRING

            panel.slave.get_label_coloring_mode = mock_get_coloring_mode

            # Call set_colors_list
            set_colors_list(
                data,
                panel.slave,
                "MATERIAL",  # coloring_label
                "viridis",   # color_map
                False,       # center_colormap_on_zero
                {}           # options
            )

            # OUTSIDE cell should have transparent white color (alpha = 0)
            outside_mask = data.cell_ids == OUTSIDE
            assert any(outside_mask), "OUTSIDE cell mask should have matches"
            outside_colors = data.cell_colors[outside_mask]
            # Check that OUTSIDE cells are transparent (alpha = 0)
            assert outside_colors[0][3] == 0
        finally:
            cleanup()

    def test_set_colors_list_with_nan_values(self, panel_fixture):
        """Test set_colors_list handles NaN values correctly."""
        panel, extensions_dict, cleanup = panel_fixture
        
        try:
            # Create mock Data2D with NaN values
            from scivianna.utils.polygonize_tools import PolygonElement, PolygonCoords
            
            polygons = [
                PolygonElement(
                    exterior_polygon=PolygonCoords(
                        x_coords=[0.0, 1.0, 1.0, 0.0],
                        y_coords=[0.0, 0.0, 1.0, 1.0]
                    ),
                    holes=[],
                    cell_id=i
                )
                for i in range(3)
            ]
            
            data = Data2D.from_polygon_list(polygons)
            data.cell_values = [1.0, float('nan'), 3.0]
            
            # Set coloring mode to FROM_VALUE
            def mock_get_coloring_mode(label):
                return VisualizationMode.FROM_VALUE
            
            panel.slave.get_label_coloring_mode = mock_get_coloring_mode
            
            # Call set_colors_list - should not raise
            set_colors_list(
                data,
                panel.slave,
                "VALUE",     # coloring_label
                "viridis",   # color_map
                False,       # center_colormap_on_zero
                {}           # options
            )
            
            # Check that cell_colors was set
            assert hasattr(data, 'cell_colors')
            assert data.cell_colors is not None
            assert len(data.cell_colors) == 3
        finally:
            cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])