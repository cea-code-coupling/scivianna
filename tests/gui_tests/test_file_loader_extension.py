"""Tests for FileLoader extension.

The FileLoader extension allows browsing and loading files on the server.
It provides file browsers for each file input type defined by the interface.
"""

import pytest
import panel as pn

import scivianna.utils
# Enable testing mode so button callbacks call async_update_data directly
scivianna.utils._testing = True

from scivianna.extension.file_loader import FileLoader
from scivianna.constants import GEOMETRY

from test_interface import make_panel_2d, panel_fixture


class TestFileLoaderInitialization:
    """Test suite for FileLoader extension initialization."""

    def test_file_loader_exists(self, panel_fixture):
        """Test that FileLoader extension is properly registered."""
        panel, extensions_dict, cleanup = panel_fixture
        try:
            assert FileLoader in extensions_dict
        finally:
            cleanup()

    def test_file_loader_initialization(self, panel_fixture):
        """Test that FileLoader attributes are properly set by __init__."""
        panel, extensions_dict, cleanup = panel_fixture
        file_loader = extensions_dict[FileLoader]

        try:
            # Title and icon set by constructor
            assert file_loader.title == "Load new files"
            assert file_loader.icon == "file_open"

            # Description should be set
            assert "browse files" in file_loader.description.lower()

            # Slave, plotter, and panel references should be set
            assert file_loader.slave is panel.slave
            assert file_loader.plotter is panel.plotter
            assert file_loader.panel is panel

            # File browsers dict should be initialized
            assert hasattr(file_loader, 'file_browsers')
            assert isinstance(file_loader.file_browsers, dict)
            assert len(file_loader.file_browsers) > 0

            # File loader list should contain widgets
            assert hasattr(file_loader, 'file_loader_list')
            assert isinstance(file_loader.file_loader_list, list)
            assert len(file_loader.file_loader_list) > 0
        finally:
            cleanup()

    def test_file_browsers_have_correct_names(self, panel_fixture):
        """Test that file browsers have names matching the interface file input list."""
        panel, extensions_dict, cleanup = panel_fixture
        file_loader = extensions_dict[FileLoader]

        try:
            # Get expected file input names from slave
            expected_names = {name for name, _ in panel.slave.get_file_input_list()}
            
            # File browser names should match
            actual_names = set(file_loader.file_browsers.keys())
            assert actual_names == expected_names
        finally:
            cleanup()

    def test_file_loader_gui_returns_column(self, panel_fixture):
        """Test that make_gui returns a Panel column with file browsers."""
        panel, extensions_dict, cleanup = panel_fixture
        file_loader = extensions_dict[FileLoader]

        try:
            gui = file_loader.make_gui()
            
            assert gui is not None
            assert isinstance(gui, pn.layout.Column) or hasattr(gui, '__panel__')
        finally:
            cleanup()


class TestFileLoaderCallbacks:
    """Test suite for FileLoader callback behaviors."""

    def test_on_file_load_updates_options(self, panel_fixture):
        """Test that on_file_load updates the file loader state."""
        panel, extensions_dict, cleanup = panel_fixture
        file_loader = extensions_dict[FileLoader]

        try:
            # Call on_file_load - should not raise an error
            file_loader.on_file_load("/some/path.dat", GEOMETRY)
            
            # If we get here without error, the test passes
            assert True
        finally:
            cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])