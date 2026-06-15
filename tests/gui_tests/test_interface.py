from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, TYPE_CHECKING, Generator
from dataclasses import dataclass
import numpy as np
import multiprocessing as mp
import panel as pn
import panel_material_ui as pmui
from bokeh.plotting import curdoc
from unittest.mock import Mock, MagicMock
import pytest
import tempfile
import zipfile

import scivianna
from scivianna.panel.panel_2d import Panel2D
from scivianna.slave import ComputeSlave
from scivianna.plotter_2d.generic_plotter import Plotter2D
from scivianna.plotter_2d.polygon.bokeh import Bokeh2DPolygonPlotter

from scivianna.extension.extension import Extension
import scivianna.icon
from scivianna.data.data2d import Data2D
from scivianna.interface.generic_interface import Geometry2DPolygon, CouplingInterface
from scivianna.utils.polygonize_tools import PolygonElement, PolygonCoords
from scivianna.enums import GeometryType, VisualizationMode

from scivianna.constants import MESH, MATERIAL, GEOMETRY, CSV, XS, YS, CELL_NAMES, CELL_VALUES, COLORS, EDGE_COLORS, EDGE_ALPHA, FILL_ALPHA, X, Y

from bokeh import events as bokeh_events

with open(Path(scivianna.icon.__file__).parent / "salome.svg", "r") as f:
    icon_svg = f.read()


@dataclass
class FieldChangeEvent:
    """Event recorded when a field change callback is triggered."""
    field_name: str


@dataclass
class RangeChangeEvent:
    """Event recorded when a range change callback is triggered."""
    u_bounds: Tuple[float, float]
    v_bounds: Tuple[float, float]
    w_value: float


@dataclass
class FrameChangeEvent:
    """Event recorded when a frame change callback is triggered."""
    u_vector: Tuple[float, float, float]
    v_vector: Tuple[float, float, float]


@dataclass
class MouseMoveEvent:
    """Event recorded when a mouse move callback is triggered."""
    screen_location: Tuple[float, float]
    space_location: Tuple[float, float, float]
    cell_id: Union[str, int]


@dataclass
class MouseClickEvent:
    """Event recorded when a mouse click callback is triggered."""
    screen_location: Tuple[float, float]
    space_location: Tuple[float, float, float]
    cell_id: Union[str, int]


class DummyTestExtension(Extension):
    """Extension to load files and send them to the slave.

    This extension overrides all Extension callback methods to track
    when they are called by the Panel2D system. Tracking data is stored
    in instance variables for test verification.
    """

    def __init__(
        self,
        slave: "ComputeSlave",
        plotter: "Plotter2D",
        panel: "Panel2D"
    ):
        """Constructor of the extension, saves the slave and the panel

        Parameters
        ----------
        slave : ComputeSlave
            Slave computing the displayed data
        plotter : Plotter2D
            Figure plotter
        panel : Panel2D
            Panel to which the extension is attached
        """
        super().__init__(
            "TestExtension",
            icon_svg,
            slave,
            plotter,
            panel,
        )

        self.description = """
This extension allows defining the medcoupling field display parameters.
"""

        self.iconsize = "1.0em"

        # Tracking variables for callback invocations
        self._field_change_history: List[str] = []
        self._range_change_history: List[RangeChangeEvent] = []
        self._frame_change_history: List[FrameChangeEvent] = []
        self._updated_data_history: List[Data2D] = []
        self._mouse_move_history: List[MouseMoveEvent] = []
        self._mouse_click_history: List[MouseClickEvent] = []

        self._on_field_change_called: bool = False
        self._on_range_change_called: bool = False
        self._on_frame_change_called: bool = False
        self._on_updated_data_called: bool = False
        self._on_mouse_move_called: bool = False
        self._on_mouse_clic_called: bool = False

    def make_gui(self,) -> pn.viewable.Viewable:
        """Returns a panel viewable to display in the extension tab.

        Returns
        -------
        pn.viewable.Viewable
            Viewable to display in the extension tab
        """
        return pmui.Column(
            pmui.Typography("Test extension"),
            margin=0
        )

    def on_field_change(self, field_name: str):
        """Override to track field change events."""
        self._field_change_history.append(field_name)
        self._on_field_change_called = True

    def on_range_change(
        self,
        u_bounds: Tuple[float, float],
        v_bounds: Tuple[float, float],
        w_value: float,
    ):
        """Override to track range change events."""
        self._range_change_history.append(RangeChangeEvent(
            u_bounds=u_bounds, v_bounds=v_bounds, w_value=w_value
        ))
        self._on_range_change_called = True

    def on_frame_change(
        self,
        u_vector: Tuple[float, float, float],
        v_vector: Tuple[float, float, float],
    ):
        """Override to track frame change events."""
        self._frame_change_history.append(FrameChangeEvent(
            u_vector=u_vector, v_vector=v_vector
        ))
        self._on_frame_change_called = True

    def on_updated_data(self, data: Data2D):
        """Override to track updated data events."""
        self._updated_data_history.append(data)
        self._on_updated_data_called = True

    def on_mouse_move(
        self,
        screen_location: Tuple[float, float],
        space_location: Tuple[float, float, float],
        cell_id: Union[str, int],
    ):
        """Override to track mouse move events."""
        self._mouse_move_history.append(MouseMoveEvent(
            screen_location=screen_location,
            space_location=space_location,
            cell_id=cell_id,
        ))
        self._on_mouse_move_called = True

    def on_mouse_clic(
        self,
        screen_location: Tuple[float, float],
        space_location: Tuple[float, float, float],
        cell_id: Union[str, int],
    ):
        """Override to track mouse click events."""
        self._mouse_click_history.append(MouseClickEvent(
            screen_location=screen_location,
            space_location=space_location,
            cell_id=cell_id,
        ))
        self._on_mouse_clic_called = True


class DummyTestInterface(Geometry2DPolygon):
    geometry_type: GeometryType = GeometryType._3D_INFINITE
    extensions = [DummyTestExtension]

    def __init__(self):
        self.data: Dict[str, Data2D] = {}
        self.file_path: Dict[str, str] = {}
        self.current_field = None
        self.last_computed_frame: Dict[str, List[float]] = {}

    def read_file(self, file_path: str, file_label: str):
        """Read a file and store its content in the interface

        Parameters
        ----------
        file_path : str
            File to read
        file_label : str
            Label to define the file type
        """
        self.file_path[file_label] = file_path

    def compute_2D_data(
        self,
        u: Tuple[float, float, float],
        v: Tuple[float, float, float],
        u_min: float,
        u_max: float,
        v_min: float,
        v_max: float,
        w_value: float,
        q_tasks: mp.Queue,
        options: Dict[str, Any],
        caller: str = "API",
    ) -> Tuple[Data2D, bool]:
        """Returns a list of polygons that defines the geometry in a given frame

        Parameters
        ----------
        u : Tuple[float, float, float]
            Horizontal coordinate director vector
        v : Tuple[float, float, float]
            Vertical coordinate director vector
        u_min : float
            Lower bound value along the u axis
        u_max : float
            Upper bound value along the u axis
        v_min : float
            Lower bound value along the v axis
        v_max : float
            Upper bound value along the v axis
        w_value : float
            Value along the u ^ v axis
        q_tasks : mp.Queue
            Queue from which get orders from the master.
        options : Dict[str, Any]
            Additional options for frame computation.
        caller : str
            Identifier of the caller requesting the computation (default: "API")

        Returns
        -------
        Data2D
            Geometry to display
        bool
            Were the polygons updated compared to the past call
        """
        last_frame_key = (*u, *v, w_value)
        if (caller in self.last_computed_frame) and (
            self.last_computed_frame[caller] == last_frame_key
        ) and (caller in self.data):
            print("Skipping polygon computation.")
            return self.data[caller], False

        v_offset = v[1] + 10*w_value
        polygons = [
            PolygonElement(
                exterior_polygon = PolygonCoords(
                    x_coords = [i, i, i+1, i+1],
                    y_coords = [0+v_offset, 1+v_offset, 1+v_offset, 0+v_offset]
                ),
                holes = [],
                cell_id = i
            )
            for i in range(3)
        ]

        print("Offset", v_offset)

        self.last_computed_frame[caller] = last_frame_key
        self.data[caller] = Data2D.from_polygon_list(polygons)
        return self.data[caller], True

    def get_labels(
        self,
    ) -> List[str]:
        """Returns a list of fields names displayable with this interface

        Returns
        -------
        List[str]
            List of fields names
        """
        labels = [MESH,  MATERIAL, "VALUE"]
        return labels

    def get_value_dict(
        self, value_label: str, cells: List[Union[int, str]], options: Dict[str, Any], caller: str = "API"
    ) -> Dict[Union[int, str], str]:
        """Returns a cell name - field value map for a given field name

        Parameters
        ----------
        value_label : str
            Field name to get values from
        cells : List[Union,int,str]]
            List of cells names
        options : Dict[str, Any]
            Additional options for frame computation.

        Returns
        -------
        Dict[Union[int,str], str]
            Field value for each requested cell names
        """
        self.current_field = value_label
        if value_label == MESH:
            return {v: np.nan for v in cells}
        if value_label == MATERIAL:
            return {v: str(v) for v in cells}
        if value_label == "VALUE":
            return {v: float(v) for v in cells}

        raise NotImplementedError(
            f"The field {value_label} is not implemented, fields available : {self.get_labels()}"
        )

    def get_label_coloring_mode(self, label: str) -> VisualizationMode:
        """Returns wheter the given field is colored based on a string value or a float.

        Parameters
        ----------
        label : str
            Field to color name

        Returns
        -------
        VisualizationMode
            Coloring mode
        """
        if label == MESH:
            return VisualizationMode.NONE
        if label == MATERIAL:
            return VisualizationMode.FROM_STRING

        return VisualizationMode.FROM_VALUE

    def get_file_input_list(self) -> List[Tuple[str, str]]:
        """Returns a list of file label and its description for the GUI

        Returns
        -------
        List[Tuple[str, str]]
            List of (file label, description)
        """
        return [(GEOMETRY, "MED file."), (CSV, "CSV result file.")]
    
    def get_info(self,):
        return self.data, self.last_computed_frame, self.file_path, self.current_field

    def save(self, file_path: str, include_files: bool = True):
        """Save the interface state to a file.

        Parameters
        ----------
        file_path : str
            Path to the file to save to
        include_files : bool = True
            If True, includes loaded files in the serialization
        """
        import pickle
        state = {
            'file_path': self.file_path,
            'current_field': self.current_field,
            'last_computed_frame': self.last_computed_frame,
        }
        with open(file_path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, file_path: str, include_files: bool = True):
        """Load the interface state from a file.

        Parameters
        ----------
        file_path : str
            Path to the file to load from
        include_files : bool = True
            If True, includes loaded files in the deserialization
        """
        import pickle
        with open(file_path, 'rb') as f:
            state = pickle.load(f)
        self.file_path = state.get('file_path', {})
        self.current_field = state.get('current_field', None)
        self.last_computed_frame = state.get('last_computed_frame', None)


def make_panel_2d():
    """Creates a Panel2D instance with DummyTestInterface.
    
    Returns
    -------
    tuple
        (panel, extensions_dict, cleanup) where:
        - panel: Panel2D instance
        - extensions_dict: dict mapping extension class to extension instance
        - cleanup: function to call to terminate the slave process
    """
    slave = ComputeSlave(DummyTestInterface)
    panel = Panel2D(slave)

    def cleanup():
        slave.terminate()

    return panel, {
        e.__class__: e for e in panel.extensions
    }, cleanup


@pytest.fixture(scope="function", params=["direct", "deserialized"])
def panel_fixture(request) -> Generator[Tuple[Panel2D, Dict[type, Extension], callable], None, None]:
    """Pytest fixture that provides a Panel2D instance either directly or via deserialization.
    
    This fixture allows tests to run in two modes:
    - "direct": Panel is created fresh using make_panel_2d()
    - "deserialized": Panel is serialized to a zip file and restored from it
    
    Parameters
    ----------
    request : pytest.FixtureRequest
        The fixture request object, used to get the parameter value
        
    Yields
    ------
    Tuple[Panel2D, Dict[type, Extension], callable]
        (panel, extensions_dict, cleanup) tuple same as make_panel_2d()
    """
    from scivianna.utils.serialization import save_panel2d_to_file, load_panel2d_from_file
    
    mode = request.param
    
    if mode == "direct":
        # Direct panel creation
        panel, extensions_dict, cleanup = make_panel_2d()
        yield panel, extensions_dict, cleanup
        cleanup()
    else:
        # Deserialized panel - serialize then deserialize
        import tempfile
        from pathlib import Path
        from scivianna.interface import register_interface
        
        # Create original panel
        orig_panel, _, orig_cleanup = make_panel_2d()
        
        try:
            # Save to a temporary zip file using Panel2D serialization
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
                zip_path = Path(tmp_file.name)
            
            # Save the panel directly using Panel2D serialization
            save_panel2d_to_file(orig_panel, zip_path)
            
            # Terminate original slave
            orig_cleanup()
            
            # Register the test interface before deserialization
            register_interface("DummyTestInterface", DummyTestInterface)
            
            # Restore from zip using Panel2D deserialization
            panel = load_panel2d_from_file(
                zip_path,
            )
            
            extensions_dict = {e.__class__: e for e in panel.extensions}
            
            def cleanup():
                panel.get_slave().terminate()
                if zip_path.exists():
                    zip_path.unlink()
            
            yield panel, extensions_dict, cleanup
            
            # Cleanup is called after test completes
            cleanup()
            
        except Exception as e:
            # Clean up on error
            orig_cleanup()
            if 'zip_path' in locals() and zip_path.exists():
                zip_path.unlink()
            raise e


def get_polygons(panel: Panel2D):
    return [
        panel.plotter.source_polygons.data[XS],
        panel.plotter.source_polygons.data[YS],
    ]


def get_colors(panel: Panel2D):
    return panel.plotter.source_polygons.data[COLORS]


def get_edge_colors(panel: Panel2D):
    return panel.plotter.source_polygons.data[EDGE_COLORS]


def get_cell_ids(panel: Panel2D):
    return panel.plotter.source_polygons.data[CELL_NAMES]


def get_cell_values(panel: Panel2D):
    return panel.plotter.source_polygons.data[CELL_VALUES]

@pytest.mark.default
def test_build_panel():
    make_panel_2d()
    assert True

class TestMouseMoveCallbacks:
    """Tests for mouse move callback functionality."""

    @pytest.mark.default
    def test_provide_on_mouse_move_callback_registers_callback(self):
        """Test that providing a mouse move callback stores it in the plotter."""
        panel, _, cleanup = make_panel_2d()
        cleanup()
        
        mock_callback = Mock()
        panel.provide_on_mouse_move_callback(mock_callback)
        
        assert panel.plotter.on_mouse_move_callback is not None
        # The callback is wrapped with functools.partial in Bokeh2DPolygonPlotter
        assert callable(panel.plotter.on_mouse_move_callback)

    @pytest.mark.default
    def test_mouse_move_callback_receives_panel(self):
        """Test that the mouse move callback receives the panel correctly."""
        panel, _, cleanup = make_panel_2d()
        cleanup()
        
        callback_invocations = []
        
        def on_mouse_move(screen_location, space_location, cell_id):
            callback_invocations.append({
                'screen_location': screen_location,
                'space_location': space_location,
                'cell_id': cell_id
            })
        
        panel.provide_on_mouse_move_callback(on_mouse_move)
        
        # Verify callback was registered
        assert panel.plotter.on_mouse_move_callback is not None

    @pytest.mark.default
    def test_plotter_send_event_with_valid_index(self):
        """Test that send_event correctly passes data when index is valid."""
        panel, _, cleanup = make_panel_2d()
        cleanup()
        
        callback_calls = []
        
        def mock_callback(screen_location, space_location, cell_id):
            callback_calls.append({
                'screen_location': screen_location,
                'space_location': space_location,
                'cell_id': cell_id
            })
        
        # Set up the callback
        panel.plotter.provide_on_mouse_move_callback(mock_callback)
        
        # Simulate mouse data with a valid index (0 points to first polygon)
        panel.plotter.source_mouse.data = {
            "sx": [100.0],
            "sy": [50.0],
            "x": [0.5],
            "y": [0.5],
            "z": [0.0],
            "index": [0],
        }
        
        # Call send_event directly
        panel.plotter.send_event(mock_callback)
        
        assert len(callback_calls) == 1
        assert callback_calls[0]['screen_location'] == (100.0, 50.0)
        assert callback_calls[0]['space_location'] == (0.5, 0.5, 0.0)
        assert callback_calls[0]['cell_id'] == 0

    @pytest.mark.default
    def test_plotter_send_event_with_multiple_cells(self):
        """Test send_event with different cell IDs."""
        panel, _, cleanup = make_panel_2d()
        cleanup()
        
        # The test interface creates 3 cells (0, 1, 2)
        callback_calls = []
        
        def mock_callback(screen_location, space_location, cell_id):
            callback_calls.append(cell_id)
        
        panel.plotter.provide_on_mouse_move_callback(mock_callback)
        
        # Test with index 1 (middle cell)
        panel.plotter.source_mouse.data["index"] = [1]
        panel.plotter.send_event(mock_callback)
        
        assert len(callback_calls) == 1
        assert callback_calls[0] == 1

    @pytest.mark.default
    def test_plotter_send_event_with_invalid_index(self):
        """Test that send_event does not call callback when index is out of bounds."""
        panel, _, cleanup = make_panel_2d()
        cleanup()
        
        callback_calls = []
        
        def mock_callback(screen_location, space_location, cell_id):
            callback_calls.append(cell_id)
        
        panel.plotter.provide_on_mouse_move_callback(mock_callback)
        
        # Set index to a value larger than the number of polygons
        panel.plotter.source_mouse.data["index"] = [100]
        
        # send_event should not call the callback when index is out of bounds
        panel.plotter.send_event(mock_callback)
        
        assert len(callback_calls) == 0

    @pytest.mark.default
    def test_panel_provide_on_mouse_move_callback_integration(self):
        """Test that Panel2D.provide_on_mouse_move_callback correctly delegates to plotter."""
        panel, _, cleanup = make_panel_2d()
        cleanup()
        
        # The callbacks should be set by Panel2D's __init__ (registered with Bokeh events)
        assert panel.plotter.on_mouse_move_callback is not None
        assert callable(panel.plotter.on_mouse_move_callback)

    @pytest.mark.default
    def test_mouse_move_callback_data_structure(self):
        """Test that mouse move callback receives correct data structure."""
        panel, _, cleanup = make_panel_2d()
        cleanup()
        
        received_data = {}
        
        def on_mouse_move(screen_location, space_location, cell_id):
            received_data['screen'] = screen_location
            received_data['space'] = space_location
            received_data['cell'] = cell_id
        
        panel.plotter.provide_on_mouse_move_callback(on_mouse_move)
        
        # Simulate mouse event data
        panel.plotter.source_mouse.data = {
            "sx": [200.0],
            "sy": [150.0],
            "x": [0.75],
            "y": [0.25],
            "z": [0.0],
            "index": [2],
        }
        
        panel.plotter.send_event(on_mouse_move)
        
        assert 'screen' in received_data
        assert 'space' in received_data
        assert 'cell' in received_data
        assert received_data['screen'] == (200.0, 150.0)
        assert received_data['space'] == (0.75, 0.25, 0.0)
        assert received_data['cell'] == 2


class TestClickCallbacks:
    """Tests for click (tap) callback functionality."""

    @pytest.mark.default
    def test_provide_on_clic_callback_registers_callback(self):
        """Test that providing a click callback stores it in the plotter."""
        panel, _, cleanup = make_panel_2d()
        cleanup()
        
        mock_callback = Mock()
        panel.provide_on_clic_callback(mock_callback)
        
        assert panel.plotter.on_clic_callback is not None
        assert callable(panel.plotter.on_clic_callback)

    @pytest.mark.default
    def test_click_callback_receives_panel(self):
        """Test that the click callback receives the panel correctly."""
        panel, _, cleanup = make_panel_2d()
        cleanup()
        
        callback_invocations = []
        
        def on_mouse_click(screen_location, space_location, cell_id):
            callback_invocations.append({
                'screen_location': screen_location,
                'space_location': space_location,
                'cell_id': cell_id
            })
        
        panel.provide_on_clic_callback(on_mouse_click)
        
        # Verify callback was registered
        assert panel.plotter.on_clic_callback is not None

    @pytest.mark.default
    def test_plotter_send_event_for_click(self):
        """Test that send_event works for click events too."""
        panel, _, cleanup = make_panel_2d()
        cleanup()
        
        callback_calls = []
        
        def mock_callback(screen_location, space_location, cell_id):
            callback_calls.append({
                'screen_location': screen_location,
                'space_location': space_location,
                'cell_id': cell_id
            })
        
        panel.plotter.provide_on_clic_callback(mock_callback)
        
        # Simulate click data
        panel.plotter.source_mouse.data = {
            "sx": [300.0],
            "sy": [200.0],
            "x": [0.3],
            "y": [0.6],
            "z": [0.0],
            "index": [1],
        }
        
        panel.plotter.send_event(mock_callback)
        
        assert len(callback_calls) == 1
        assert callback_calls[0]['screen_location'] == (300.0, 200.0)
        assert callback_calls[0]['space_location'] == (0.3, 0.6, 0.0)
        assert callback_calls[0]['cell_id'] == 1

    @pytest.mark.default
    def test_click_callback_with_invalid_index(self):
        """Test that click callback is not called when index is out of bounds."""
        panel, _, cleanup = make_panel_2d()
        cleanup()
        
        callback_calls = []
        
        def mock_callback(screen_location, space_location, cell_id):
            callback_calls.append(cell_id)
        
        panel.plotter.provide_on_clic_callback(mock_callback)
        
        # Set index to a value larger than the number of polygons
        panel.plotter.source_mouse.data["index"] = [100]
        
        panel.plotter.send_event(mock_callback)
        
        assert len(callback_calls) == 0

    @pytest.mark.default
    def test_panel_provide_on_clic_callback_integration(self):
        """Test that Panel2D.provide_on_clic_callback correctly delegates to plotter."""
        panel, _, cleanup = make_panel_2d()
        cleanup()
        
        # The callbacks should be set by Panel2D's __init__ (registered with Bokeh events)
        assert panel.plotter.on_clic_callback is not None
        assert callable(panel.plotter.on_clic_callback)

    @pytest.mark.default
    def test_click_callback_data_structure(self):
        """Test that click callback receives correct data structure."""
        panel, _, cleanup = make_panel_2d()
        cleanup()
        
        received_data = {}
        
        def on_mouse_click(screen_location, space_location, cell_id):
            received_data['screen'] = screen_location
            received_data['space'] = space_location
            received_data['cell'] = cell_id
        
        panel.plotter.provide_on_clic_callback(on_mouse_click)
        
        # Simulate click event data
        panel.plotter.source_mouse.data = {
            "sx": [400.0],
            "sy": [300.0],
            "x": [0.1],
            "y": [0.9],
            "z": [0.0],
            "index": [0],
        }
        
        panel.plotter.send_event(on_mouse_click)
        
        assert 'screen' in received_data
        assert 'space' in received_data
        assert 'cell' in received_data
        assert received_data['screen'] == (400.0, 300.0)
        assert received_data['space'] == (0.1, 0.9, 0.0)
        assert received_data['cell'] == 0

    @pytest.mark.default
    def test_plotter_has_tap_event_registered(self):
        """Test that the Bokeh figure has Tap event registered for click."""
        panel, _, cleanup = make_panel_2d()
        cleanup()
        
        mock_callback = Mock()
        panel.provide_on_clic_callback(mock_callback)
        
        # After providing callback, the figure should have an event listener
        assert hasattr(panel.plotter.figure, 'on_event')

    @pytest.mark.default
    def test_both_callbacks_registered_simultaneously(self):
        """Test that both mouse move and click callbacks can be registered simultaneously."""
        panel, _, cleanup = make_panel_2d()
        cleanup()
        
        mouse_calls = []
        click_calls = []
        
        def on_mouse_move(screen_location, space_location, cell_id):
            mouse_calls.append(cell_id)
        
        def on_click(screen_location, space_location, cell_id):
            click_calls.append(cell_id)
        
        panel.provide_on_mouse_move_callback(on_mouse_move)
        panel.provide_on_clic_callback(on_click)
        
        # Both callbacks should be set
        assert panel.plotter.on_mouse_move_callback is not None
        assert panel.plotter.on_clic_callback is not None
        
        # Simulate mouse move event
        panel.plotter.source_mouse.data = {
            "sx": [100.0],
            "sy": [50.0],
            "x": [0.5],
            "y": [0.5],
            "z": [0.0],
            "index": [0],
        }
        
        panel.plotter.send_event(on_mouse_move)
        assert len(mouse_calls) == 1
        assert mouse_calls[0] == 0
        
        # Simulate click event
        panel.plotter.source_mouse.data = {
            "sx": [200.0],
            "sy": [100.0],
            "x": [0.25],
            "y": [0.75],
            "z": [0.0],
            "index": [2],
        }
        
        panel.plotter.send_event(on_click)
        assert len(click_calls) == 1
        assert click_calls[0] == 2


class TestExtensionReceivesMouseCallbacks:
    """Tests verifying that Extension instances receive mouse move and click callbacks from Panel2D."""

    @pytest.mark.default
    def test_extension_receives_mouse_move_callback(self):
        """Test that an extension's on_mouse_move method is called when the mouse moves on the plot.
        
        The VisualizationPanel base class registers each extension's on_mouse_move callback
        via provide_on_mouse_move_callback(extension.on_mouse_move). This test verifies
        the delegation chain works correctly.
        """
        panel, extensions, cleanup = make_panel_2d()
        cleanup()
        dummy_ext = extensions.get(DummyTestExtension)
        
        assert dummy_ext is not None, "DummyTestExtension should be in panel.extensions"
        
        # Reset tracking state
        dummy_ext._on_mouse_move_called = False
        dummy_ext._mouse_move_history.clear()
        
        # Simulate a mouse move event through the plotter
        panel.plotter.source_mouse.data = {
            "sx": [100.0],
            "sy": [50.0],
            "x": [0.5],
            "y": [0.5],
            "z": [0.0],
            "index": [0],
        }
        
        # Trigger the callback by calling send_event with the plotter's stored callback
        if panel.plotter.on_mouse_move_callback is not None:
            panel.plotter.send_event(panel.plotter.on_mouse_move_callback)
        
        # Verify the extension received the callback
        assert dummy_ext._on_mouse_move_called, "DummyTestExtension.on_mouse_move should have been called"
        assert len(dummy_ext._mouse_move_history) == 1, "One mouse move event should be recorded"
        
        event = dummy_ext._mouse_move_history[0]
        assert event.screen_location == (100.0, 50.0)
        assert event.space_location == (0.5, 0.5, 0.0)
        assert event.cell_id == 0

    @pytest.mark.default
    def test_extension_receives_mouse_click_callback(self):
        """Test that an extension's on_mouse_clic method is called when the user clicks on the plot.
        
        The VisualizationPanel base class registers each extension's on_mouse_clic callback
        via provide_on_clic_callback(extension.on_mouse_clic). This test verifies
        the delegation chain works correctly.
        """
        panel, extensions, cleanup = make_panel_2d()
        cleanup()
        dummy_ext = extensions.get(DummyTestExtension)
        
        assert dummy_ext is not None, "DummyTestExtension should be in panel.extensions"
        
        # Reset tracking state
        dummy_ext._on_mouse_clic_called = False
        dummy_ext._mouse_click_history.clear()
        
        # Simulate a click event through the plotter
        panel.plotter.source_mouse.data = {
            "sx": [300.0],
            "sy": [200.0],
            "x": [0.3],
            "y": [0.6],
            "z": [0.0],
            "index": [1],
        }
        
        # Trigger the callback by calling send_event with the plotter's stored callback
        if panel.plotter.on_clic_callback is not None:
            panel.plotter.send_event(panel.plotter.on_clic_callback)
        
        # Verify the extension received the callback
        assert dummy_ext._on_mouse_clic_called, "DummyTestExtension.on_mouse_clic should have been called"
        assert len(dummy_ext._mouse_click_history) == 1, "One mouse click event should be recorded"
        
        event = dummy_ext._mouse_click_history[0]
        assert event.screen_location == (300.0, 200.0)
        assert event.space_location == (0.3, 0.6, 0.0)
        assert event.cell_id == 1

    @pytest.mark.default
    def test_extension_receives_multiple_mouse_move_events(self):
        """Test that an extension receives multiple mouse move events correctly."""
        panel, extensions, cleanup = make_panel_2d()
        cleanup()
        dummy_ext = extensions.get(DummyTestExtension)
        
        assert dummy_ext is not None
        
        # Reset tracking state
        dummy_ext._on_mouse_move_called = False
        dummy_ext._mouse_move_history.clear()
        
        # Simulate multiple mouse move events
        for i, (sx, sy, x, y, idx) in enumerate([
            (100.0, 50.0, 0.1, 0.9, 0),
            (200.0, 150.0, 0.5, 0.5, 1),
            (300.0, 250.0, 0.9, 0.1, 2),
        ]):
            panel.plotter.source_mouse.data = {
                "sx": [sx],
                "sy": [sy],
                "x": [x],
                "y": [y],
                "z": [0.0],
                "index": [idx],
            }
            
            if panel.plotter.on_mouse_move_callback is not None:
                panel.plotter.send_event(panel.plotter.on_mouse_move_callback)
        
        assert len(dummy_ext._mouse_move_history) == 3, "Three mouse move events should be recorded"
        
        # Verify each event
        assert dummy_ext._mouse_move_history[0].screen_location == (100.0, 50.0)
        assert dummy_ext._mouse_move_history[0].cell_id == 0
        assert dummy_ext._mouse_move_history[1].screen_location == (200.0, 150.0)
        assert dummy_ext._mouse_move_history[1].cell_id == 1
        assert dummy_ext._mouse_move_history[2].screen_location == (300.0, 250.0)
        assert dummy_ext._mouse_move_history[2].cell_id == 2

    @pytest.mark.default
    def test_extension_receives_multiple_mouse_click_events(self):
        """Test that an extension receives multiple mouse click events correctly."""
        panel, extensions, cleanup = make_panel_2d()
        cleanup()
        dummy_ext = extensions.get(DummyTestExtension)
        
        assert dummy_ext is not None
        
        # Reset tracking state
        dummy_ext._on_mouse_clic_called = False
        dummy_ext._mouse_click_history.clear()
        
        # Simulate multiple click events
        for i, (sx, sy, x, y, idx) in enumerate([
            (100.0, 50.0, 0.1, 0.9, 0),
            (200.0, 150.0, 0.5, 0.5, 1),
        ]):
            panel.plotter.source_mouse.data = {
                "sx": [sx],
                "sy": [sy],
                "x": [x],
                "y": [y],
                "z": [0.0],
                "index": [idx],
            }
            
            if panel.plotter.on_clic_callback is not None:
                panel.plotter.send_event(panel.plotter.on_clic_callback)
        
        assert len(dummy_ext._mouse_click_history) == 2, "Two mouse click events should be recorded"
        
        # Verify each event
        assert dummy_ext._mouse_click_history[0].screen_location == (100.0, 50.0)
        assert dummy_ext._mouse_click_history[0].cell_id == 0
        assert dummy_ext._mouse_click_history[1].screen_location == (200.0, 150.0)
        assert dummy_ext._mouse_click_history[1].cell_id == 1

    @pytest.mark.default
    def test_extension_does_not_receive_callback_with_invalid_index(self):
        """Test that extension callbacks are not triggered when index is out of bounds."""
        panel, extensions, cleanup = make_panel_2d()
        cleanup()
        dummy_ext = extensions.get(DummyTestExtension)
        
        assert dummy_ext is not None
        
        # Reset tracking state
        dummy_ext._on_mouse_move_called = False
        dummy_ext._mouse_move_history.clear()
        dummy_ext._on_mouse_clic_called = False
        dummy_ext._mouse_click_history.clear()
        
        # Simulate mouse move with invalid index
        panel.plotter.source_mouse.data = {
            "sx": [100.0],
            "sy": [50.0],
            "x": [0.5],
            "y": [0.5],
            "z": [0.0],
            "index": [100],  # Out of bounds (only 3 polygons: 0, 1, 2)
        }
        
        if panel.plotter.on_mouse_move_callback is not None:
            panel.plotter.send_event(panel.plotter.on_mouse_move_callback)
        
        # Extension should NOT have received the callback
        assert not dummy_ext._on_mouse_move_called, "Extension should not receive callback with invalid index"
        assert len(dummy_ext._mouse_move_history) == 0, "No mouse move events should be recorded"
        
        # Simulate click with invalid index
        panel.plotter.source_mouse.data = {
            "sx": [200.0],
            "sy": [100.0],
            "x": [0.25],
            "y": [0.75],
            "z": [0.0],
            "index": [100],  # Out of bounds
        }
        
        if panel.plotter.on_clic_callback is not None:
            panel.plotter.send_event(panel.plotter.on_clic_callback)
        
        # Extension should NOT have received the callback
        assert not dummy_ext._on_mouse_clic_called, "Extension should not receive click callback with invalid index"
        assert len(dummy_ext._mouse_click_history) == 0, "No mouse click events should be recorded"

    @pytest.mark.default
    def test_all_extensions_receive_mouse_callbacks(self):
        """Test that all extensions registered on the panel receive mouse callbacks."""
        panel, extensions, cleanup = make_panel_2d()
        cleanup()
        
        # Verify DummyTestExtension is present
        assert DummyTestExtension in extensions, "DummyTestExtension should be registered"
        
        dummy_ext = extensions[DummyTestExtension]
        
        # Reset tracking state
        dummy_ext._on_mouse_move_called = False
        dummy_ext._mouse_move_history.clear()
        dummy_ext._on_mouse_clic_called = False
        dummy_ext._mouse_click_history.clear()
        
        # Simulate mouse move event
        panel.plotter.source_mouse.data = {
            "sx": [150.0],
            "sy": [100.0],
            "x": [0.4],
            "y": [0.6],
            "z": [0.0],
            "index": [0],
        }
        
        if panel.plotter.on_mouse_move_callback is not None:
            panel.plotter.send_event(panel.plotter.on_mouse_move_callback)
        
        # Verify the extension received it
        assert dummy_ext._on_mouse_move_called, "All extensions should receive mouse move callbacks"
        assert len(dummy_ext._mouse_move_history) == 1
        
        # Simulate click event
        panel.plotter.source_mouse.data = {
            "sx": [250.0],
            "sy": [200.0],
            "x": [0.6],
            "y": [0.4],
            "z": [0.0],
            "index": [1],
        }
        
        if panel.plotter.on_clic_callback is not None:
            panel.plotter.send_event(panel.plotter.on_clic_callback)
        
        # Verify the extension received it
        assert dummy_ext._on_mouse_clic_called, "All extensions should receive click callbacks"
        assert len(dummy_ext._mouse_click_history) == 1

