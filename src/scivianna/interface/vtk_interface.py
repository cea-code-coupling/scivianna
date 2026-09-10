"""
VTK interface for Scivianna.

This module provides a VTK/PyVista-based interface for reading and visualizing
VTK file formats (.pvd, .vtu, .vtp, etc.) in Scivianna. It supports both 2D
slice visualization and 3D geometry display, with time-dependent data handling.

Features
--------
- Read VTK file formats using PyVista
- Time-dependent data visualization
- 2D slice extraction with polygon conversion
- 3D geometry display
- Interactive time slider via VTKExtension

Classes
-------
VTKInterface
    Main interface class for VTK file handling
VTKExtension
    Extension providing time slider GUI for VTK files

Example
-------
>>> from scivianna import ComputeSlave
>>> from scivianna.interface import VTKInterface
>>>
>>> slave = ComputeSlave(VTKInterface)
>>> slave.read_file("mesh.pvd", "Geometry")
>>> panel = Panel2D(slave, name="VTK View")
>>> panel.show()

Dependencies
------------
Requires pyvista and vtk packages. Install with:
    pip install scivianna[3d]
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import panel as pn
import panel_material_ui as pmui

try:
    import pyvista as pv
    import vtk

    _PYVISTA_AVAILABLE = True
except ImportError as e:
    pv = None  # type: ignore[assignment]
    vtk = None  # type: ignore[assignment]
    _PYVISTA_AVAILABLE = False
    raise e

import scivianna
from scivianna.data.data3d import Data3D
from scivianna.extension.extension import Extension
from scivianna.icon import get_icon
from scivianna.interface.generic_interface import Geometry2DPolygon, Geometry3D
from scivianna.logging_config import get_logger

if TYPE_CHECKING:
    import pyvista as pv

    from scivianna.panel.visualisation_panel import VisualizationPanel
    from scivianna.slave import ComputeSlave

from scivianna.constants import (
    CSV,
    GEOMETRY,
    MESH,
)
from scivianna.data.data2d import Data2D
from scivianna.enums import GeometryType, VisualizationMode
from scivianna.plotter_2d.generic_plotter import Plotter2D
from scivianna.plotter_3d.generic_plotter import Plotter3D
from scivianna.utils.polygonize_tools import PolygonCoords, PolygonElement

# Module logger
logger = get_logger(__name__)

icon_svg = get_icon("vtk")


def _require_pyvista() -> None:
    """
    Raise an error if pyvista is not available.

    Raises
    ------
    ImportError
        If pyvista package is not installed
    """
    if not _PYVISTA_AVAILABLE:
        raise ImportError(
            "PyVista could not be imported. Please install with: "
            "pip install scivianna[3d] or pip install pyvista"
        )


def extract_unstructured_grid(dataset: Any) -> Optional[pv.UnstructuredGrid]:
    """
    Extract an UnstructuredGrid from a pyvista dataset.

    This function recursively searches through MultiBlock datasets to find
    and extract unstructured grid data, which is required for VTK visualization.

    Parameters
    ----------
    dataset : Any
        PyVista dataset (UnstructuredGrid, MultiBlock, or other)

    Returns
    -------
    Optional[pv.UnstructuredGrid]
        UnstructuredGrid if found, None otherwise

    Example
    -------
    >>> multiblock = pv.read("complex_mesh.vtm")
    >>> unstructured = extract_unstructured_grid(multiblock)
    >>> if unstructured is not None:
    ...     print(f"Found {unstructured.number_of_cells} cells")
    """
    _require_pyvista()

    if isinstance(dataset, pv.UnstructuredGrid):
        return dataset

    elif isinstance(dataset, pv.MultiBlock):
        for block in dataset:
            if block is not None:
                result = extract_unstructured_grid(block)
                if result is not None:
                    return result
        return None

    else:
        # Try to cast to unstructured grid
        try:
            return dataset.cast_to_unstructured_grid()
        except Exception as e:
            logger.debug("Could not cast dataset to UnstructuredGrid: %s", e)
            return None


class VTKExtension(Extension):
    """
    Extension for VTK file visualization with time slider control.

    This extension provides a GUI with a time slider for navigating through
    time-dependent VTK data. It allows users to select specific time steps
    and optionally recompute the mesh when time changes.

    Attributes
    ----------
    time_slider : pmui.DiscreteSlider
        Slider widget for time step selection
    recompute_on_change : pmui.Checkbox
        Checkbox to enable/disable mesh recomputation on time change
    time_values : List[float]
        Available time steps from the VTK file

    Example
    -------
    >>> # Extension is automatically added to VTKInterface
    >>> # Users can interact via the GUI time slider
    >>> extension = VTKExtension(slave, plotter, panel)
    >>> extension.time_slider.value = 5.0  # Set time to 5.0
    """

    def __init__(
        self,
        slave: "ComputeSlave",
        plotter: Union[Plotter2D, "Plotter3D"],
        panel: "VisualizationPanel",
    ):
        """
        Initialize the VTK extension.

        Parameters
        ----------
        slave : ComputeSlave
            Compute slave for data queries
        plotter : Plotter2D or Plotter3D
            Plotter for visualization
        panel : VisualizationPanel
            Parent visualization panel

        Raises
        ------
        ImportError
            If pyvista is not available
        AssertionError
            If plotter is not Plotter2D or Plotter3D
        """
        _require_pyvista()

        super().__init__(
            title="VTK",
            icon=icon_svg,
            slave=slave,
            plotter=plotter,
            panel=panel,
        )

        # Get time values from interface
        self.time_values: List[float] = self.slave.call_custom_function("get_time_values", {})
        if not self.time_values:
            self.time_values = [0.0]

        # Create GUI components
        self.time_slider = pmui.DiscreteSlider(
            label="Time",
            options=self.time_values,
            width=260,
        )
        self.recompute_on_change = pmui.Checkbox(
            label="Recompute mesh on time change",
            value=False,
            width=260,
        )

        # Validate plotter type
        assert isinstance(
            plotter, (Plotter2D, Plotter3D)
        ), f"VTKExtension requires Plotter2D or Plotter3D, found {type(plotter)}"

        self.description = """
        This extension allows defining the VTK plot parameters including
        time step selection for time-dependent simulations.
        """

        self.iconsize = "1.0em"

        # Register callback for time slider
        self.time_slider.param.watch(self._on_time_change, "value_throttled")

    def _on_time_change(self, event: Any) -> None:
        """
        Callback triggered when time slider value changes.

        Parameters
        ----------
        event : Any
            Panel event object containing old/new values
        """
        logger.debug("Time changed to %s", event.new)
        self.panel.recompute()

    def on_file_load(self, file_path: str, file_key: str) -> None:
        """
        Handle file load events to update time values.

        Parameters
        ----------
        file_path : str
            Path to the loaded file
        file_key : str
            File key/label (e.g., "Geometry")
        """
        if file_key == GEOMETRY:
            logger.info("Updating time values for file: %s", file_path)
            self.time_values = self.slave.call_custom_function("get_time_values", {})
            if not self.time_values:
                self.time_values = [0.0]
            self.time_slider.options = self.time_values

    def provide_options(self) -> Dict[str, Any]:
        """
        Provide computation options to the interface.

        Returns
        -------
        Dict[str, Any]
            Dictionary with time and recompute options
        """
        return {
            "time": self.time_slider.values[self.time_slider.value_throttled],
            "recompute": self.recompute_on_change.value,
        }

    def make_gui(self) -> pn.viewable.Viewable:
        """
        Build the extension GUI panel.

        Returns
        -------
        pn.viewable.Viewable
            Panel layout with time slider and options
        """
        return pmui.Column(
            self.time_slider,
            self.recompute_on_change,
            margin=0,
        )


class VTKInterface(Geometry2DPolygon, Geometry3D):
    """
    VTK file interface for Scivianna visualization.

    This interface reads VTK file formats (particularly .pvd files) using PyVista
    and provides 2D slice and 3D geometry visualization capabilities. It supports
    time-dependent data with interactive time step selection.

    Features
    --------
    - Read .pvd, .vtu, .vtp, and other VTK formats
    - Time-dependent data visualization
    - 2D slice extraction with polygon conversion
    - 3D unstructured grid display
    - Automatic point-to-cell data conversion

    Attributes
    ----------
    reader : pv.reader
        PyVista reader object for the loaded file
    mesh : pv.UnstructuredGrid
        Current mesh data
    times : List[float]
        Available time steps
    current_time : float
        Currently selected time step
    data : Dict[str, Data2D]
        Cached 2D data per caller
    last_computed_frame : Dict[str, List[float]]
        Cache keys for computed frames per caller

    Example
    -------
    >>> from scivianna import ComputeSlave
    >>> from scivianna.interface import VTKInterface
    >>>
    >>> slave = ComputeSlave(VTKInterface)
    >>> slave.read_file("simulation.pvd", "Geometry")
    >>> labels = slave.get_labels()
    >>> print(f"Available fields: {labels}")

    See Also
    --------
    VTKExtension : Extension providing time slider GUI
    """

    extensions = [VTKExtension]
    geometry_type = GeometryType._3D_INFINITE

    def __init__(self) -> None:
        """
        Initialize the VTK interface.

        Sets up data structures for mesh storage, time tracking, and caching.
        """
        _require_pyvista()

        self.data: Dict[str, Data2D] = {}
        """Cache of computed 2D data per caller."""

        self.reader: Optional[Any] = None
        """PyVista reader object."""

        self.mesh: Optional[pv.UnstructuredGrid] = None
        """Current mesh data."""

        self.times: List[float] = []
        """Available time steps from the file."""

        self.results: Dict[str, Any] = {}
        """Additional result data (e.g., from CSV files)."""

        self.last_computed_frame: Dict[str, List[float]] = {}
        """Cache keys for last computed frame per caller."""

        self.current_time: float = 0.0
        """Currently selected time step."""

        self.last_3d_frame: Optional[Dict[str, Any]] = None
        """Cache key for last 3D computation."""

        self.file_infos: List[Tuple[str, str]] = []
        """List of (file_path, file_label) tuples for files loaded by this interface."""

        logger.debug("VTKInterface initialized")

    def read_file(self, file_path: Union[str, Path], file_label: str) -> None:
        """
        Read a VTK file and store its content.

        Supports .pvd files for time-dependent data and multi-block datasets.
        Automatically converts point data to cell data for visualization.

        Parameters
        ----------
        file_path : str or Path
            Path to the VTK file (.pvd, .vtu, .vtp, etc.)
        file_label : str
            File type label ("Geometry" or "MULTI_BLOCK")

        Raises
        ------
        ImportError
            If pyvista is not available
        NotImplementedError
            If file_label is not supported
        FileNotFoundError
            If file does not exist

        Example
        -------
        >>> interface = VTKInterface()
        >>> interface.read_file("mesh.pvd", "Geometry")
        >>> print(f"Loaded {len(interface.times)} time steps")
        """
        _require_pyvista()

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"VTK file not found: {file_path}")

        logger.info("Reading VTK file: %s as %s", file_path, file_label)

        if file_label == GEOMETRY:
            try:
                self.reader = pv.get_reader(str(file_path))
                if hasattr(self.reader, "time_values"):
                    self.times = self.reader.time_values
                else:
                    logger.info(
                        "Reader %s does not support time series; treating file as static",
                        type(self.reader).__name__,
                    )
                    self.times = [-1.]
                logger.info("Loaded %d time steps from %s", len(self.times), file_path)

                # Store file info for rebuild on load
                self.file_infos.append((str(file_path), file_label))

                # Load last time step by default
                if self.times:
                    self.load_at_time(self.times[-1])
                else:
                    logger.warning("No time values found in file")
                    self.times = [0.0]
                    self.current_time = 0.0

            except Exception as e:
                logger.error("Failed to read VTK file %s: %s", file_path, e)
                raise

        elif file_label == "MULTI_BLOCK":
            try:
                self._read_multi_block(file_path)
                # Store file info for rebuild on load
                self.file_infos.append((str(file_path), file_label))
            except Exception as e:
                logger.error("Failed to read multi-block file %s: %s", file_path, e)
                raise
        else:
            raise NotImplementedError(
                f"File label '{file_label}' not supported by VTKInterface. "
                f"Use '{GEOMETRY}' or 'MULTI_BLOCK'."
            )

    def _read_multi_block(self, file_path: Path) -> None:
        """
        Read a multi-block VTK dataset.

        Parameters
        ----------
        file_path : Path
            Path to the multi-block file
        """
        logger.debug("Reading multi-block file: %s", file_path)

        reader = vtk.vtkXMLMultiBlockDataReader()
        reader.SetFileName(str(file_path))
        reader.Update()

        data = reader.GetOutput()

        # Append all unstructured grids
        append_filter = vtk.vtkAppendFilter()

        def add_blocks(block: Any) -> None:
            """Recursively add blocks to append filter."""
            if block is None:
                return

            if block.IsA("vtkUnstructuredGrid"):
                append_filter.AddInputData(block)
            elif block.IsA("vtkMultiBlockDataSet"):
                for i in range(block.GetNumberOfBlocks()):
                    add_blocks(block.GetBlock(i))
            elif block.IsA("vtkMultiPieceDataSet"):
                for i in range(block.GetNumberOfPieces()):
                    add_blocks(block.GetPiece(i))

        add_blocks(data)
        append_filter.Update()

        # Convert to pyvista and add cell IDs
        self.mesh = pv.wrap(append_filter.GetOutput())
        self.mesh = self.mesh.point_data_to_cell_data()
        self.mesh.cell_data["cell_id"] = list(range(self.mesh.number_of_cells))

        self.times = [0.0]
        self.current_time = 0.0

        logger.info("Loaded multi-block dataset with %d cells", self.mesh.number_of_cells)

    def load_at_time(self, time: float) -> None:
        """
        Load data at a specific time step.

        Parameters
        ----------
        time : float
            Time value to load

        Raises
        ------
        ValueError
            If time is not in available time steps

        Example
        -------
        >>> interface.load_at_time(5.0)  # Load data at t=5.0
        """
        _require_pyvista()

        if hasattr(self.reader, "time_values"):
            if time not in self.times:
                raise ValueError(f"Time {time} not available. Available times: {self.times}")

            logger.debug("Loading data at time %s", time)

            self.reader.set_active_time_value(time)
            dataset = self.reader.read()[0]
        else:
            logger.debug("No time present in file")
            dataset = self.reader.read()

        self.mesh = extract_unstructured_grid(dataset)

        if self.mesh is None:
            raise ValueError("Could not extract unstructured grid from dataset")

        # Convert point data to cell data for visualization
        self.mesh = self.mesh.point_data_to_cell_data()
        self.mesh.cell_data["cell_id"] = list(range(self.mesh.number_of_cells))
        self.current_time = time

        logger.debug("Loaded mesh at time %s with %d cells", time, self.mesh.number_of_cells)

    def compute_2D_data(
        self,
        u: Tuple[float, float, float],
        v: Tuple[float, float, float],
        origin: Tuple[float, float, float],
        size_u: float,
        size_v: float,
        q_tasks: mp.Queue,
        options: Dict[str, Any],
        caller: str = "API",
    ) -> Tuple[Data2D, bool]:
        """
        Compute a 2D slice of the VTK geometry.

        Extracts a 2D cross-section from the 3D mesh using the specified
        plane (defined by u, v axes and origin). Converts the slice to
        polygons for visualization.

        Parameters
        ----------
        u : Tuple[float, float, float]
            Horizontal axis direction vector
        v : Tuple[float, float, float]
            Vertical axis direction vector
        origin : Tuple[float, float, float]
            Physical 3D position of slice center
        size_u : float
            Size of slice along u axis
        size_v : float
            Size of slice along v axis
        q_tasks : mp.Queue
            Task queue for multiprocessing
        options : Dict[str, Any]
            Additional options (time, recompute)
        caller : str
            Identifier of the caller

        Returns
        -------
        Tuple[Data2D, bool]
            Data2D object with polygon geometry and whether polygons were updated

        Raises
        ------
        ValueError
            If u and v vectors are parallel or zero
            If slice produces no cells
        """
        _require_pyvista()

        # Set default options
        if "recompute" not in options:
            options["recompute"] = True
        if "time" not in options:
            options["time"] = self.current_time

        # Load data at requested time
        time_updated = False
        if self.current_time != options["time"]:
            logger.debug("Time changed from %s to %s", self.current_time, options["time"])
            self.load_at_time(options["time"])
            time_updated = True

        # Check cache
        cache_key = [*u, *v, *origin, size_u, size_v]
        if (
            caller in self.last_computed_frame
            and self.last_computed_frame[caller] == cache_key
            and caller in self.data
            and not (options["recompute"] and time_updated)
        ):
            logger.debug("Using cached 2D data for caller %s", caller)
            return self.data[caller], False

        # Normalize axes
        u_arr = np.array(u, dtype=float)
        v_arr = np.array(v, dtype=float)
        u_arr /= np.linalg.norm(u_arr)
        v_arr /= np.linalg.norm(v_arr)

        # Compute normal vector
        w_arr = np.cross(u_arr, v_arr)
        w_norm = np.linalg.norm(w_arr)

        if w_norm == 0.0:
            raise ValueError(
                f"Vectors u={u} and v={v} are parallel or zero. "
                "Cannot compute cross product for slice normal."
            )

        w_arr /= w_norm

        # Ensure origin is numpy array
        origin_arr = np.array(origin, dtype=float)

        logger.debug("Computing 2D slice at origin %s with normal %s", origin_arr, w_arr)

        # Extract slice
        try:
            mesh_slice: pv.PolyData = self.mesh.slice(
                normal=w_arr, origin=origin_arr, generate_triangles=True
            )
        except Exception as e:
            logger.error("Failed to slice mesh: %s", e)
            raise

        if mesh_slice.GetNumberOfCells() == 0:
            bounds = self.mesh.bounds
            raise ValueError(
                f"Slice at origin {origin_arr} with normal {w_arr} produced no cells. "
                f"Mesh bounds: {bounds}"
            )

        # Convert to polygons
        polygon_elements = self._mesh_slice_to_polygons(mesh_slice, u_arr, v_arr)

        # Create Data2D object
        self.data[caller] = Data2D.from_polygon_list(polygon_elements)
        self.last_computed_frame[caller] = cache_key

        logger.debug(
            "Computed 2D slice with %d polygons for caller %s", len(polygon_elements), caller
        )

        return self.data[caller], True

    def _mesh_slice_to_polygons(
        self, mesh_slice: pv.PolyData, u_arr: np.ndarray, v_arr: np.ndarray
    ) -> List[PolygonElement]:
        """
        Convert a mesh slice to PolygonElement list.

        Parameters
        ----------
        mesh_slice : pv.PolyData
            Sliced mesh data
        u_arr : np.ndarray
            Horizontal axis unit vector
        v_arr : np.ndarray
            Vertical axis unit vector

        Returns
        -------
        List[PolygonElement]
            List of polygons for visualization
        """
        polygon_elements: List[PolygonElement] = []

        cell_ids = mesh_slice.cell_data["cell_id"]

        for i in range(mesh_slice.GetNumberOfCells()):
            cell = mesh_slice.GetCell(i)
            point_ids = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]

            # Project points to 2D coordinates
            points_3d = np.array([mesh_slice.points[pid] for pid in point_ids])
            x_coords = points_3d @ u_arr
            y_coords = points_3d @ v_arr

            polygon_elements.append(
                PolygonElement(
                    exterior_polygon=PolygonCoords(
                        x_coords=x_coords.tolist(), y_coords=y_coords.tolist()
                    ),
                    holes=[],
                    cell_id=cell_ids[i] if i < len(cell_ids) else i,
                )
            )

        return polygon_elements

    def get_value_dict(
        self,
        value_label: str,
        cells: List[Union[int, str]],
        options: Dict[str, Any],
        caller: str = "API",
    ) -> Dict[Union[int, str], Any]:
        """
        Get field values for specified cells.

        Parameters
        ----------
        value_label : str
            Field name to retrieve
        cells : List[Union[int, str]]
            List of cell identifiers
        options : Dict[str, Any]
            Additional options (time, recompute)
        caller : str
            Caller identifier

        Returns
        -------
        Dict[Union[int, str], Any]
            Mapping of cell IDs to values

        Raises
        ------
        NotImplementedError
            If field is not available

        Example
        -------
        >>> values = interface.get_value_dict("Temperature", [0, 1, 2], {})
        >>> print(f"Cell 0 temperature: {values[0]}")
        """
        # Set default options
        if "recompute" not in options:
            options["recompute"] = True
        if "time" not in options:
            options["time"] = self.current_time

        # Load data at requested time
        if self.current_time != options["time"]:
            logger.debug("Loading data at time %s for field query", options["time"])
            self.load_at_time(options["time"])

        if self.mesh is None:
            raise ValueError("No mesh loaded. Call read_file first.")

        # Handle mesh-only visualization
        if value_label == MESH:
            return {cell: np.nan for cell in cells}

        # Try to get from mesh cell data
        if value_label in self.mesh.array_names:
            try:
                data = self.mesh.cell_data[value_label]
                # Handle both integer and string cell IDs
                if all(isinstance(c, (int, np.integer)) for c in cells):
                    return {cell: data[int(cell)] for cell in cells}
                else:
                    # Map cell IDs to indices
                    return {cell: data[int(cell)] for cell in cells if int(cell) < len(data)}
            except (KeyError, IndexError) as e:
                logger.warning("Failed to get cell data for %s: %s", value_label, e)

        # Try results from CSV or other sources
        for result in self.results.values():
            if hasattr(result, "get_labels") and value_label in result.get_labels():
                if hasattr(result, "get_values"):
                    values = result.get_values([], cells, [], value_label)
                    return {cells[i]: values[i] for i in range(len(cells))}

        # Field not found
        available_fields = self.get_labels()
        raise NotImplementedError(
            f"Field '{value_label}' not found. Available fields: {available_fields}"
        )

    def get_labels(self) -> List[str]:
        """
        Get list of displayable field names.

        Returns
        -------
        List[str]
            List of available field names

        Example
        -------
        >>> labels = interface.get_labels()
        >>> print(f"Available fields: {labels}")
        """
        labels = [MESH]

        if self.mesh is not None:
            # Add mesh array names, excluding TIME
            labels.extend([name for name in self.mesh.array_names if name != "TIME"])

        # Add fields from results
        for result in self.results.values():
            if hasattr(result, "get_labels"):
                labels.extend(result.get_labels())

        logger.debug("Available fields: %s", labels)
        return labels

    def get_label_coloring_mode(self, label: str) -> VisualizationMode:
        """
        Get coloring mode for a field.

        Parameters
        ----------
        label : str
            Field name

        Returns
        -------
        VisualizationMode
            Coloring mode (FROM_VALUE, FROM_STRING, or NONE)
        """
        if label == MESH:
            return VisualizationMode.NONE
        else:
            return VisualizationMode.FROM_VALUE

    def get_file_input_list(self) -> List[Tuple[str, str]]:
        """
        Get list of supported file types.

        Returns
        -------
        List[Tuple[str, str]]
            List of (file_label, description) tuples
        """
        return [
            (GEOMETRY, "VTK file (.pvd, .vtu, .vtp, etc.)"),
            (CSV, "CSV result file for additional fields"),
        ]

    def get_time_values(self) -> List[float]:
        """
        Get available time values from the loaded file.

        Returns
        -------
        List[float]
            List of time values

        Example
        -------
        >>> times = interface.get_time_values()
        >>> print(f"Time range: {min(times)} to {max(times)}")
        """
        return self.times.copy()

    def compute_3D_data(self, options: Dict[str, Any]) -> Tuple[Data3D, bool]:
        """
        Compute 3D geometry data.

        Parameters
        ----------
        options : Dict[str, Any]
            Additional options (time, recompute)

        Returns
        -------
        Tuple[Data3D, bool]
            Data3D object and whether geometry was updated

        Raises
        ------
        ImportError
            If pyvista is not available
        ValueError
            If no mesh is loaded
        """
        _require_pyvista()

        if self.mesh is None:
            raise ValueError("No mesh loaded. Call read_file first.")

        # Set default options
        if "recompute" not in options:
            options["recompute"] = True
        if "time" not in options:
            options["time"] = self.current_time

        # Load data at requested time
        if self.current_time != options["time"]:
            logger.debug("Loading 3D data at time %s", options["time"])
            self.load_at_time(options["time"])

        # Check cache
        if self.last_3d_frame == options:
            logger.debug("Using cached 3D data")
            try:
                return Data3D.from_vtk(self.mesh), False
            except (ImportError, AttributeError):
                logger.warning("Data3D.from_vtk not available, returning None")
                return None, False

        # Update cache
        self.last_3d_frame = options.copy()

        try:
            data_3d = Data3D.from_vtk(self.mesh)
            logger.debug("Computed 3D data with %d cells", len(data_3d.cell_ids))
            return data_3d, True
        except (ImportError, AttributeError) as e:
            logger.error("Failed to create Data3D from VTK: %s", e)
            raise

    def get_3d_value_dict(
        self,
        value_label: str,
        cells: List[Union[int, str]],
        options: Dict[str, Any],
        caller: str = "API",
    ) -> Dict[Union[int, str], Any]:
        """
        Get field values for 3D cells.

        Parameters
        ----------
        value_label : str
            Field name
        cells : List[Union[int, str]]
            Cell identifiers
        options : Dict[str, Any]
            Additional options
        caller : str
            Caller identifier

        Returns
        -------
        Dict[Union[int, str], Any]
            Cell ID to value mapping

        Note
        ----
        This is a wrapper around get_value_dict for 3D visualization.
        """
        return self.get_value_dict(value_label, cells, options, caller)

    def _rebuild_reader(self) -> None:
        """
        Re-read all files from saved paths to restore the reader and mesh.

        This is called after loading a pickled state to reconstruct the PyVista
        readers so that time-based data can be reloaded via load_at_time().
        """
        if not self.file_infos:
            logger.warning("No file information available to rebuild reader")
            return

        # Store a copy of file_infos before clearing (read_file appends to file_infos)
        files_to_read = list(self.file_infos)

        # Clear current state before re-reading files
        self.reader = None
        self.mesh = None
        self.times = []
        self.current_time = 0.0
        self.file_infos.clear()

        for file_path, file_label in files_to_read:
            try:
                logger.info("Re-reading saved file: %s as %s", file_path, file_label)
                self.read_file(file_path, file_label)
            except Exception as e:
                logger.error("Failed to re-read saved file %s: %s", file_path, e)

        # Restore current_time if possible
        if self.times and self.current_time not in self.times:
            if 0.0 in self.times:
                self.current_time = 0.0
            elif self.times:
                self.current_time = self.times[-1]

    def save(self, file_path: Path, include_files: bool):
        """Pickle saves the slave content to a file, allows slave state reload.

        Two modes are available:
            -   If **include_files** is at True, all loaded data are saved, the pickled file can be loaded on its own to recover last session.
            -   If **include_files** is at False, only the computed data are loaded, enabling faster first computation allowing a smaller pickle file size.

        Parameters
        ----------
        file_path : Path
            File to which save the slave
        include_files : bool
            Included loaded file
        """
        _require_pyvista()

        with open(file_path, "wb") as f:
            data = (
                scivianna.__version__,
                pv.__version__ if pv is not None else "unknown",
                sys.version,
                include_files,
                "VTKInterface",
            )

            if include_files:
                full_data = (
                    self.times,
                    self.current_time,
                    self.file_infos,
                    self.last_computed_frame,
                    self.data,
                    self.last_3d_frame,
                    self.results,
                    self.mesh,
                )
                pickle.dump((*data, *full_data), f)
            else:
                # Even in minimal mode, save file_infos so we can rebuild the reader
                minimal_data = (
                    self.file_infos,
                    self.last_computed_frame,
                    self.data,
                    self.last_3d_frame,
                    self.results,
                    self.mesh,
                )
                pickle.dump((*data, *minimal_data), f)

    def load(self, file_path: Path, include_files: bool):
        """Pickle loads the slave content to a file, allows slave state reload.

        Two modes are available:
            -   If **include_files** is at True, all loaded data are saved, the pickled file can be loaded on its own to recover last session.
            -   If **include_files** is at False, only the computed data are loaded, enabling faster first computation allowing a smaller pickle file size.

        Parameters
        ----------
        file_path : Path
            File from which load the slave
        include_files : bool
            Included loaded file
        """
        if not Path(file_path).is_file():
            raise ValueError(f"Provided path {file_path} does not exist")

        _require_pyvista()

        with open(file_path, "rb") as f:
            data = pickle.load(f)

            assert len(data) > 5, "Loaded data is not meant for VTKInterface"
            version, pv_version, python_version, inc_files, interface_name = data[:5]
            if version != scivianna.__version__:
                logger.warning(
                    f"Loading file built with scivianna {version}, current version: {scivianna.__version__}."
                )
            if pv is not None and pv_version != pv.__version__:
                logger.warning(
                    f"Loading file built with pyvista {pv_version}, current version: {pv.__version__}."
                )
            if python_version != sys.version:
                logger.warning(
                    f"Loading file built with Python {python_version}, current version: {sys.version}."
                )

            assert (
                inc_files == include_files
            ), f"Loaded file has include_files at {inc_files}, currently calling with include_files at {include_files}."

            assert (
                interface_name == "VTKInterface"
            ), f"Loaded file is built by interface {interface_name}, trying to load with VTKInterface."

            if include_files:
                (
                    self.times,
                    self.current_time,
                    self.file_infos,
                    self.last_computed_frame,
                    self.data,
                    self.last_3d_frame,
                    self.results,
                    self.mesh,
                ) = data[5:]

            else:
                (
                    self.file_infos,
                    self.last_computed_frame,
                    self.data,
                    self.last_3d_frame,
                    self.results,
                    self.mesh,
                ) = data[5:]

            self._rebuild_reader()

    def custom_function(self, function_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a custom function by name.

        This method allows extensions to call interface-specific functions
        that are not part of the standard interface API.

        Parameters
        ----------
        function_name : str
            Name of the function to call
        arguments : Dict[str, Any]
            Function arguments

        Returns
        -------
        Any
            Function result

        Raises
        ------
        AttributeError
            If function does not exist
        """
        if not hasattr(self, function_name):
            raise AttributeError(f"Function '{function_name}' not found in VTKInterface")

        func = getattr(self, function_name)
        return func(**arguments)
