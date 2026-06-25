from enum import Enum, auto
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Type, Tuple

from scivianna.coupling.problem_server import ServerManager, ProblemClient
from scivianna.enums import UpdatePolicy
from scivianna.layout.gridstack import GridStackLayout
from scivianna.layout.split import SplitDirection, SplitItem, SplitLayout
from scivianna.panel.panel_1d import Panel1D
from scivianna.panel.panel_2d import Panel2D
from scivianna.interface.generic_interface import GenericInterface, Geometry2D, Value1DAtLocation, ValueAtLocation
from scivianna.interface.time_dataframe import TimeDataFrame
from scivianna.interface import INTERFACES
from scivianna.panel.visualisation_panel import VisualizationPanel
from scivianna.slave import ComputeSlave
from scivianna.notebook_tools import get_med_panel

from scivianna.coupling.icoco import LayoutProblem

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    PositiveFloat,
    model_validator,
    field_validator
)

import numpy as np
import medcoupling as mc


class CouplingPanel(BaseModel):
    """
    Base model for coupling visualization panels.

    This class defines the common properties shared by all coupling panel types,
    including name, update policy, interface configuration, and field templates.

    Attributes
    ----------
    name : str
        Plot panel name.
    update_policy : UpdatePolicy
        Code interface management definition.
    interface : Type[GenericInterface]
        Code interface type.

    Example
    -------
    >>> panel = CouplingPanel(
    ...     name="temperature",
    ...     update_policy=UpdatePolicy.UPDATE_DATA,
    ...     interface=MyInterface
    ... )
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    update_policy: UpdatePolicy
    interface: Type[GenericInterface]
    template: List[Tuple[str, any]] | None = None

    @field_validator("interface", mode="before")
    @classmethod
    def resolve_interface(cls, v):
        """
        Validate and resolve the interface attribute.

        Converts string interface identifiers to their corresponding
        interface classes using the global INTERFACES registry.

        Parameters
        ----------
        v : any
            The interface value to validate (type or string).

        Returns
        -------
        Type[GenericInterface]
            The resolved interface class.

        Raises
        ------
        ValueError
            Raised if the string identifier is not found in scivianna INTERFACES.
        TypeError
            Raised if the value is neither a type nor a string.
        """
        if isinstance(v, type):
            return v
        if isinstance(v, str):
            resolved = INTERFACES.get(v)
            if resolved is None:
                raise ValueError(f"Unknown interface identifier: {v}")
            return resolved
        raise TypeError(f"interface must be str or type, got {type(v).__name__}")

class FieldPanel(CouplingPanel):
    """
    Panel configuration for 2D field visualization.

    Extends CouplingPanel with 2D geometry-specific properties including
    coordinate system vectors, axis bounds, and display settings.

    Attributes
    ----------
    interface : Type[Geometry2D]
        2D Geometry code interface type.
    u : tuple[float, float, float] | None
        Horizontal axis vector.
    v : tuple[float, float, float] | None
        Vertical axis vector.
    u_min : float | None
        Horizontal axis lower bound.
    u_max : float | None
        Horizontal axis upper bound.
    v_min : float | None
        Vertical axis lower bound.
    v_max : float | None
        Vertical axis upper bound.
    w : float | None
        Displayed frame normal vector coordinate (center goes to
        u_min * u + v_min * v + w * u^v).
    color_map : str | None
        Plot color map name.
    displayed_fields : str
        Displayed field when the GUI is opened.

    Example
    -------
    >>> panel = FieldPanel(
    ...     name="temperature_field",
    ...     interface=Geometry2D,
    ...     u=(1.0, 0.0, 0.0),
    ...     v=(0.0, 1.0, 0.0),
    ...     color_map="viridis",
    ...     displayed_field="power"
    ... )
    """
    interface: Type[Geometry2D]
    """2D Geometry code interface"""

    u: tuple[float, float, float] | None = None
    """Horizontal axis vector"""
    v: tuple[float, float, float] | None = None
    """Vertical axis vector"""

    u_min: float | None = None
    """Horizontal axis lower bound"""
    u_max: float | None = None
    """Horizontal axis upper bound"""

    v_min: float | None = None
    """Vertical axis lower bound"""
    v_max: float | None = None
    """Vertical axis upper bound"""

    w: float | None = None
    """Displayed frame normal vector coordinate (center goes to u_min * u + v_min * v + w * u^v)"""

    color_map: str | None = None
    """Plot color map"""

    displayed_field: str
    """Displayed field on GUI opening"""

class ValuePanel(CouplingPanel):
    """
    Panel configuration for 1D value visualization.

    Extends CouplingPanel with 1D plot-specific properties including
    axis bounds and data interface type.

    Attributes
    ----------
    min_time : Optional[NonNegativeFloat]
        1D plot minimum horizontal axis value.
    max_time : Optional[PositiveFloat]
        1D plot maximum horizontal axis value.
    min_value : Optional[NonNegativeFloat]
        1D plot minimum vertical axis value.
    max_value : Optional[PositiveFloat]
        1D plot maximum vertical axis value.
    interface : Union[Type[Value1DAtLocation], Type[ValueAtLocation]]
        1D data code interface type.
    displayed_fields : List[str]
        List of displayed fields when the GUI is opened.

    Example
    -------
    >>> panel = ValuePanel(
    ...     name="pressure_at_point",
    ...     interface=ValueAtLocation,
    ...     min_time=0.0,
    ...     max_time=10.0,
    ...     displayed_fields=["min", "max"]
    ... )
    """
    min_time: Optional[NonNegativeFloat] = None
    """1D plot minimum horizontal axis value"""
    max_time: Optional[PositiveFloat] = None
    """1D plot maximum horizontal axis value"""

    min_value: Optional[NonNegativeFloat] = None
    """1D plot minimum vertical axis value"""
    max_value: Optional[PositiveFloat] = None
    """1D plot maximum vertical axis value"""

    interface: Union[Type[Value1DAtLocation], Type[ValueAtLocation]]
    """1D data code interface"""
    displayed_fields: List[str]
    """Displayed fields on GUI opening"""

class GridLayoutData(BaseModel):
    """
    Grid layout configuration for visualization panels.

    Defines a 2D grid structure where each cell contains either a FieldPanel,
    ValuePanel, or None (for empty cells). Includes automatic validation to
    ensure panel names are unique.

    Attributes
    ----------
    grid : List[List[Optional[Union[FieldPanel, ValuePanel]]]]
        2D grid of visualization panels.
    title : str
        Title for the layout.

    Example
    -------
    >>> layout = GridLayoutData(
    ...     title="Temperature Distribution",
    ...     grid=[
    ...         [field_panel_1, field_panel_2],
    ...         [None, value_panel]
    ...     ]
    ... )
    """
    grid: List[List[Optional[Union[FieldPanel, ValuePanel]]]]
    title: str

    @model_validator(mode="after")
    def check(self):

        found_names = set()

        for line in self.grid:
            for element in line:
                if element is None:
                    continue
                if element.name in found_names:
                    raise ValueError(f"name {element.name} appears at least twice.")

                found_names.add(element.name)

        return self

class SplitLayoutData(BaseModel):
    """
    Recursive split layout configuration for visualization panels.

    Defines a hierarchical tree structure where panels can be split vertically
    or horizontally. Supports nested layouts for complex visual arrangements.
    Includes automatic validation to ensure panel names are unique.

    Attributes
    ----------
    split : List[Union[FieldPanel, ValuePanel, "SplitLayoutData"]]
        List of child elements (panels or nested split layouts).
    vertical_cut : bool
        Direction of the cut between panels: True for vertical separator,
        False for horizontal separator.
    name : str
        Name identifier for the layout section.

    Example
    -------
    >>> layout = SplitLayoutData(
    ...     name="analysis_view",
    ...     vertical_cut=True,
    ...     split=[field_panel, value_panel]
    ... )
    """
    split: List[Union[FieldPanel, ValuePanel, "SplitLayoutData"]]
    vertical_cut: bool

    name: str

    @model_validator(mode="after")
    def check(self):
        found_names = set()

        for element in self.get_panels():
            if element.name in found_names:
                raise ValueError(f"name {element.name} appears at least twice.")

            found_names.add(element.name)

        return self

    def get_panels(self,) -> List[Union[FieldPanel, ValuePanel]]:
        """
        Get all FieldPanel and ValuePanel objects in this layout and children.

        Recursively traverses the split layout tree to collect all leaf
        panel objects (FieldPanel and ValuePanel).

        Returns
        -------
        List[Union[FieldPanel, ValuePanel]]
            List of all actual panels contained in the layout hierarchy.

        Raises
        ------
        TypeError
            Raised if one of the contained objects is not implemented
            (i.e., not a FieldPanel, ValuePanel, or SplitLayoutData).
        """
        panels = []
        for e in self.split:
            if isinstance(e, (FieldPanel, ValuePanel)):
                panels.append(e)
            elif isinstance(e, SplitLayoutData):
                panels += e.get_panels()
            else:
                raise TypeError(f"SplitLayoutData only accepts FieldPanel, ValuePanel, SplitLayoutData objects, found {type(e)}")
        return panels

    def get_serializable(self,) -> "SplitLayoutData":
        """
        Get a serializable version of this layout.

        Replaces interface type objects with their string keys so the
        layout can be serialized to JSON. Recursively processes all nested
        SplitLayoutData objects.

        Returns
        -------
        SplitLayoutData
            A new SplitLayoutData instance with serializable interface references.

        Raises
        ------
        TypeError
            Raised if self contains a not implemented type.
        """
        splits = []
        for element in self.split:
            if isinstance(element, (FieldPanel, ValuePanel)):
                slave = element.interface.get_slave()

                assert element.interface in INTERFACES.values(), f"Interface {element.interface} is not registered in scivianna interfaces. Please first call scivianna.interface.register_interface(key, interface)"
                element.interface = list(INTERFACES.keys())[list(INTERFACES.values()).index(element.interface)]

                splits.append(element)

            elif isinstance(element, SplitLayoutData):
                splits.append(element.get_serializable())

            else:
                raise TypeError(f"SplitLayoutData only accepts FieldPanel, ValuePanel, SplitLayoutData objects, found {type(element)}")
        return SplitLayoutData(split = splits, vertical_cut=self.vertical_cut, name = self.name)

    def build_item(self, panels: Dict[str, VisualizationPanel]) -> SplitLayout:
        """
        Build a SplitLayout widget from the panel configuration.

        Recursively constructs Panel widgets from the layout data using
        the provided visualization panel dictionary.

        Parameters
        ----------
        panels : Dict[str, VisualizationPanel]
            Dictionary mapping panel names to VisualizationPanel instances.

        Returns
        -------
        SplitLayout
            The constructed SplitLayout widget.

        Raises
        ------
        ValueError
            Raised if the split list is empty.
        TypeError
            Raised if contained objects are not implemented types.
        """
        direction = SplitDirection.VERTICAL if self.vertical_cut else SplitDirection.HORIZONTAL

        if len(self.split) == 0:
            raise ValueError("Can't build a SplitLayout with no children")
        elif len(self.split) == 1:
            if isinstance(self.split[0], SplitLayoutData):
                return self.split[0].build_item(panels)
            elif isinstance(self.split[0], (FieldPanel, ValuePanel)):
                return panels[self.split[0].name]
            else:
                raise TypeError(f"SplitLayoutData only accepts FieldPanel, ValuePanel, SplitLayoutData objects, found {type(self.split[0])}")
        else:
            if isinstance(self.split[0], SplitLayoutData):
                item = self.split[0].build_item(panels)
            elif isinstance(self.split[0], (FieldPanel, ValuePanel)):
                item = panels[self.split[0].name]
            else:
                raise TypeError(f"SplitLayoutData only accepts FieldPanel, ValuePanel, SplitLayoutData objects, found {type(self.split[0])}")

            for i in range(1, len(self.split)):
                if isinstance(self.split[i], SplitLayoutData):
                    new_panel = self.split[i].build_item(panels)
                elif isinstance(self.split[i], (FieldPanel, ValuePanel)):
                    new_panel = panels[self.split[i].name]
                else:
                    raise TypeError(f"SplitLayoutData only accepts FieldPanel, ValuePanel, SplitLayoutData objects, found {type(self.split[0])}")
                item = SplitItem(
                    item,
                    new_panel,
                    direction=direction
                )

            return item

def get_serializable_data(
    visualiser_data: Union[GridLayoutData, SplitLayoutData]
) -> Union[GridLayoutData, SplitLayoutData]:
    """
    Get a serializable version of the visualization layout data.

    Converts interface type objects to their string keys for JSON serialization.
    Handles both GridLayoutData and SplitLayoutData input types.

    Parameters
    ----------
    visualiser_data : Union[GridLayoutData, SplitLayoutData]
        The visualization layout data to serialize.

    Returns
    -------
    Union[GridLayoutData, SplitLayoutData]
        A new layout instance with serializable interface references.

    Raises
    ------
    AssertionError
        Raised if an interface is not registered in scivianna interfaces.
    """
    if isinstance(visualiser_data, SplitLayoutData):
        return visualiser_data.get_serializable()

    serializable_grid: List[List] = []
    for line in visualiser_data.grid:
        serializable_grid.append([])

        for element in line:
            if element is not None:
                assert element.interface in INTERFACES.values(), f"Interface {element.interface} is not registered in scivianna interfaces. Please first call scivianna.interface.register_interface(key, interface)"
                element.interface = list(INTERFACES.keys())[list(INTERFACES.values()).index(element.interface)]

            serializable_grid[-1].append(element)

    return GridLayoutData(title=visualiser_data.title, grid=serializable_grid)



class GridStackProblem(LayoutProblem):
    """
    ICoCo Problem for grid-stack visualization layouts.

    Extends LayoutProblem to support GridStackLayout-based visualizations
    where panels are arranged in a fixed grid structure.
    """

    def __init__(self, working_directory: Path, show_server: bool = True, start: bool = True):
        """
        Initialize the GridStackProblem.

        Parameters
        ----------
        working_directory : Path
            Directory for saving layout data and temporary files.
        show_server : bool, optional
            Whether to show the Panel server window (default: True).
        start : bool, optional
            Whether to start a server to access the layout (default: True).
        """
        super().__init__(layout=None, show_server = show_server, start = start)

        self._working_directory = working_directory

    def setDataFile(self, datafile):
        """
        Set the data file path for the layout configuration.

        Parameters
        ----------
        datafile : Path or str
            Path to the layout configuration file.

        Returns
        -------
        Result of parent setDataFile call.

        Raises
        ------
        TypeError
            Raised if datafile is not a Path or string.
        """
        if not isinstance(datafile, (Path, str)):
            raise TypeError(f"expected Path, got {type(datafile)}.")
        return super().setDataFile(Path(datafile))

    def initialize(self):
        """
        Initialize the GridStackProblem visualization.

        Reads the layout configuration from the data file path and creates
        the GridStackLayout with appropriate visualization panels.
        """
        import os

        print(f"server pid = {os.getpid()}")

        if isinstance(self.data_file_path, (str, Path)):
            data_to_view = GridLayoutData.model_validate_json(
                Path(self.data_file_path).read_text()
            )
        elif isinstance(self.data_file_path, GridLayoutData):
            data_to_view = self.data_file_path
        else:
            raise TypeError(f"Provided data_file_path type not implemented: {type(self.data_file_path)}")

        np_x = 1
        for line in data_to_view.grid:
            np_x *= len(line)

        self._meshes: Dict[str, mc.MEDCouplingMesh] = {}

        visualisation_panels: Dict[str, VisualizationPanel] = {}
        bounds_x = {}
        bounds_y = {}
        for ip_y, line in enumerate(data_to_view.grid):
            n_x = np_x // len(line)
            for i_x, element in enumerate(line):

                if element is None:
                    continue

                name = element.name
                if isinstance(element, ValuePanel):
                    slave_1d = ComputeSlave(element.interface)

                    slave_1d.set_time(0.)
                    for field in element.displayed_fields:
                        slave_1d.update_data(field, np.nan)

                    visualisation_panels[name] = Panel1D(slave_1d, name)
                    visualisation_panels[name].set_field(element.displayed_fields)

                elif isinstance(element, FieldPanel):
                    element: FieldPanel

                    slave_2d = ComputeSlave(element.interface)
                    slave_2d.set_time(0.)
                    slave_2d.update_policy = element.update_policy

                    if element.template is not None:
                        for template_name, value in element.template:
                            slave_2d.set_template(template_name, value)

                    visualisation_panels[name] = Panel2D(slave_2d, name=name)

                    visualisation_panels[name].set_coordinates(
                        u = element.u,
                        v = element.v,
                        u_min = element.u_min,
                        u_max = element.u_max,
                        v_min = element.v_min,
                        v_max = element.v_max,
                        w = element.w,
                    )
                    if element.color_map is not None:
                        visualisation_panels[name].set_colormap(element.color_map)

                    visualisation_panels[name].set_field(element.displayed_field)
                else:
                    raise

                ip_x = i_x * n_x
                bounds_x[name] = (ip_x, ip_x + n_x)
                bounds_y[name] = (ip_y, ip_y + 1)

        self.layout = GridStackLayout(
            visualisation_panels,
            bounds_x,
            bounds_y
        )

        self.layout.add_time_widget()
        for panel in self.layout.visualisation_panels.values():
            panel.panel_coupling_extension = self.layout.time_widget

        return super().initialize()


class SplitLayoutProblem(LayoutProblem):
    """
    ICoCo Problem for split-tree visualization layouts.

    Extends LayoutProblem to support SplitLayout-based visualizations
    where panels are arranged in a recursive tree structure with
    horizontal or vertical splits.
    """

    def __init__(self, working_directory: Path, show_server: bool = True, start: bool = True):
        """
        Initialize the SplitLayoutProblem.

        Parameters
        ----------
        working_directory : Path
            Directory for saving layout data and temporary files.
        show_server : bool, optional
            Whether to show the Panel server window (default: True).
        start : bool, optional
            Whether to start a server to access the layout (default: True).
        """
        super().__init__(layout=None, show_server = show_server, start = start)

        self._working_directory = working_directory

    def setDataFile(self, datafile):
        """
        Set the data file path for the layout configuration.

        Parameters
        ----------
        datafile : Path or str
            Path to the layout configuration file.

        Returns
        -------
        Result of parent setDataFile call.

        Raises
        ------
        TypeError
            Raised if datafile is not a Path or string.
        """
        if not isinstance(datafile, (Path, str)):
            raise TypeError(f"expected Path, got {type(datafile)}.")
        return super().setDataFile(Path(datafile))

    def initialize(self):
        """
        Initialize the SplitLayoutProblem visualization.

        Reads the layout configuration from the data file path and creates
        the SplitLayout with appropriate visualization panels.
        """
        import os

        print(f"server pid = {os.getpid()}")

        if isinstance(self.data_file_path, (str, Path)):
            data_to_view = SplitLayoutData.model_validate_json(
                Path(self.data_file_path).read_text()
            )
        elif isinstance(self.data_file_path, SplitLayoutData):
            data_to_view = self.data_file_path
        else:
            raise TypeError(f"Provided data_file_path type not implemented: {type(self.data_file_path)}")

        visualisation_panels: Dict[str, VisualizationPanel] = {}

        for element in data_to_view.get_panels():
            name = element.name
            if isinstance(element, ValuePanel):
                slave_1d = ComputeSlave(element.interface)

                slave_1d.set_time(0.)
                for field in element.displayed_fields:
                    slave_1d.update_data(field, np.nan)

                visualisation_panels[name] = Panel1D(slave_1d, name)
                visualisation_panels[name].set_field(element.displayed_fields)

            elif isinstance(element, FieldPanel):
                element: FieldPanel

                slave_2d = ComputeSlave(element.interface)
                slave_2d.set_time(0.)
                slave_2d.update_policy = element.update_policy

                if element.template is not None:
                    for template_name, value in element.template:
                        slave_2d.set_template(template_name, value)

                visualisation_panels[name] = Panel2D(slave_2d, name=name)

                visualisation_panels[name].set_coordinates(
                    u = element.u,
                    v = element.v,
                    u_min = element.u_min,
                    u_max = element.u_max,
                    v_min = element.v_min,
                    v_max = element.v_max,
                    w = element.w,
                )
                if element.color_map is not None:
                    visualisation_panels[name].set_colormap(element.color_map)

                visualisation_panels[name].set_field(element.displayed_field)
            else:
                raise

        self.layout = SplitLayout(data_to_view.build_item(
            visualisation_panels
        ))

        self.layout.add_time_widget()
        for panel in self.layout.visualisation_panels.values():
            panel.panel_coupling_extension = self.layout.time_widget

        return super().initialize()

def get_problem(
        working_directory: Path,
        data_to_view: Union[GridLayoutData, SplitLayoutData],
        use_server: bool = True,
        show: bool = True,
        start: bool = True
    ) -> Tuple[Union[GridStackProblem, SplitLayoutProblem], str]:
    """
    Create visualization objects from a working directory and layout data.

    This function creates the appropriate ICoCo problem (GridStackProblem or
    SplitLayoutProblem) based on the input data type, optionally running
    it in a separate server process for better performance.

    Parameters
    ----------
    working_directory : Path
        Directory where the MED files and visualization config are stored.
    data_to_view : Union[GridLayoutData, SplitLayoutData]
        Layout data defining what to display.
    use_server : bool, optional
        Whether to use a separate server process for the visualizer
        (default: True). Running in a server prevents the visualizer from
        slowing down the main simulation.
    show : bool, optional
        Whether to display the server window on start (default: True).
    start : bool, optional
        Whether to start a server to access the layout (default: True).

    Returns
    -------
    Tuple[Union[GridStackProblem, SplitLayoutProblem], str]
        A tuple containing:

        - The ICoCo problem instance for the visualizer.
        - The path to the serialized visualization data file (visu.json).

    Examples
    --------
    >>> layout = GridLayoutData(title="Test", grid=[[field_panel]])
    >>> problem, data_file = get_problem(Path("."), layout)
    """
    data_to_view = get_serializable_data(
        visualiser_data = data_to_view
    )
    data_file = working_directory / "visu.json"
    data_file.write_text(data_to_view.model_dump_json(indent=4), encoding="utf-8")

    if use_server:
        if isinstance(data_to_view, GridLayoutData):
            typeid = ServerManager.register(GridStackProblem)
        elif isinstance(data_to_view, SplitLayoutData):
            typeid = ServerManager.register(SplitLayoutProblem)
        else:
            raise TypeError(f"Data type {type(data_to_view)} not implemented")

        print(f"Client pid = {os.getpid()}")

        problem = ProblemClient(
            typeid=typeid,
            working_directory=working_directory,
            show_server = show,
            start = start
        )  # pylint: disable=abstract-class-instantiated
    else:
        if isinstance(data_to_view, GridLayoutData):
            problem = GridStackProblem(working_directory=working_directory, show_server = show, start = start)
        elif isinstance(data_to_view, SplitLayoutData):
            problem = SplitLayoutProblem(working_directory=working_directory, show_server = show, start = start)

    return problem, data_file
