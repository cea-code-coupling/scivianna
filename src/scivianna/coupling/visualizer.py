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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    """Plot panel name"""
    update_policy: UpdatePolicy
    """Code interface management definition"""
    interface: Type[GenericInterface]
    """Code interface"""
    template: List[Tuple[str, any]] | None = None
    """Template per displayable field (used of field projection)"""

    @field_validator("interface", mode="before")
    @classmethod
    def resolve_interface(cls, v):
        if isinstance(v, type):
            return v
        if isinstance(v, str):
            resolved = INTERFACES.get(v)
            if resolved is None:
                raise ValueError(f"Unknown interface identifier: {v}")
            return resolved
        raise TypeError(f"interface must be str or type, got {type(v).__name__}")

class FieldPanel(CouplingPanel):
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


class ValuePanel(CouplingPanel):
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

class GridLayoutData(BaseModel):
    grid: List[List[Optional[Union[FieldPanel, ValuePanel]]]]
    """Grid of the plots"""

    title: str

    @model_validator(mode="after")
    def check(self):

        found_names = set()

        for line in self.grid:
            for element in line:
                if element.name in found_names:
                    raise ValueError(f"name {element.name} appears at least twice.")

                found_names.add(element.name)

        return self

class SplitLayoutData(BaseModel):
    split: List[Union[FieldPanel, ValuePanel, "SplitLayoutData"]]
    """Intricated plit layout data"""
    vertical_cut: bool
    """Direction of the cut between two panels in the list, if True, the line separating is vertical."""

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
        """Returns a list of all FieldPanel and ValuePanel present in self and children

        Returns
        -------
        List[Union[FieldPanel, ValuePanel]]
            List of actual panels contained

        Raises
        ------
        TypeError
            One of the contained objects is not implemented.
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
        """Returns a serializable version of self (interfaces are replaced by its string key)

        Returns
        -------
        SplitLayoutData
            Serializable version of self

        Raises
        ------
        TypeError
            Self contains a not implemented type
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
    if isinstance(visualiser_data, SplitLayoutData):
        return visualiser_data.get_serializable()

    serializable_grid: List[List] = []
    for line in visualiser_data.grid:
        serializable_grid.append([])

        for element in line:
            assert element.interface in INTERFACES.values(), f"Interface {element.interface} is not registered in scivianna interfaces. Please first call scivianna.interface.register_interface(key, interface)"
            element.interface = list(INTERFACES.keys())[list(INTERFACES.values()).index(element.interface)]

            serializable_grid[-1].append(element)

    return GridLayoutData(title=visualiser_data.title, grid=serializable_grid)



class GridStackProblem(LayoutProblem):
    def __init__(self, working_directory: Path, show_server: bool = True, start: bool = True):
        super().__init__(layout=None, show_server = show_server, start = start)

        self._working_directory = working_directory

    def setDataFile(self, datafile):
        if not isinstance(datafile, (Path, str)):
            raise TypeError(f"expected Path, got {type(datafile)}.")
        return super().setDataFile(Path(datafile))

    def initialize(self):
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
                    slave_1d.update_data(name, np.nan)

                    visualisation_panels[name] = Panel1D(slave_1d, name)

                elif isinstance(element, FieldPanel):
                    element: FieldPanel

                    slave_2d = ComputeSlave(element.interface)
                    slave_2d.set_time(0.)
                    slave_2d.update_policy = element.update_policy

                    if element.template is not None:
                        for template_name, value in element.template:
                            slave_2d.set_template(template_name, value)

                    visualisation_panels[name] = Panel2D(slave_2d, name=name)
                    if element.template is not None:
                        visualisation_panels[name].set_field(element.template[-1][0])

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
    def __init__(self, working_directory: Path, show_server: bool = True, start: bool = True):
        super().__init__(layout=None, show_server = show_server, start = start)

        self._working_directory = working_directory

    def setDataFile(self, datafile):
        if not isinstance(datafile, (Path, str)):
            raise TypeError(f"expected Path, got {type(datafile)}.")
        return super().setDataFile(Path(datafile))

    def initialize(self):
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
                slave_1d.update_data(name, np.nan)

                visualisation_panels[name] = Panel1D(slave_1d, name)

            elif isinstance(element, FieldPanel):
                element: FieldPanel

                slave_2d = ComputeSlave(element.interface)
                slave_2d.set_time(0.)
                slave_2d.update_policy = element.update_policy

                if element.template is not None:
                    for template_name, value in element.template:
                        slave_2d.set_template(template_name, value)

                visualisation_panels[name] = Panel2D(slave_2d, name=name)
                if element.template is not None:
                    visualisation_panels[name].set_field(element.template[-1][0])

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
    """Creates the visualisation objects from a working dir

    Parameters
    ----------
    working_directory : str
        Directory where the med files are
    data_to_view : Union[GridLayoutData, SplitLayoutData]
        Data to diplay
    use_server : bool
        Use a server to have the visualizer running on another process, by default True
    show : bool
        Display the server on start

    Returns
    -------
    GridStackProblem
        Icoco problem for the visualizer
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
