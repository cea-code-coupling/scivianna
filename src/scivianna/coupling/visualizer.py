from enum import Enum, auto
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Type, Tuple

from scivianna.coupling.problem_server import ServerManager, ProblemClient
from scivianna.enums import UpdatePolicy
from scivianna.layout.gridstack import GridStackLayout
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


class VisuPanel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    update_policy: UpdatePolicy
    interface: Type[GenericInterface]
    template: List[Tuple[str, any]] | None = None

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

class FieldPanel(VisuPanel):
    interface: Type[Geometry2D]


class ValuePanel(VisuPanel):
    min_time: Optional[NonNegativeFloat] = None
    max_time: Optional[PositiveFloat] = None

    min_value: Optional[NonNegativeFloat] = None
    max_value: Optional[PositiveFloat] = None

    interface: Union[Type[Value1DAtLocation], Type[ValueAtLocation]]


class ReductionType(Enum):
    MAX = "MAX"
    MIN = "MIN"
    AVERAGE = "AVERAGE"
    SUM = "SUM"


class VisualizerData(BaseModel):
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


def get_serializable_data(
    visualiser_data: VisualizerData, working_directory: Path
) -> VisualizerData:
    serializable_grid: List[List] = []
    for line in visualiser_data.grid:
        serializable_grid.append([])

        for element in line:
            slave = element.interface.get_slave()

            assert element.interface in INTERFACES.values(), f"Interface {element.interface} is not registered in scivianna interfaces. Please first call scivianna.interface.register_interface(key, interface)"
            element.interface = list(INTERFACES.keys())[list(INTERFACES.values()).index(element.interface)]
            
            serializable_grid[-1].append(element)

    return VisualizerData(title=visualiser_data.title, grid=serializable_grid)



class GridStackProblem(LayoutProblem):
    def __init__(self, working_directory: Path, show_server: bool = True):
        super().__init__(layout=None, show_server = show_server)

        self._working_directory = working_directory

        self._field_values: Dict[str, ReductionType] = {}

    def setDataFile(self, datafile):
        # if not isinstance(datafile, (Path, str)):
        #     raise TypeError(f"expected Path, got {type(datafile)}.")
        # return super().setDataFile(Path(datafile))
        return super().setDataFile(datafile)

    def initialize(self):
        import os

        print(f"server pid = {os.getpid()}")

        if isinstance(self.data_file_path, (str, Path)):
            data_to_view = VisualizerData.model_validate_json(
                Path(self.data_file_path).read_text()
            )
        elif isinstance(self.data_file_path, VisualizerData):
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
                else:
                    raise

                ip_x = i_x * n_x
                bounds_x[name] = (ip_x, ip_x + n_x)
                bounds_y[name] = (ip_y, ip_y + 1)

        # mfldsn
        #   Adding the run button to be able to start the synchronisation to the coupling
        self.layout = GridStackLayout(
            visualisation_panels, 
            bounds_x, 
            bounds_y
        )

        self.layout.add_time_widget()
        for panel in self.layout.visualisation_panels.values():
            panel.panel_coupling_extension = self.layout.time_widget

        return super().initialize()

    def setInputMEDDoubleField(self, name, afield):
        print(f"-------> {name=}, {type(afield)=}")
        if name in self._field_values:

            if not isinstance(afield, mc.DataArrayDouble):
                afield = afield.getArray()

            if self._field_values[name] == ReductionType.MAX:
                value = afield.getMaxValue()[0]
            elif self._field_values[name] == ReductionType.MIN:
                value = afield.getMinValue()[0]
            elif self._field_values[name] == ReductionType.AVERAGE:
                value = afield.getAverageValue()
            elif self._field_values[name] == ReductionType.SUM:
                value = afield.accumulate()
            else:
                raise ValueError(
                    f"{name} not in {[member.name for member in ReductionType]}"
                )

            return self.setInputDoubleValue(name, value)

        if isinstance(afield, mc.DataArrayDouble):
            afield = self.get_field_template(
                array=afield, mesh=self._meshes[name], name=name
            )
        print(f"-------> {name=}, {type(afield)=}")
        return super().setInputMEDDoubleField(name, afield)

    def getInputMEDDoubleFieldTemplate(self, name):
        if name in self._field_values:
            return self.get_field_template(mesh=self._meshes[name], name=name)

        return super().getInputMEDDoubleFieldTemplate(name)

    def setInputDoubleValue(self, name, val):
        return super().setInputDoubleValue(name, val)

    def get_field_template(self, name: str,
                        mesh: mc.MEDCouplingUMesh,
                        array: Optional[mc.DataArrayDouble] = None,
                        nature: Optional[int] = mc.IntensiveMaximum
                        ) -> mc.MEDCouplingFieldDouble:

        mcfield = mc.MEDCouplingFieldDouble(mc.ON_CELLS, mc.ONE_TIME)
        mcfield.setName(name)
        mcfield.setTime(0., 0, 0)
        mcfield.setMesh(mesh)
        if array is None:
            array = mc.DataArrayDouble([0.] * mesh.getNumberOfCells())
        mcfield.setArray(array)
        if nature is not None:
            mcfield.setNature(nature)
        return mcfield


def get_grid_stack_problem(
        working_directory: Path, 
        data_to_view: VisualizerData,
        use_server: bool = True,
        show: bool = True
    ) -> GridStackProblem:
    """Creates the visualisation objects from a working dir

    Parameters
    ----------
    working_directory : str
        Directory where the med files are
    data_to_view : str
        Data to diplay
    use_server : bool
        Use a server to have the visualizer running on another process, by default True
    show : bool
        Display the server on start

    Returns
    -------
    GridStackProblem
        Icoco problem for the visializer
    """
    data_to_view = get_serializable_data(
        working_directory = working_directory, 
        visualiser_data = data_to_view
    )
    data_file = working_directory / "data_inputs_neutro.json"
    data_file.write_text(data_to_view.model_dump_json(indent=4), encoding="utf-8")

    if use_server:
        typeid = ServerManager.register(GridStackProblem)

        print(f"Client pid = {os.getpid()}")

        problem = ProblemClient(
            typeid=typeid, 
            working_directory=working_directory,
            show_server = show
        )  # pylint: disable=abstract-class-instantiated
    else:
        problem = GridStackProblem(working_directory=working_directory, show_server = show)

    return problem, data_file
