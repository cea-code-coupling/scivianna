from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from scivianna.coupling.problem_server import ServerManager, ProblemClient

from scivianna.layout.gridstack import GridStackLayout
from scivianna.panel.panel_1d import Panel1D
from scivianna.interface.time_dataframe import TimeDataFrame
from scivianna.panel.visualisation_panel import VisualizationPanel
from scivianna.slave import ComputeSlave
from scivianna.notebook_tools import get_med_panel

from scivianna.coupling.icoco import GridStackProblem

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    PositiveFloat,
    model_validator,
)

import numpy as np
import medcoupling as mc


class VisuPanel(BaseModel):
    name: str


class FieldPanel(VisuPanel):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mesh: Union[Path, mc.MEDCouplingMesh]


class ValuePanel(VisuPanel):

    min_time: Optional[NonNegativeFloat] = None
    max_time: Optional[PositiveFloat] = None

    min_value: Optional[NonNegativeFloat] = None
    max_value: Optional[PositiveFloat] = None


class ReductionType(Enum):

    MAX = "MAX"
    MIN = "MIN"
    AVERAGE = "AVERAGE"
    SUM = "SUM"


class FieldValuePanel(VisuPanel):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mesh: Union[Path, mc.MEDCouplingMesh]

    reduction_type: ReductionType


class VisualizerData(BaseModel):

    grid: List[List[Optional[Union[FieldValuePanel, FieldPanel, ValuePanel]]]]
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

            if isinstance(element, (FieldPanel, FieldValuePanel)) and isinstance(
                element.mesh, mc.MEDCouplingMesh
            ):
                mesh_file = (
                    working_directory / f"mesh_{element.name.replace(' ', '_')}.med"
                )
                mc.WriteMesh(str(mesh_file), element.mesh, True)
                if isinstance(element, FieldPanel):
                    element = FieldPanel(name=element.name, mesh=mesh_file)
                else:
                    element = FieldValuePanel(
                        name=element.name,
                        mesh=mesh_file,
                        reduction_type=element.reduction_type,
                    )

            serializable_grid[-1].append(element)

    return VisualizerData(title=visualiser_data.title, grid=serializable_grid)


class VisuProblem(GridStackProblem):

    def __init__(self, working_directory: Path):
        super().__init__(grid_stack=None)

        self._working_directory = working_directory

        self._field_values: Dict[str, ReductionType] = {}

    def setDataFile(self, datafile):
        if not isinstance(datafile, (Path, str)):
            raise TypeError(f"expected Path, got {type(datafile)}.")
        return super().setDataFile(Path(datafile))

    def initialize(self):
        import os

        print(f"server pid = {os.getpid()}")

        data_to_view = VisualizerData.model_validate_json(
            Path(self.data_file_path).read_text()
        )

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
                    slave_result = ComputeSlave(TimeDataFrame)

                    slave_result.setTime(-1.0)
                    slave_result.setInputDoubleValue(name, np.nan)

                    visualisation_panels[name] = Panel1D(slave_result, name)

                elif isinstance(element, FieldValuePanel):
                    slave_result = ComputeSlave(TimeDataFrame)

                    slave_result.setTime(-1.0)
                    slave_result.setInputDoubleValue(name, np.nan)

                    visualisation_panels[name] = Panel1D(slave_result, name)
                    self._field_values[name] = element.reduction_type
                    mesh = mc.ReadMeshFromFile(str(element.mesh))
                    self._meshes[name] = mesh

                elif isinstance(element, FieldPanel):
                    element: FieldPanel

                    mesh = mc.ReadMeshFromFile(str(element.mesh))
                    self._meshes[name] = mesh

                    field = self.get_field_template(mesh=mesh, name=name)
                    field_path = (
                        self._working_directory / f"field_{name.replace(' ', '_')}.med"
                    )
                    mc.WriteField(str(field_path), field, True)
                    visualisation_panels[name] = get_med_panel(
                        str(field_path), title=name
                    )
                    visualisation_panels[name].set_field(name)

                else:
                    raise

                ip_x = i_x * n_x
                bounds_x[name] = (ip_x, ip_x + n_x)
                bounds_y[name] = (ip_y, ip_y + 1)

        # mfldsn
        #   Adding the run button to be able to start the synchronisation to the coupling
        self.gridstack = GridStackLayout(
            visualisation_panels, bounds_x, bounds_y, add_run_button=True
        )

        self.gridstack.add_time_widget()

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

        long_name = f"{name}@{name}"

        if isinstance(afield, mc.DataArrayDouble):
            afield = self.get_field_template(
                array=afield, mesh=self._meshes[name], name=long_name
            )
        print(f"-------> {long_name=}, {type(afield)=}")
        return super().setInputMEDDoubleField(long_name, afield)

    def getInputMEDDoubleFieldTemplate(self, name):
        if name in self._field_values:
            return self.get_field_template(mesh=self._meshes[name], name=name)

        name = f"{name}@{name}"
        return super().getInputMEDDoubleFieldTemplate(name)

    def setInputDoubleValue(self, name, val):
        name = f"{name}@{name}"
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




class VisuClient(ProblemClient):

    def setInputMEDDoubleField(self, name, afield):
        return super().setInputMEDDoubleField(name, afield.getArray())


def get_problem(working_directory: Path, data_to_view: VisualizerData) -> VisuProblem:
    """Creates the visualisation objects from a working dir

    Parameters
    ----------
    working_directory : str
        Directory where the med files are
    data_to_view : str
        Data to diplay

    Returns
    -------
    VisuProblem
        Icoco problem for the visializer
    """

    data_to_view = get_serializable_data(
        working_directory=working_directory, visualiser_data=data_to_view
    )
    data_file = working_directory / "data_inputs_neutro.json"
    data_file.write_text(data_to_view.model_dump_json(indent=4), encoding="utf-8")

    typeid = ServerManager.register(VisuProblem)

    import os

    print(f"client pid = {os.getpid()}")

    problem = VisuClient(
        typeid=typeid, working_directory=working_directory
    )  # pylint: disable=abstract-class-instantiated

    return problem, data_file
