import os
from pathlib import Path
from c3po.physicsDrivers.ICOCODriver import ICOCODriver
import c3po

import scivianna
from scivianna.constants import X, Z
from scivianna_example.c3po_coupling.fake_driver import DecreasingFieldProblem
from scivianna.coupling.visualizer import GridLayoutData, SplitLayoutData, FieldPanel, ValuePanel, get_problem, UpdatePolicy

from scivianna.interface.med_interface import MEDInterface
from scivianna.interface.time_dataframe import TimeDataFrame

def get_panel(
        working_directory: Path = Path("./results"),
        computation_time = .01,
        use_server = True,
        show = False,
        grid = False,
        *args, **kwargs
    ):
    # Building of objects driving codes
    fieldDriver = DecreasingFieldProblem(
        str(Path(scivianna.__file__).parent / "input_file" / "power.med")
    )
    fieldDriver.initialize()

    if grid:
        visualizer_data = GridLayoutData(
            grid=[
                # first line
                [
                    FieldPanel(
                        name = "Field value",
                        interface = MEDInterface,
                        update_policy = UpdatePolicy.APPEND_DATA,
                        template = [
                            ("Field value", str(Path(scivianna.__file__).parent / "input_file" / "power.med")),
                        ],
                        u=X,
                        v=Z
                    ),
                    ValuePanel(
                        name = "MAX",
                        interface = TimeDataFrame,
                        update_policy = UpdatePolicy.APPEND_DATA
                    ),
                ]
            ],
            title="C3PO coupling demo",
        )
    else:
        visualizer_data = SplitLayoutData(
            split = [
                FieldPanel(
                    name = "Field value",
                    interface = MEDInterface,
                    update_policy = UpdatePolicy.APPEND_DATA,
                    template = [
                        ("Field value", str(Path(scivianna.__file__).parent / "input_file" / "power.med")),
                    ],
                    u=X,
                    v=Z
                ),
                ValuePanel(
                    name = "MAX",
                    interface = TimeDataFrame,
                    update_policy = UpdatePolicy.APPEND_DATA
                ),
            ],
            vertical_cut=True,
            name="C3PO coupling demo",
        )

    os.makedirs(working_directory, exist_ok=True)
    visu_problem, visu_data_file = get_problem(
        working_directory = working_directory,
        data_to_view = visualizer_data,
        use_server = use_server,
        show = show
    )
    myVIZDriver = ICOCODriver(visu_problem)
    myVIZDriver.setDataFile(visu_data_file)
    myVIZDriver.init()

    LocalExchanger = c3po.tracer()(c3po.LocalExchanger)

    # 3. Exchanges between the codes and the visualizer
    Exchanger_to_Visualizer = LocalExchanger(
        method=c3po.DirectMatching(),

        fieldsToGet=[(fieldDriver, "VALUE")],
        fieldsToSet=[(myVIZDriver, "Field value@Field value")],

        valuesToGet=[
            (fieldDriver, "MAX"),
        ],
        valuesToSet=[
            (myVIZDriver, "MAX@MAX"),
        ],
    )

    # Definition of a class defining what "doing an time-step" means.
    class ExplicitCoupler(c3po.Coupler):
        def __init__(self, physics, exchangers, dataManagers=[]):
            c3po.Coupler.__init__(self, physics, exchangers, dataManagers)
            self.maxPower = 0.0

        def solveTimeStep(self):
            self._physicsDrivers["PHY"].solveTimeStep()
            self._exchangers["PHY_2_VIZ"].exchange()
            power = self._physicsDrivers["PHY"].getOutputDoubleValue("MAX")
            self._physicsDrivers["VISU"].solve()
            if power > self.maxPower:
                self.maxPower = power
            return self.getSolveStatus()

        def computeTimeStep(self):
            return (5.0e-4, False)  # This define time-step size

    # Building of a ExplicitCoupler object
    transientCoupler = ExplicitCoupler(
        {
            "PHY": fieldDriver,
            "VISU": myVIZDriver
        },
        {
            "PHY_2_VIZ": Exchanger_to_Visualizer,
        },
    )

    # Transient
    transientCoupler.maxPower = 0.0
    transientCoupler.setStationaryMode(False)
    transientCoupler.solveTransient(computation_time)

    myVIZDriver.term()


if __name__ == "__main__":
    get_panel(show=True, computation_time=10.)
