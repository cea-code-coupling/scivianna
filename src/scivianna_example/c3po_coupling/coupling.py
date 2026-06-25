import os
from pathlib import Path
from typing import Optional, Tuple, List, Any

from c3po.physicsDrivers.ICOCODriver import ICOCODriver
import c3po

import scivianna
from scivianna.constants import X, Z
from scivianna_example.c3po_coupling.fake_driver import DecreasingFieldProblem
from scivianna.coupling.visualizer import (
    GridLayoutData,
    SplitLayoutData,
    FieldPanel,
    ValuePanel,
    get_problem,
    UpdatePolicy
)

from scivianna.interface.med_interface import MEDInterface
from scivianna.interface.time_dataframe import TimeDataFrame


def get_panel(
        working_directory: Path = Path("./results"),
        computation_time: float = 0.01,
        use_server: bool = True,
        show: bool = False,
        grid: bool = False,
        return_slaves: bool = False,
        start: bool = True,
        *args,
        **kwargs
    ) -> Optional[Any]:
    """
    Set up and run a coupled C3PO-scivianna visualization demo.

    This function creates a complete coupling chain between a physics driver
    (DecreasingFieldProblem) and the scivianna visualizer. It handles the
    creation of exchangers, coupler, and executes a transient simulation
    while displaying real-time visualization.

    The function supports two layout modes:
    - Grid layout (grid=True): Panels arranged in a fixed grid structure
    - Split layout (grid=False): Panels arranged in a recursive tree structure

    Parameters
    ----------
    working_directory : Path, optional
        Directory for output files and visualization data (default: "./results").
    computation_time : float, optional
        Total simulation time in seconds (default: 0.01).
    use_server : bool, optional
        Whether to run the visualizer in a separate server process
        (default: True). Running in a server prevents the visualizer from
        slowing down the main simulation.
    show : bool, optional
        Whether to display the visualization window on start (default: False).
    grid : bool, optional
        If True, use GridLayoutData for panel arrangement; otherwise use
        SplitLayoutData (default: False).
    return_slaves : bool, optional
        If True, return the visualization slaves for programmatic access
        (default: False).
    start : bool, optional
        Whether to start the simulation immediately (default: True).
    *args : tuple
        Additional positional arguments passed to internal components.
    **kwargs : dict
        Additional keyword arguments passed to internal components.

    Returns
    -------
    Optional[Any]
        If return_slaves is False: returns the visualization layout (or None
        if use_server is True).
        If return_slaves is True: returns a tuple of (layout, slaves_list)
        or (None, slaves_list) if using server.

    Examples
    --------
    Run with default settings::

        from pathlib import Path
        get_panel(working_directory=Path("./results"), computation_time=10.)

    Run with grid layout and return slaves::

        layout, slaves = get_panel(grid=True, return_slaves=True)

    See Also
    --------
    scivianna.coupling.visualizer.get_problem : Creates the visualization problem.
    c3po.LocalExchanger : Handles data exchange between codes.
    c3po.Coupler : Manages the coupled simulation time stepping.
    """
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
                        name = "Field",
                        interface = MEDInterface,
                        update_policy = UpdatePolicy.APPEND_DATA,
                        template = [
                            ("Field value", str(Path(scivianna.__file__).parent / "input_file" / "power.med")),
                        ],
                        u=X,
                        v=Z,
                        displayed_field = "Field value"
                    ),
                    ValuePanel(
                        name = "Temperature",
                        interface = TimeDataFrame,
                        update_policy = UpdatePolicy.APPEND_DATA,
                        displayed_fields=["MIN", "AVERAGE", "MAX"]
                    ),
                ]
            ],
            title="C3PO coupling demo",
        )
    else:
        visualizer_data = SplitLayoutData(
            split = [
                FieldPanel(
                    name = "Field",
                    interface = MEDInterface,
                    update_policy = UpdatePolicy.APPEND_DATA,
                    template = [
                        ("Field value", str(Path(scivianna.__file__).parent / "input_file" / "power.med")),
                    ],
                    u=X,
                    v=Z,
                    displayed_field = "Field value"
                ),
                ValuePanel(
                    name = "Temperature",
                    interface = TimeDataFrame,
                    update_policy = UpdatePolicy.APPEND_DATA,
                    displayed_fields=["MIN", "AVERAGE", "MAX"]
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
        show = show,
        start = start
    )
    myVIZDriver = ICOCODriver(visu_problem)
    myVIZDriver.setDataFile(visu_data_file)
    myVIZDriver.init()

    LocalExchanger = c3po.tracer()(c3po.LocalExchanger)

    Exchanger_to_Visualizer = LocalExchanger(
        method=c3po.DirectMatching(),

        fieldsToGet=[(fieldDriver, "VALUE")],
        fieldsToSet=[(myVIZDriver, "Field@Field value")],

        valuesToGet=[
            (fieldDriver, "MAX"),
            (fieldDriver, "AVERAGE"),
            (fieldDriver, "MIN"),
        ],
        valuesToSet=[
            (myVIZDriver, "Temperature@MAX"),
            (myVIZDriver, "Temperature@AVERAGE"),
            (myVIZDriver, "Temperature@MIN"),
        ],
    )

    class ExplicitCoupler(c3po.Coupler):
        """
        Coupler that defines the coupled simulation logic.

        During each time step, this coupler:
        1. Solves the physics driver
        2. Exchanges data with the visualizer
        3. Updates the visualization
        4. Tracks the maximum power value

        Parameters
        ----------
        physics : dict
            Dictionary mapping driver names to physics driver instances.
        exchangers : dict
            Dictionary mapping exchanger names to exchanger instances.
        dataManagers : list, optional
            List of data manager instances (default: []).

        Attributes
        ----------
        maxPower : float
            Tracks the maximum power value observed during simulation.
        """

        def __init__(self, physics, exchangers, dataManagers=None):
            if dataManagers is None:
                dataManagers = []
            c3po.Coupler.__init__(self, physics, exchangers, dataManagers)
            self.maxPower = 0.0

        def solveTimeStep(self):
            """
            Execute one coupled time step.

            Solves the physics driver, exchanges data with the visualizer,
            updates the visualization, and tracks the maximum power value.

            Returns
            -------
            bool
                The solve status from the parent Coupler class.
            """
            self._physicsDrivers["PHY"].solveTimeStep()
            self._exchangers["PHY_2_VIZ"].exchange()
            power = self._physicsDrivers["PHY"].getOutputDoubleValue("MAX")
            self._physicsDrivers["VISU"].solve()
            if power > self.maxPower:
                self.maxPower = power
            return self.getSolveStatus()

        def computeTimeStep(self):
            """
            Compute the preferred time step for the coupled simulation.

            Returns
            -------
            tuple
                A tuple containing (time_step, stop_flag) where time_step
                is 5.0e-4 seconds and stop_flag is False.
            """
            return (5.0e-4, False)  # This define time-step size

    transientCoupler = ExplicitCoupler(
        {
            "PHY": fieldDriver,
            "VISU": myVIZDriver
        },
        {
            "PHY_2_VIZ": Exchanger_to_Visualizer,
        },
    )

    transientCoupler.maxPower = 0.0
    transientCoupler.setStationaryMode(False)
    transientCoupler.solveTransient(computation_time)

    if start:
        myVIZDriver.term()

    if return_slaves:
        return visu_problem.layout if not use_server else None, [
            v.get_slave() for v in visu_problem.layout.visualisation_panels.values()
        ]
    else:
        return visu_problem.layout if not use_server else None


if __name__ == "__main__":
    get_panel(show=True, computation_time=0.1)
