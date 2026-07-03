# pylint: disable=notimplemented-raised, too-many-lines, too-many-public-methods
"""
ICoCo compatible interface to couple the visualizer to other codes using the coupling tool C3PO

ICOCO interface API:
https://github.com/cea-trust-platform/icoco-coupling

C3PO coupling tool:
https://github.com/code-coupling/c3po


The process threads structure of a coupling with the visualiser works as follow:


|             C3PO used process               |   Slave 1 process  |
|   C3PO Thread     -       Visualizer thread |   Slave 1 thread   |

        *
            C3PO objects, and drivers creation
        *   --------------------------------------->    *
            For each Visualization panel, a new process is created with a slave that will provide the data to the visualizer (1 in this example)

        *
            LayoutDriver initialisation
        *   ------------------->    *
            pn.serve() -> creation of the visualizer thread
                                    *   ----------->    *
                                    *   <-----------    *
            At each update of a panel (requested by the visualizer thread, on a button clic for example), an exchange of request is done between the visualizer thread and its associated thread.
                                    *
            The visualizer thread can then request itself to update the visualizer at the end of the tick based on the data received from the slave.

        *   --------------------------------------->    *
            At each setInputMedFieldDouble call, the C3PO thread provides the slave associated to the field the new version of the field.
        *
            If the C3PO Thread requests an update to the Visualizer, in the end, the C3PO Thread requests a plot update, two cases:
                -   The C3PO Thread doesn't have the hand on the plots, throwing an exception;
                -   The C3PO Thread adds a task, it is not read by the Visualizer thread, therefore not updating.

                                    *
            The visualizer update has to come from the Visualizer thread to be able to work:
                -   A play button was added, if clicked, a periodic task is added (from the Visualizer thread which handles the GUI) every second to refresh the visualization;
                -   The C3PO Thread will set a boolean to mark the need for an update;
                -   At each periodic task, if the boolean is at True, the Visualizer thread refreshes the geometry
"""
from pathlib import Path
from typing import List, Tuple
import time

import atexit
import medcoupling  # type: ignore
from icoco.exception import WrongContext, WrongArgument
from icoco.problem import Problem, ValueType

import panel as pn
import socket

from scivianna.enums import UpdatePolicy
from scivianna.interface.generic_interface import CouplingInterface
from scivianna.layout.generic_layout import GenericLayout
from scivianna.layout.split import SplitLayout
from scivianna.layout.gridstack import GridStackLayout
from scivianna.panel.panel_1d import Panel1D
from scivianna.utils.serialization import save_gridstack_to_zip, save_layout_to_zip


class Value:
    """
    Container for ICoCo value name constants used in the coupling system.

    Attributes
    ----------
    UPDATE_RATE : str
        Key for the update rate parameter. Defines how frequently updates
        are performed (1 update per UPDATE_RATE calls to solveTimeStep).
    """

    UPDATE_RATE = "UPDATE_RATE"
    NOTIFICATION_SUCCESS = "NOTIFICATION_SUCCESS"
    NOTIFICATION_INFO = "NOTIFICATION_INFO"
    NOTIFICATION_ERROR = "NOTIFICATION_ERROR"

class LayoutProblem(Problem):
    """
    ICoCo Problem implementation for the C3PO coupling visualizer.

    This class extends the ICoCo Problem interface to provide visualization
    capabilities through Panel. It manages the simulation time stepping,
    field data exchange, and connection monitoring for client sessions.
    """

    def __init__(
        self,
        layout: GenericLayout,
        title="C3PO Coupling visualizer",
        show_server: bool = True,
        start: bool = True
    ):
        """
        ICoCo Problem implementation for the C3PO coupling visualizer.

        This class extends the ICoCo Problem interface to provide visualization
        capabilities through Panel. It manages the simulation time stepping,
        field data exchange, and connection monitoring for client sessions.

        Parameters
        ----------
        layout : GenericLayout
            The layout containing visualization panels to manage.
        title : str, optional
            Title for the visualizer window (default: "C3PO Coupling visualizer").
        show_server : bool, optional
            Whether to show the Panel server window (default: True).
        start : bool, optional
            Whether to start the server immediately (default: True).
        """

        self._working_directory: Path = None
        self._active_connections: set = set()
        self.layout: GenericLayout = layout

        self.time = 0.0
        self._dt: float = -1.0
        self._up_rate = 1
        self._up_skipped = 0
        self.stationary = True

        self.data_file_path = None
        self.title = title
        self.show_server = show_server
        self.start = start
        self._initialized = False
        atexit.register(self._terminate)

        # Register session lifecycle callbacks to track active connections
        if start:
            pn.state.on_session_created(self._on_session_created)

    @staticmethod
    def _split_name(name: str) -> tuple[str, str]:
        """Gets visualization panel name and field name."""
        visualization_panel = name.split("@")[0]
        return visualization_panel, name.replace(f"{visualization_panel}@", "")

    def setDataFile(self, datafile: str) -> None:
        """(Optional) Provide the relative path of a data file to be used by the code.

        This method must be called before initialize().

        Parameters
        ----------
        datafile : str
            relative path to the data file.

        Raises
        ------
        WrongContext
            exception if called multiple times or after initialize().
        WrongArgument
            exception if an invalid path is provided.
        """
        self.data_file_path = datafile

    def initialize(self) -> bool:
        """
        Initialize the current problem instance.

        In this method the code should allocate all its internal structures and be ready to execute.

        File reads, memory allocations, and other operations likely to fail
        should be performed here, and not in the constructor (and not in the
        setDataFile() or in the setMPIComm() methods either).

        This method must be called only once (after a potential call to
        setMPIComm() and/or setDataFile()) and cannot be called again before
        terminate() has been performed.

        Returns
        -------
        bool
            True if all OK, otherwise False.

        Raises
        ------
        WrongContext
            Raised if called multiple times or after initialize().
        """
        print("\n\nVisualizer initializing\n\n")
        ip_adress = socket.gethostbyname(socket.gethostname())

        for panel in self.layout.visualisation_panels:
            assert "@" not in panel, f"Character @ can't be in a panel name, found {panel}."

        """
            Catching a free port to provide to pn.serve
        """
        sock = socket.socket()
        sock.bind((ip_adress, 0))
        port = sock.getsockname()[1]
        sock.close()

        self.layout.open_time_panel()

        self.server = pn.serve(
            self.layout.main_frame,
            address=ip_adress,
            websocket_origin=f"{ip_adress}:{port}",
            port=port,
            show = self.show_server,
            threaded=True,
            title=self.title,
            start = self.start
        )

        self.panels_to_recompute: List[str] = []

        self._dt = None
        self.time = 0.0
        self._up_rate = 1
        self._up_skipped = 0

        self._initialized = True
        return self._initialized

    def _on_session_created(self, session_context):
        """
        Callback called when a new client session is created.

        Adds the session ID to the active connections set and registers
        a destruction callback to track when the session ends.

        Parameters
        ----------
        session_context : SessionContext
            The session context for the new connection.
        """
        self._active_connections.add(session_context.id)
        pn.state.on_session_destroyed(self._on_session_destroyed)
        print(f"[LayoutProblem] New connection: {session_context.id}. Active connections: {len(self._active_connections)}")

    def _on_session_destroyed(self, session_context):
        """
        Callback called when a client session is destroyed.

        Removes the session ID from the active connections set to track
        disconnections.

        Parameters
        ----------
        session_context : SessionContext
            The session context for the disconnected client.
        """
        self._active_connections.discard(session_context.id)
        print(f"[LayoutProblem] Connection closed: {session_context.id}. Active connections: {len(self._active_connections)}")

    def wait_for_disconnect(self, poll_interval: float = 0.5) -> None:
        """
        Wait until all client connections have been disconnected.

        This method blocks the calling thread until no active connections remain.
        It polls at the specified interval to check for remaining connections.

        Parameters
        ----------
        poll_interval : float, optional
            Time in seconds between connection checks (default: 0.5).
        """
        if self._active_connections:
            self.layout.notifications.append(("success", "COUPLING COMPLETE. CLOSE YOUR BROWSER TAB TO TERMINATE THE SIMULATION.", 0))

        print(f"[LayoutProblem] Waiting for {len(self._active_connections)} active connection(s) to disconnect...")
        while len(self._active_connections) > 0:
            time.sleep(poll_interval)
        print("[LayoutProblem] All connections disconnected.")

    def terminate(self) -> None:
        """
        Terminate the current problem instance and release all allocated resources.

        Terminate the computation, free the memory and save whatever needs to be saved.

        This method is called once at the end of the computation or after
        a non-recoverable error. No other ICoCo method except setDataFile(),
        setMPIComm() and initialize() may be called after this.

        Raises
        ------
        WrongContext
            Raised if called before initialize() or after terminate().
        WrongContext
            Raised if called inside the TIME_STEP_DEFINED context
            (see Problem documentation).
        """
        if not self._terminate():
            raise WrongContext(method="terminate",
                               precondition="called before initialize() or after terminate()",
                               prob="LayoutProblem")


    def _terminate(self) -> bool:
        if not self._initialized:
            return False

        self._dt = -1.0
        self.time = 0.0
        self._up_rate = 1
        self._up_skipped = 0

        Path(self._working_directory).mkdir(exist_ok=True, parents=True)

        if self._working_directory is not None:
            if isinstance(self.layout, GridStackLayout):
                save_gridstack_to_zip(self.layout, Path(self._working_directory) / "save_layout")
            elif isinstance(self.layout, SplitLayout):
                save_layout_to_zip(self.layout, Path(self._working_directory) / "save_layout")
            else:
                raise TypeError(f"Layout type {type(self.layout)} not implemented.")

        # Wait for all client connections to disconnect before stopping the server
        self.wait_for_disconnect()

        for panel in self.layout.visualisation_panels.values():
            print(f"Terminating panel {panel.panel_name}")
            panel.get_slave().terminate()

        self.server.stop()

        self.data_file_path = None
        self._initialized = False
        return True

    def presentTime(self) -> float:
        """
        Return the current time of the simulation.

        This method can be called any time between initialize() and terminate().
        The current time can only change during a call to validateTimeStep() or
        to reset_time().

        Returns
        -------
        float
            The current (physical) time of the simulation.

        Raises
        ------
        WrongContext
            Raised if called before initialize() or after terminate().
        """
        return self.time

    def computeTimeStep(self) -> Tuple[float, bool]:
        """
        Return the next preferred time step and whether to stop.

        Both values are only indicative; the supervisor is not required to
        take them into account. This method is marked as mandatory since most
        of the coupling schemes expect the code to provide this information
        (those schemes then typically compute the minimum of the time steps
        of all the codes being coupled).

        This method can be called whenever the code is outside the
        TIME_STEP_DEFINED context (see Problem documentation).

        Returns
        -------
        Tuple[float, bool]
            A tuple containing:

            - The preferred time step for this code (only valid if stop is False).
            - Stop flag set to True if the code wants to stop. It can be used
              to indicate that, according to a certain criterion, the end of
              the transient computation is reached from the code point of view.

        Raises
        ------
        WrongContext
            Raised if called before initialize() or after terminate().
        WrongContext
            Raised if called inside the TIME_STEP_DEFINED context
            (see Problem documentation).
        """
        return 1e8, False

    def initTimeStep(self, dt: float) -> bool:
        """
        Provide the next time step (time increment) to be used by the code.

        After this call (if successful), the computation time step is defined
        to ]t, t + dt] where t is the value returned by presentTime(). The code
        enters the TIME_STEP_DEFINED context.

        A time step of 0.0 may be used when the stationaryMode is set to True
        for codes solving directly for the steady-state.

        Parameters
        ----------
        dt : float
            The time step to be used by the code.

        Returns
        -------
        bool
            False means the given time step is not compatible with the code
            time scheme.

        Raises
        ------
        WrongContext
            Raised if called before initialize() or after terminate().
        WrongContext
            Raised if called inside the TIME_STEP_DEFINED context
            (see Problem documentation).
        WrongContext
            Raised if called several times without resolution.
        WrongArgument
            Raised if dt is invalid (dt < 0.0).
        """

        if self._dt is not None and self._dt < 0.0:
            raise WrongContext(
                prob=f"{self.__class__.__module__}.{self.__class__.__name__}",
                method="initTimeStep",
                precondition="must be called after initialize.",
            )
        if self._dt is not None:
            raise WrongContext(
                prob=f"{self.__class__.__module__}.{self.__class__.__name__}",
                method="initTimeStep",
                precondition="must be called outside time step context.",
            )
        if dt < 0.0:
            raise WrongArgument(
                prob=f"{self.__class__.__module__}.{self.__class__.__name__}",
                method="initTimeStep",
                arg="dt",
                condition="must be >= 0.0.",
            )

        self._dt = dt

    def solveTimeStep(self) -> bool:
        """
        Perform the computation on the current time interval.

        This method can be called whenever the code is inside the
        TIME_STEP_DEFINED context (see Problem documentation).

        If the update rate has been reached, it marks panels for
        recomputation and triggers a visual update.

        Returns
        -------
        bool
            True if computation was successful, False otherwise.

        Raises
        ------
        WrongContext
            Raised if called before initialize() or after terminate().
        WrongContext
            Raised if called outside the TIME_STEP_DEFINED context
            (see Problem documentation).
        WrongContext
            Raised if called several times without a call to
            validateTimeStep() or abortTimeStep().
        """
        if self._up_skipped == self._up_rate:
            # here we should transfer data and update the visu
            self.layout.mark_to_recompute(self.panels_to_recompute)
            self.panels_to_recompute.clear()
        return True

    def validateTimeStep(self) -> None:
        """
        Validate the computation performed by solveTimeStep.

        This method can be called whenever the code is inside the
        TIME_STEP_DEFINED context (see Problem documentation).

        After this call:

        - The present time has been advanced to the end of the computation
          time step.
        - The computation time step is undefined (the code leaves the
          TIME_STEP_DEFINED context).

        Raises
        ------
        WrongContext
            Raised if called before initialize() or after terminate().
        WrongContext
            Raised if called outside the TIME_STEP_DEFINED context
            (see Problem documentation).
        WrongContext
            Raised if called before the solveTimeStep() method.
        """

        self.time += self._dt
        self._dt = None

        if self._up_skipped == self._up_rate:
            self.layout.time_widget.add_time_value(self.time)

        if self._up_skipped == self._up_rate:
            self._up_skipped = 0
        else:
            self._up_skipped += 1

    def abortTimeStep(self):
        """
        Abort the current time step computation.

        Resets the time step to None without validating the computation.
        """
        self._dt = None

    def setStationaryMode(self, stationaryMode: bool) -> None:
        """
        Set whether the code should compute a stationary or transient solution.

        New in version 2 of ICoCo. By default the code is assumed to be in
        stationary mode False (i.e., set up for a transient computation).

        If set to True, solveTimeStep() can be used either to solve a time step
        in view of an asymptotic solution, or to solve directly for the
        steady-state. In this last case, a time step of 0. can be used with
        initTimeStep() (whose call is always needed).

        The stationary mode status of the code can only be modified by this
        method (or by a call to terminate() followed by initialize()).

        Parameters
        ----------
        stationaryMode : bool
            True if the code should compute a stationary solution.

        Raises
        ------
        WrongContext
            Raised if called inside the TIME_STEP_DEFINED context
            (see Problem documentation).
        WrongContext
            Raised if called before initialize() or after terminate().
        """
        self.stationary = stationaryMode

    def getStationaryMode(self) -> bool:
        """
        Indicate whether the code should compute a stationary or transient solution.

        See also setStationaryMode().

        This method can be called whenever the code is outside the
        TIME_STEP_DEFINED context (see Problem documentation).

        Returns
        -------
        bool
            True if the code has been set to compute a stationary solution.

        Raises
        ------
        WrongContext
            Raised if called before initialize() or after terminate().
        WrongContext
            Raised if called inside the TIME_STEP_DEFINED context
            (see Problem documentation).
        """
        return self.stationary

    def getValueType(self, name: str) -> ValueType:
        """
        Get the value type for a given parameter name.

        Parameters
        ----------
        name : str
            The name of the parameter to query.

        Returns
        -------
        ValueType
            The ICoCo value type corresponding to the name.

        Raises
        ------
        NotImplemetedError
            Raised if the name is not recognized.
        """
        if name == Value.UPDATE_RATE:
            return ValueType.Int
        if name in [
            Value.NOTIFICATION_SUCCESS,
            Value.NOTIFICATION_INFO,
            Value.NOTIFICATION_ERROR
        ]:
            return "String"

        super().getValueType(name=name)

    def getInputValuesNames(self):
        """
        Get the list of input value names.

        Returns
        -------
        list of str
            List of input value parameter names.
        """
        return [
            Value.UPDATE_RATE,
            Value.NOTIFICATION_SUCCESS,
            Value.NOTIFICATION_INFO,
            Value.NOTIFICATION_ERROR
        ]

    def setInputIntValue(self, name: str, val: float):
        """
        Set an integer input value by name.

        Parameters
        ----------
        name : str
            Name of the scalar value to set.
        val : float
            Value passed to the code.

        Raises
        ------
        WrongArgument
            Raised if the scalar name is invalid.
        """
        if name == Value.UPDATE_RATE:
            self._up_rate = val
            return

        super().setInputIntValue(name=name, val=val)

    def setInputStringValue(self, name: str, val: str):
        """
        Set an string input value by name.

        Parameters
        ----------
        name : str
            Name of the scalar value to set.
        val : float
            Value passed to the code.

        Raises
        ------
        WrongArgument
            Raised if the scalar name is invalid.
        """
        if val == "":
            return

        if name == Value.NOTIFICATION_SUCCESS:
            self.layout.notifications.append(
                ("success", val, 5000)
            )
            return
        elif name == Value.NOTIFICATION_INFO:
            self.layout.notifications.append(
                ("info", val, 5000)
            )
            return
        elif name == Value.NOTIFICATION_ERROR:
            self.layout.notifications.append(
                ("error", val, 5000)
            )
            return

        super().setInputStringValue(name=name, val=val)

    def getInputMEDDoubleFieldTemplate(
        self, name: str
    ) -> medcoupling.MEDCouplingFieldDouble:
        """
        Retrieve an empty shell for an input field.

        This shell can be filled by the caller and then given to the code via
        setInputField(). The field has the MEDDoubleField format.

        The code uses this method to populate 'afield' with all the data that
        represents the context of the field (i.e., its support mesh, its
        discretization -- on nodes, on elements, ...). The remaining job for the
        caller is to fill the actual values of the field itself. When this is
        done, the field can be sent back to the code through setInputField().

        This method is not mandatory but is useful to know the mesh,
        discretization, etc. on which an input field is expected.

        See Problem documentation for more details on the time semantic of a
        field.

        Parameters
        ----------
        name : str
            Name of the field for which we would like the empty shell. The format
            should be "visualization_panel@field_name".

        Returns
        -------
        medcoupling.MEDCouplingFieldDouble
            Field object (in MEDDoubleField format) that will be populated with
            all the contextual information. Any previous information in this
            object will be discarded.

        Raises
        ------
        WrongContext
            Raised if called before initialize() or after terminate().
        WrongArgument
            Raised if the field name is invalid.
        """
        visualization_panel, field_name = self._split_name(name)

        slave = self.layout.get_panel(visualization_panel).get_slave()

        return slave.get_template(field_name)

    def setInputMEDDoubleField(
        self, name: str, afield: medcoupling.MEDCouplingFieldDouble
    ) -> None:
        """
        Provide the code with input data in the form of a MEDDoubleField.

        The method getInputFieldTemplate(), if implemented, may be used first
        to prepare an empty shell of the field to pass to the code.

        See Problem documentation for more details on the time semantic of a
        field.

        Parameters
        ----------
        name : str
            Name of the field that is given to the code. The format should be
            "visualization_panel@field_name".
        afield : medcoupling.MEDCouplingFieldDouble
            Field object (in MEDDoubleField format) containing the input data to
            be read by the code. The name of the field set on this instance (with
            the Field::setName() method) should not be checked. However its time
            value should ensure it is within the proper time interval ]t, t+dt].

        Returns
        -------
        None

        Raises
        ------
        WrongContext
            Raised if called before initialize() or after terminate().
        WrongArgument
            Raised if the field name ('name' parameter) is invalid.
        WrongArgument
            Raised if the time property of 'afield' does not belong to the
            currently computed time step ]t, t + dt].
        """

        visualization_panel, field_name = self._split_name(name)

        panel = self.layout.get_panel(visualization_panel)
        slave = panel.get_slave()

        #   The time is set before the field
        slave.set_time(self.time + (self._dt if self._dt else 0.0))
        if slave.update_policy == UpdatePolicy.UPDATE_DATA:
            return_val = slave.update_data(field_name, afield)
        elif slave.update_policy == UpdatePolicy.UPDATE_MESH:
            return_val = slave.update_mesh(field_name, afield)
        elif slave.update_policy == UpdatePolicy.APPEND_DATA:
            return_val = slave.append_data(field_name, afield)
        elif slave.update_policy == UpdatePolicy.APPEND_MESH:
            return_val = slave.append_mesh(field_name, afield)
        else:
            raise ValueError(f"Update policy {slave.update_policy} not implemented in LayoutProblem.")

        self.panels_to_recompute.append(visualization_panel)

        return return_val

    def setInputDoubleValue(self, name: str, val: float) -> None:
        """
        Provide the code with a scalar double data.

        See Problem documentation for more details on the time semantic of a
        scalar value.

        Parameters
        ----------
        name : str
            Name of the scalar value that is given to the code. The format should
            be "visualization_panel@field_name".
        val : float
            Value passed to the code.

        Returns
        -------
        None

        Raises
        ------
        WrongArgument
            Raised if the scalar name ('name' parameter) is invalid.
        WrongContext
            Raised if called before initialize() or after terminate().
        """

        visualization_panel, field_name = self._split_name(name)

        panel = self.layout.get_panel(visualization_panel)
        slave = panel.get_slave()

        #   The time is set before the field
        slave.set_time(self.time + (self._dt if self._dt else 0.0))
        if slave.update_policy == UpdatePolicy.UPDATE_DATA:
            return_val = slave.update_data(field_name, val)
        elif slave.update_policy == UpdatePolicy.UPDATE_MESH:
            return_val = slave.update_mesh(field_name, val)
        elif slave.update_policy == UpdatePolicy.APPEND_DATA:
            return_val = slave.append_data(field_name, val)
        elif slave.update_policy == UpdatePolicy.APPEND_MESH:
            return_val = slave.append_mesh(field_name, val)
        else:
            raise ValueError(f"Update policy {slave.update_policy} not implemented in LayoutProblem.")

        self.panels_to_recompute.append(visualization_panel)

        return return_val
