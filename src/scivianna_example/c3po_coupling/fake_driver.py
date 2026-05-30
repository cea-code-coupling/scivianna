import os
from typing import Tuple

import medcoupling  # type: ignore
from icoco.exception import WrongContext, WrongArgument
from icoco.problem import Problem, ValueType


class DecreasingFieldProblem(Problem):
    def __init__(
        self, file_path: str
    ):
        if not os.path.isfile(file_path):
            raise ValueError(f"Provided file name does not exist {file_path}")

        self.meshnames = medcoupling.GetMeshNames(file_path)
        self.fieldnames = medcoupling.GetAllFieldNamesOnMesh(
            file_path, self.meshnames[0]
        )

        self.fields_iterations = {}

        for field in self.fieldnames:
            iterations = medcoupling.GetFieldIterations(
                medcoupling.ON_CELLS, file_path, self.meshnames[0], field
            )

            for iteration in iterations:
                self.fields_iterations[field] = [tuple(iteration)]

        self.mesh = medcoupling.ReadMeshFromFile(file_path, 0)

        value_label = list(self.fields_iterations.keys())[0]

        options = {
                    "Iteration": self.fields_iterations[value_label][0][0],
                    "Order": self.fields_iterations[value_label][0][1],
                }
        
        self.field: medcoupling.MEDCouplingFieldDouble = medcoupling.ReadField(
            medcoupling.ON_CELLS,
            file_path,
            self.meshnames[0],
            0,
            value_label,
            options["Iteration"],
            options["Order"],
        )

        self.time = 0.
        self._dt = None
        self._up_rate = 1
        self._up_skipped = 0

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
        pass

    def initialize(self) -> bool:
        """(Mandatory) Initialize the current problem instance.

        In this method the code should allocate all its internal structures and be ready to execute.
        File reads, memory allocations, and other operations likely to fail should be performed
        here, and not in the constructor (and not in the setDataFile() or in the setMPIComm()
        methods either).
        This method must be called only once (after a potential call to setMPIComm() and/or
        setDataFile()) and cannot be called again before terminate() has been performed.

        Returns
        -------
        bool
            true if all OK, otherwise false.

        Raises
        ------
        WrongContext
            exception if called multiple times or after initialize().
        """
        return True

    def terminate(self) -> None:
        """(Mandatory) Terminate the current problem instance and release all allocated resources.

        Terminate the computation, free the memory and save whatever needs to be saved.
        This method is called once at the end of the computation or after a non-recoverable error.
        No other ICoCo method except setDataFile(), setMPIComm() and initialize() may be called
        after this.

        Raises
        ------
        WrongContext
            exception if called before initialize() or after terminate().
            exception if called inside the TIME_STEP_DEFINED context (see Problem documentation).
        """
        self._dt = -1.0
        self.time = 0.0
        self._up_rate = 1
        self._up_skipped = 0

    def presentTime(self) -> float:
        """(Mandatory) Return the current time of the simulation.

        Can be called any time between initialize() and terminate().
        The current time can only change during a call to validateTimeStep() or to resetTime().

        Returns
        -------
        float
            the current (physical) time of the simulation

        Raises
        ------
        WrongContext
            exception if called before initialize() or after terminate().
        """
        return self.time

    def computeTimeStep(self) -> Tuple[float, bool]:
        """(Mandatory) Return the next preferred time step (time increment) for this code, and
        whether the code wants to stop.

        Both data are only indicative, the supervisor is not required to take them into account.
        This method is however marked as mandatory, since most of the coupling schemes expect the
        code to provide this information (those schemes then typically compute the minimum of the
        time steps of all the codes being coupled). Hence a possible implementation is to return a
        huge value, if a precise figure can not be computed.

        Can be called whenever the code is outside the TIME_STEP_DEFINED context (see Problem
        documentation).

        Returns
        -------
        Tuple[float, bool]
            - the preferred time step for this code (only valid if stop is false).
            - stop set to true if the code wants to stop. It can be used for example to indicate
              that, according to a certain criterion, the end of the transient computation is
              reached from the code point of view.

        Raises
        ------
        WrongContext
            exception if called before initialize() or after terminate().
            exception if called inside the TIME_STEP_DEFINED context (see Problem documentation).
        """
        return 1e8, False

    def initTimeStep(self, dt: float) -> bool:
        """(Mandatory) Provide the next time step (time increment) to be used by the code.

        After this call (if successful), the computation time step is defined to ]t, t + dt] where
        t is the value returned by presentTime(). The code enters the TIME_STEP_DEFINED context.

        A time step = 0.0 may be used when the stationaryMode is set to true for codes solving
        directly for the steady-state.

        Parameters
        ----------
        dt : float
            dt the time step to be used by the code

        Returns
        -------
        bool
            false means that given time step is not compatible with the code time scheme.

        Raises
        ------
        WrongContext
            exception if called before initialize() or after terminate().
            exception if called inside the TIME_STEP_DEFINED context (see Problem documentation).
            exception if called several times without resolution.
        WrongArgument
            exception if dt is invalid (dt < 0.0).
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
        """(Mandatory) Perform the computation on the current time interval.

        Can be called whenever the code is inside the TIME_STEP_DEFINED context
        (see Problem documentation).

        Returns
        -------
        bool
            true if computation was successful, false otherwise.

        Raises
        ------
        WrongContext
            exception if called before initialize() or after terminate().
            exception if called outside the TIME_STEP_DEFINED context (see Problem documentation).
            exception exception if called several times without a call to validateTimeStep() or to
            abortTimeStep().
        """
        self.field = self.field.__pow__(0.97)

        return True

    def validateTimeStep(self) -> None:
        """(Mandatory) Validate the computation performed by solveTimeStep.

        Can be called whenever the code is inside the TIME_STEP_DEFINED context (see Problem
        documentation).

        After this call:
        - the present time has been advanced to the end of the computation time step
        - the computation time step is undefined (the code leaves the TIME_STEP_DEFINED context).

        Raises
        ------
        WrongContext
            exception if called before initialize() or after terminate().
            exception if called outside the TIME_STEP_DEFINED context (see Problem documentation).
            exception if called before the solveTimeStep() method.
        """

        self.time += self._dt
        self._dt = None

        if self._up_skipped == self._up_rate:
            self._up_skipped = 0
        else:
            self._up_skipped += 1

    def abortTimeStep(self):
        self._dt = None

    def setStationaryMode(self, stationaryMode: bool) -> None:
        """(Mandatory) Set whether the code should compute a stationary solution or a transient one.

        New in version 2 of ICoCo. By default the code is assumed to be in stationary mode False
        (i.e. set up for a transient computation).
        If set to True, solveTimeStep() can be used either to solve a time step in view of an
        asymptotic solution, or to solve directly for the steady-state. In this last case, a time
        step = 0. can be used with initTimeStep() (whose call is always needed).
        The stationary mode status of the code can only be modified by this method (or by a call to
        terminate() followed by initialize()).

        Parameters
        ----------
        stationaryMode : bool
            true if the code should compute a stationary solution.

        Raises
        ------
        WrongContext
            called inside the TIME_STEP_DEFINED context (see Problem documentation).
            called before initialize() or after terminate().
        """
        self.stationary = stationaryMode

    def getStationaryMode(self) -> bool:
        """(Mandatory) Indicate whether the code should compute a stationary solution or a
        transient one.

        See also setStationaryMode().

        Can be called whenever the code is outside the TIME_STEP_DEFINED context
        (see Problem documentation).

        Returns
        -------
        bool
            true if the code has been set to compute a stationary solution.

        Raises
        ------
        WrongContext
            exception if called before initialize() or after terminate().
            exception if called inside the TIME_STEP_DEFINED context (see Problem documentation).
        """
        return self.stationary

    def getValueType(self, name: str) -> ValueType:

        super().getValueType(name=name)


    def getInputMEDDoubleFieldTemplate(
        self, name: str
    ) -> medcoupling.MEDCouplingFieldDouble:
        """(Optional) Retrieve an empty shell for an input field. This shell can be filled by the
        caller and then be given to the code via setInputField(). The field has the MEDDoubleField
        format.

        The code uses this method to populate 'afield' with all the data that represents the context
        of the field (i.e. its support mesh, its discretization -- on nodes, on elements, ...).
        The remaining job for the caller of this method is to fill the actual values of the field
        itself.
        When this is done the field can be sent back to the code through the method setInputField().
        This method is not mandatory but is useful to know the mesh, discretization... on which an
        input field is expected.

        See Problem documentation for more details on the time semantic of a field.

        Parameters
        ----------
        name : str
            name of the field for which we would like the empty shell

        Returns
        -------
        medcoupling.MEDCouplingFieldDouble
            field object (in MEDDoubleField format) that will be populated with all the contextual
            information.
            Any previous information in this object will be discarded.

        Raises
        ------
        WrongContext
            exception if called before initialize() or after terminate().
        WrongArgument
            exception if the field name is invalid.
        """
        mcfield = medcoupling.MEDCouplingFieldDouble(
            medcoupling.ON_CELLS, medcoupling.ONE_TIME
        )
        mcfield.setName(name)
        mcfield.setTime(0.0, 0, 0)
        mcfield.setMesh(self.mesh)
        array = medcoupling.DataArrayDouble([0.0] * self.mesh.getNumberOfCells())
        mcfield.setArray(array)
        mcfield.setNature(self.fields.getNature())
        return mcfield

    def getOutputDoubleValue(self, name):
        if name == "MAX":
            return self.field.getMaxValue()
        if name == "MIN":
            return self.field.getMinValue()
        return 0.
    
    def getOutputMEDDoubleField(self, name):
        if name == "VALUE":
            return self.field
        raise ValueError(f"Requested field {name}, only VALUE is available.")
    
    def getSolveStatus(self,):
        return True
    