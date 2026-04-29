
from typing import List, Tuple, Union

import pytest
from scivianna.interface.generic_interface import ValueAtLocation
from scivianna.slave import ComputeSlave


class ThrowingErrorInterface(ValueAtLocation):
    def __init__(self, ):
        """Interface built to test error catching.
        """
        pass

    def get_value(
        self,
        position: Tuple[float, float, float],
        cell_index: str,
        material_name: str,
        field: str,
    ):
        """Provides the result value of a field from either the (x, y, z) position, the cell index, or the material name.

        Parameters
        ----------
        position : Tuple[float, float, float]
            Position at which the value is requested
        cell_index : str
            Index of the requested cell
        material_name : str
            Name of the requested material
        field : str
            Requested field name

        Returns
        -------
        List[Union[str, float]]
            Field value
        """
        raise ValueError("Testing valueError")

    def get_values(
        self,
        positions: List[Tuple[float, float, float]],
        cell_indexes: List[str],
        material_names: List[str],
        field: str,
    ) -> List[Union[str, float]]:
        """Provides the result values at different positions from either the (x, y, z) positions, the cell indexes, or the material names.

        Parameters
        ----------
        positions : List[Tuple[float, float, float]]
            List of position at which the value is requested
        cell_indexes : List[str]
            Indexes of the requested cells
        material_names : List[str]
            Names of the requested materials
        field : str
            Requested field name

        Returns
        -------
        List[Union[str, float]]
            Field values
        """
        raise NotImplementedError("Testing NotImplementedError")

    def get_labels(self) -> List[str]:
        """Returns the fields names providable.

        Returns
        -------
        List[str]
            Fields names
        """
        raise RuntimeError("Testing RuntimeError")


@pytest.mark.default
def test_runtime_error_allowed():
    slave = ComputeSlave(ThrowingErrorInterface, allow_errors=True)
    try:
        val = slave.get_labels()
    except RuntimeError as e:
        assert False, "RuntimeError not caught"
    else:
        assert val is None

@pytest.mark.default
def test_value_error_allowed():
    slave = ComputeSlave(ThrowingErrorInterface, allow_errors=True)
    try:
        val = slave.get_value(None, None, None, None)
    except ValueError as e:
        assert False, "ValueError not caught"
    else:
        assert val is None

@pytest.mark.default
def test_notimplemented_error_allowed():
    slave = ComputeSlave(ThrowingErrorInterface, allow_errors=True)
    try:
        val = slave.get_values(None, None, None, None)
    except NotImplementedError as e:
        assert False, "NotImplementedError not caught"
    else:
        assert val is None


@pytest.mark.default
def test_runtime_error():
    slave = ComputeSlave(ThrowingErrorInterface)
    try:
        val = slave.get_labels()
    except RuntimeError as e:
        assert True, "RuntimeError caught"
    else:
        assert False, "RuntimeError caught"

@pytest.mark.default
def test_value_error():
    slave = ComputeSlave(ThrowingErrorInterface)
    try:
        val = slave.get_value(None, None, None, None)
    except ValueError as e:
        assert True, "ValueError caught"
    else:
        assert False, "ValueError caught"

@pytest.mark.default
def test_notimplemented_error():
    slave = ComputeSlave(ThrowingErrorInterface)
    try:
        val = slave.get_values(None, None, None, None)
    except NotImplementedError as e:
        assert True, "NotImplementedError caught"
    else:
        assert False, "NotImplementedError caught"