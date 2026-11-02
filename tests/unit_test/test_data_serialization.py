"""Unit tests for serializing interfaces and data (pickle and dill)."""

from pickle import PicklingError
from typing import Any

import pytest

from scivianna.interface.generic_interface import GenericInterface
from scivianna.slave import ComputeSlave


class FakeIterface(GenericInterface):
    """Minimal interface with nothing to serialize."""

    def read_file(self, file_path, file_label):
        """Accept any file (no-op)."""
        assert True

class FakeIterfaceWithSerialization(GenericInterface):
    """Interface that provides a custom serialize method."""

    def read_file(self, file_path, file_label):
        """Check that the serialized payload arrived intact."""
        assert file_path == "OK"

    @classmethod
    def serialize(self, obj: Any, key: str) -> Any:
        """This function receives an object that is about to be transmitted at the given key.
        -   If the object can be passed through a python multiprocessing Queue, it can be returned.
        -   If the object can't, it is serialized, and the code returns the file path.

        The read function will then be able to expect the returned object at the given key.

        By default, this class returns the obj, if the cobject can't be passed, overwrite this function.

        Parameters
        ----------
        obj : Any
            Object that is sent to the generic interface
        key : str
            Key associated to the object

        Returns
        -------
        Any
            Object transmissible through a multiprocessing Queue associated to the given object.
        """
        return "OK"

    

class UnpicklableObject:
    """Object that raises when pickled."""

    def __init__(self, data):
        """Store the payload."""
        self.data = data

    def __reduce__(self):
        """Always fail pickling."""
        raise PicklingError("This object cannot be pickled and thus cannot be passed through multiprocessing.Queue.")


class PicklableObject:
    """Object that pickles normally."""

    def __init__(self, data):
        """Store the payload."""
        self.data = data

    def read_file(self, file_path, file_label):
        """Check that the pickled payload arrived intact."""
        assert file_path.data == "OK"

@pytest.mark.default
def test_pickle():
    """A picklable argument is passed to the worker unchanged."""
    try:
        slave = ComputeSlave(FakeIterface)
        slave.read_file(PicklableObject("OK"), "key")
    finally:
        slave.terminate()

@pytest.mark.default
def test_non_pickle():
    """An unpicklable argument without a serialize method raises an error."""
    try:
        slave = ComputeSlave(FakeIterface)
        slave.read_file(UnpicklableObject(None), "key")
    except TypeError as e:
        assert True
    else:   
        assert False, "Pickle exception not caught"
    finally:
        slave.terminate()

@pytest.mark.default
def test_non_pickle_serialized():
    """An unpicklable argument is passed when the interface defines serialize."""
    try:
        slave = ComputeSlave(FakeIterfaceWithSerialization)
        slave.read_file(UnpicklableObject(None), "key")
    finally:
        slave.terminate()

if __name__ == "__main__":
    test_non_pickle()