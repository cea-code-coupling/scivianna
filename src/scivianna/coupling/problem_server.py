"""
This module contains a functional client-server implementation of ICoCo ``Problem``.

Warning
-------
    This is an experimental module.

"""

from multiprocessing.managers import BaseManager
from typing import Type

from icoco.utils import ICoCoMethods
from icoco.problem import Problem


class ServerManager(BaseManager):
    """Class to register remote ``Problem``."""

    @classmethod
    def register(cls, class_type: Type, *args, **kwargs) -> str:  # pylint: disable=arguments-differ
        """Register a remote class.

        Parameters
        ----------
        class_type : Type
            Class to use on server side.

        Returns
        -------
        str
            typeid to use as the ``ProblemClient`` argument.

        Raises
        ------
        ValueError
            if class is already registerd
        """
        typeid = class_type.__name__
        if typeid in cls._registry:
            raise ValueError(f"typeid {typeid} is already registerd.")
        super().register(typeid, class_type, *args, **kwargs)
        return typeid


class RemoteException(Exception):
    """Exception raised when remote process fails."""


def _method(self, method_name, *args, **kwargs):
    try:
        # print(f"remote calls: '{method_name}'", flush=True)
        # pylint: disable=protected-access
        return getattr(self._problem, method_name)(*args, **kwargs)
    except Exception as error:
        import traceback
        tb_str = traceback.format_exc()

        # Construct a detailed error message
        error_msg = (
            f"RemoteException raised while calling method '{method_name}'\n"
            f"Args: {args}\n"
            f"Kwargs: {kwargs}\n"
            f"Traceback:\n\n{tb_str}\n"
        )

        # Raise with original exception preserved (chaining)
        raise RemoteException(error_msg) from error


def redirect_icoco_to_server(cls):
    """Redirect ICoCo methods to server.
    """
    def create_icoco_method(method_name):
        return lambda self, *args, **kwargs: _method(self, method_name, *args, **kwargs)
    for name in ICoCoMethods.ALL:
        setattr(cls, name, create_icoco_method(name))
    if not hasattr(cls, "__abstractmethods__"):
        raise AttributeError(
            "Class is expected to have '__abstractmethods__' attribute.")  # pragma: no cover
    cls.__abstractmethods__ = frozenset()
    return cls


@redirect_icoco_to_server
class ProblemClient(Problem):
    """Server-Client implementation of ICoCo problem."""

    # ******************************************************
    # section Problem
    # ******************************************************
    def __init__(self, typeid: str, *args, **kwargs) -> None:
        """Constructor.

        Notes
        -----
            Internal set up and initialization of the code should not be done here,
            but rather in initialize() method.

        Parameters
        ----------
        typeid : str
            one of the typeid provided to the register method of ``ServerManager`` class.

        The other parameters will be passed to the callable associated to the typeid.
        """
        self._server: Problem = typeid
        self._args = args
        self._kwargs = kwargs

        self._manager: ServerManager = None
        super().__init__()

    def __del__(self):
        if self._manager is not None:
            self._manager.shutdown()

    @property
    def _problem(self) -> Problem:
        if self._manager is None:
            self._manager = ServerManager()
            self._manager.start()  # pylint: disable=consider-using-with
            self._server = getattr(self._manager, self._server)(*self._args, **self._kwargs)
            # print("type(self._server)", type(self._server))
        return self._server
