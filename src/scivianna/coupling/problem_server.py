"""
This module contains a functional client-server implementation of ICoCo ``Problem``.
It helps moving scivianna on another process to prevent it to slow down the ongoing simulation.

Warning
-------
    This is an experimental module.

"""

from multiprocessing.managers import BaseManager
from typing import Type

from icoco.utils import ICoCoMethods
from icoco.problem import Problem


class ServerManager(BaseManager):
    """
    Manager class to register remote ICoCo ``Problem`` implementations.

    This class extends Python's BaseManager to allow registration of
    Problem classes that will be executed on a remote server process,
    enabling distributed visualization.

    See Also
    --------
    ProblemClient : Client-side proxy for remote problems.
    """

    @classmethod
    def register(cls, class_type: Type, *args, **kwargs) -> str:  # pylint: disable=arguments-differ
        """
        Register a remote class for distributed execution.

        Parameters
        ----------
        class_type : Type
            The Problem class to use on the server side.
        *args : tuple
            Additional positional arguments passed to the server class constructor.
        **kwargs : dict
            Additional keyword arguments passed to the server class constructor.

        Returns
        -------
        str
            The typeid string to use as the ``ProblemClient`` argument.

        Raises
        ------
        ValueError
            Raised if the class typeid is already registered.
        """
        typeid = class_type.__name__
        if typeid in cls._registry:
            raise ValueError(f"typeid {typeid} is already registerd.")
        super().register(typeid, class_type, *args, **kwargs)
        return typeid


class RemoteException(Exception):
    """
    Exception raised when a remote process fails.

    This exception wraps the traceback and error details from the remote
    server process to help diagnose failures in distributed execution.

    See Also
    --------
    ProblemClient : Uses this exception to communicate remote errors.
    """


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
    """
    Decorator that redirects ICoCo methods to the remote server.

    This decorator dynamically assigns all ICoCo protocol methods to the
    decorated class, routing each call through the ``_method`` helper to
    the underlying remote problem instance. It also removes any abstract
    method restrictions from the class.

    Parameters
    ----------
    cls : type
        The class to decorate. Must have an ``__abstractmethods__`` attribute.

    Returns
    -------
    type
        The decorated class with ICoCo methods redirected to server.

    Raises
    ------
    AttributeError
        Raised if the class does not have an ``__abstractmethods__`` attribute.

    See Also
    --------
    _method : The helper function that performs the actual method redirection.
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
    """
    Server-Client implementation of ICoCo Problem for distributed execution.

    This class acts as a proxy that delegates all ICoCo method calls to a
    remote server process running the actual Problem implementation. It enables
    visualization without slowing down the main simulation.

    Examples
    --------
    >>> typeid = ServerManager.register(MyProblem)
    >>> client = ProblemClient(typeid=typeid, arg1=val1)
    >>> client.initialize()

    See Also
    --------
    ServerManager : Registers and starts the remote problem server.
    """

    # ******************************************************
    # section Problem
    # ******************************************************
    def __init__(self, typeid: str, *args, **kwargs) -> None:
        """
        Initialize the ProblemClient proxy.

        Notes
        -----
            Internal set up and initialization of the code should not be done here,
            but rather in the ``initialize()`` method on the server side.

        Parameters
        ----------
        typeid : str
            One of the typeid strings provided by the ``ServerManager.register()``
            method.
        *args : tuple
            Positional arguments passed to the remote server class constructor.
        **kwargs : dict
            Keyword arguments passed to the remote server class constructor.
        """
        self._server: Problem = typeid
        self._args = args
        self._kwargs = kwargs

        self._manager: ServerManager = None
        super().__init__()

    def __del__(self):
        """
        Cleanup: shutdown the remote server manager when this client is garbage collected.
        """
        if self._manager is not None:
            self._manager.shutdown()

    @property
    def _problem(self) -> Problem:
        """
        Lazy-initialized property that returns the remote server instance.

        The server is started on first access and cached for subsequent calls.

        Returns
        -------
        Problem
            The remote Problem instance running on the server process.
        """
        if self._manager is None:
            self._manager = ServerManager()
            self._manager.start()  # pylint: disable=consider-using-with
            self._server = getattr(self._manager, self._server)(*self._args, **self._kwargs)
        return self._server
