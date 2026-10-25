"""Code interface of the heat network example.

Implements :class:`~scivianna.interface.generic_interface.GraphInterface` on top
of the static :func:`build_graph` defined in ``heat_network.py``: the graph
structure is fixed, so a new frame only changes the cell values (the returned
``graph_updated`` flag stays False).
"""

from typing import Any, Dict, List, Tuple, Union

import multiprocessing as mp

from scivianna.data.data_graph import DataGraph
from scivianna.enums import VisualizationMode
from scivianna.interface.generic_interface import GraphInterface


class HeatNetworkInterface(GraphInterface):
    """GraphInterface of the heat network example (a static graph)."""

    def __init__(self):
        self._graph: DataGraph = None
        """Last computed graph (its structure does not change)"""

    def _build_graph(self) -> DataGraph:
        # Imported here to avoid a circular import with heat_network.py.
        from scivianna_example.graph.heat_network import build_graph

        return build_graph()

    def read_file(self, file_path: str, file_label: str) -> None:
        """The example has no input file to read.

        Parameters
        ----------
        file_path : str
            Ignored
        file_label : str
            Ignored
        """

    def get_labels(self) -> List[str]:
        """Returns a list of fields names displayable with this interface

        Returns
        -------
        List[str]
            List of fields names
        """
        return ["Temperature"]

    def get_label_coloring_mode(self, label: str) -> VisualizationMode:
        """Returns whether the given field is colored based on a string value or a float.

        Parameters
        ----------
        label : str
            Field to color name

        Returns
        -------
        VisualizationMode
            Coloring mode
        """
        return VisualizationMode.FROM_VALUE

    def compute_graph(
        self,
        u: Tuple[float, float, float],
        v: Tuple[float, float, float],
        origin: Tuple[float, float, float],
        q_tasks: mp.Queue,
        options: Dict[str, Any],
        caller: str = "API",
    ) -> Tuple[DataGraph, bool]:
        """Returns the heat network graph to display

        Parameters
        ----------
        u : Tuple[float, float, float]
            Horizontal coordinate director vector (ignored: the layout is fixed)
        v : Tuple[float, float, float]
            Vertical coordinate director vector (ignored: the layout is fixed)
        origin : Tuple[float, float, float]
            Physical 3D position of the slice center (ignored: the layout is fixed)
        q_tasks : mp.Queue
            Queue from which get orders from the master.
        options : Dict[str, Any]
            Additional options for frame computation.
        caller : str
            Identifier of the caller requesting the computation (default: "API")

        Returns
        -------
        Tuple[DataGraph, bool]
            The heat network graph, and whether the nodes were updated (always False)
        """
        self._graph = self._build_graph()
        return self._graph, False

    def get_graph_value_dict(
        self,
        value_label: str,
        cells: List[Union[int, str]],
        options: Dict[str, Any],
        caller: str = "API",
    ) -> Dict[Union[int, str], str]:
        """Returns a cell name - field value map for a given field name

        Parameters
        ----------
        value_label : str
            Field name to get values from
        cells : List[Union[int,str]]
            List of cells names
        options : Dict[str, Any]
            Additional options for frame computation.
        caller : str
            Identifier of the caller requesting the computation (default: "API")

        Returns
        -------
        Dict[Union[int,str], str]
            Field value for each requested cell names
        """
        if self._graph is None:
            self._graph = self._build_graph()

        values_per_cell = {}
        for data in self._graph.datas.values():
            for cell_id, value in zip(data.cell_ids, data.cell_values):
                values_per_cell[cell_id] = value

        return {cell: values_per_cell[cell] for cell in cells}

    def get_file_input_list(self) -> List[Tuple[str, str]]:
        """Returns a list of file label and its description for the GUI

        Returns
        -------
        List[Tuple[str, str]]
            Empty list: the example has no input file.
        """
        return []
