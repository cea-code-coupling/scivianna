
from typing import Literal

from scivianna.data.data_container import DataContainer
from scivianna.data.data2d import Data2D
from scivianna.enums import DataType

from pydantic import BaseModel

class Link(BaseModel):
    """Directed connection between two nodes of a :class:`DataGraph`.

    A link references its endpoints by their node key in the graph's
    ``datas``/``locations`` dictionaries, and specifies on which side of each
    node the connection attaches. The directions are used to route the
    connector line around the panels when the graph is laid out.

    Attributes
    ----------
    source_data : str
        Key of the source node in the graph.
    target_data : str
        Key of the target node in the graph.
    source_direction : {'top', 'bottom', 'left', 'right'}
        Side of the source node where the link attaches.
    target_direction : {'top', 'bottom', 'left', 'right'}
        Side of the target node where the link attaches.
    """

    source_data: str
    target_data: str

    source_direction: Literal["top", "bottom", "left", "right"]
    target_direction: Literal["top", "bottom", "left", "right"]


class DataGraph(DataContainer):
    """
    Container for a graph of 2D geometry data with field values and colors.

    A ``DataGraph`` models a set of 2D plots (nodes) connected by directed
    links, as used to visualize the topology of coupled simulations: each
    node holds a :class:`Data2D` payload (polygons + field values), a 2D
    position in the layout, and edges describe how data flows between nodes.

    Attributes
    ----------
    datas : dict[str, Data2D]
        Mapping of node key to its :class:`Data2D` content. Every entry must
        use the polygon representation (``DataType.POLYGONS``).
    locations : dict[str, tuple[float, float]]
        Mapping of node key to its ``(x, y)`` position in the graph layout.
        Must share exactly the same keys as ``datas``.
    links : list[Link]
        Directed connections between nodes; both endpoints must exist in
        ``datas``.
    """

    def __init__(self, datas: dict[str, Data2D], locations: dict[str, tuple[float, float]], links: list[Link]) -> None:
        """Initializer of DataGraph

        Parameters
        ----------
        datas : dict[str, Data2D]
            Dictionary of Data2D per node (polygons only)
        locations : dict[str, tuple[float, float]]
            Dictionary of node layout positions, same keys as ``datas``
        links : list[Link]
            Directed links between nodes

        Raises
        ------
        AssertionError
            If a data entry is not polygon-based, if ``datas`` and
            ``locations`` do not share the same keys, or if a link references
            an unknown node.
        """
        for key, data in datas.items():
            assert data.data_type == DataType.POLYGONS, f"Datagraph expects polygons, found {data.data_type} in {key}"

        assert isinstance(datas, dict), f"datas must be a dictionnary, found {type(datas)}"
        assert isinstance(locations, dict), f"locations must be a dictionnary, found {type(locations)}"
        assert set(datas.keys()) == set(locations.keys()), f"datas and locations must have the same keys, found the following mismatch {set(datas.keys()) - set(locations.keys())}"

        for link in links:
            assert link.source_data in datas, f"Link source {link.source_data} not found in datas"
            assert link.target_data in datas, f"Link target {link.target_data} not found in datas"

        self.locations = locations
        self.links = links
        self.datas = datas
