"""Graph example: a hierarchical pipe network (series / parallel / elbows) displayed
as an interactive ReactFlow graph.

The network is described by a small hierarchical model (``Pipe``, ``Series``,
``Parallel``). ``build_graph`` lays it out and produces the flat data a
``DataGraph`` needs: the Data2D of every pipe, its location and rotation, and the
links between pipes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

import numpy as np

from scivianna.data.data2d import Data2D
from scivianna.data.data_graph import DataGraph, Link
from scivianna.utils.polygonize_tools import PolygonCoords, PolygonElement
from scivianna_example.graph.graph import Parallel, Series
# --------------------------------------------------------------------------- #
# Layout constants (pixels). PX_PER_UNIT converts pipe lengths to the pixel size
# of the node; it must match the size at which the figures are actually drawn.
# --------------------------------------------------------------------------- #
PX_PER_UNIT = 40.0
GAP_ALONG = 90.0  # empty space between consecutive elements, along the flow
GAP_ACROSS = 50.0  # empty space between parallel lanes

def parse_data(path):
    """Read the last column of an evolution file as a list of floats."""
    arr = []
    with open(path, "r") as f:
        lines = f.readlines()[5:]
        for l in lines:
            arr.append(float(l.split()[-1]))
    return arr

data_paths = [
    f"/partage/spatial/Theses/tmoulignier/SNPS-suite/apps/ntp_modeler/src/ntp_modeler/workdir/conjonction_scenario_depletion/step_1/c3_iter_0/fuel_element/workdir/TK_H2C_{i}.evol"
    for i in range(10)
]
data = [
    parse_data(d) for d in data_paths
]
inlet_value = np.min(np.array(data))
outlet_value = np.max(np.array(data))

data = np.load("/volatile/catA/tmoulignier/Workspace/graph_plot/scivianna/src/scivianna_example/graph/axial_power_matrices_10x50.npz")["axial_power"]
inlet_value = 0
outlet_value = 0

# --------------------------------------------------------------------------- #
# Hierarchical (fake) data model
# --------------------------------------------------------------------------- #
@dataclass
class Pipe:
    """A straight pipe. The flow goes from its left end (inlet) to its right end
    (outlet) before rotation.

    ``turn`` is relative to the incoming flow direction: ``0`` goes straight,
    ``+90`` turns clockwise (right-hand turn on screen), ``-90`` counter-clockwise.
    """

    name: str
    length: float = 8.0
    height: float = 1.0
    turn: int = 0
    values: float = np.arange(10)


Element = Union[Pipe, Series, Parallel]


def make_network() -> Series:
    """A loop: inlet -> three parallel branches (one of them made of two parallel
    pipes, another one with elbows) -> collector -> outlet, with direction changes."""
    return Series(
        [
            Pipe("Core inlet", length=.3, height=.1, values = [inlet_value]),
            Parallel(
                [
                    Pipe(f"Ring {i}", length=.1, height=.93, values = data[i])
                    for i in range(len(data))
                ]
            ),
            Pipe("Core outlet", length=.3, height=.1, values = [outlet_value]),
        ]
    )


# --------------------------------------------------------------------------- #
# Layout: hierarchy -> placed boxes
# --------------------------------------------------------------------------- #
Vec = Tuple[float, float]


def _add(a: Vec, b: Vec, k: float = 1.0) -> Vec:
    """Vector sum a + k*b."""
    return (a[0] + k * b[0], a[1] + k * b[1])


def _dot(a: Vec, b: Vec) -> float:
    """Dot product of two 2D vectors."""
    return a[0] * b[0] + a[1] * b[1]


def _cw(d: Vec) -> Vec:
    """Direction rotated 90 deg clockwise on screen (y points down)."""
    return (-d[1], d[0])


def _turned(d: Vec, turn: int) -> Vec:
    """Rotate a direction by a multiple of 90 degrees (clockwise on screen)."""
    for _ in range((turn // 90) % 4):
        d = _cw(d)
    return d


_ANGLE = {(1, 0): 0.0, (0, 1): 90.0, (-1, 0): 180.0, (0, -1): 270.0}


@dataclass
class Box:
    """Bounding box of a (possibly rotated) node: top-left position and size."""

    x: float
    y: float
    w: float
    h: float
    rotation: float

    def corners(self) -> List[Vec]:
        """The four unrotated corners of the box."""
        return [(self.x, self.y), (self.x + self.w, self.y), (self.x, self.y + self.h), (self.x + self.w, self.y + self.h)]


@dataclass
class Block:
    """A laid-out element."""

    boxes: Dict[str, Box]
    entries: List[str]  # pipes fed by the upstream element
    exits: List[str]  # pipes feeding the downstream element
    exit_point: Vec  # centre of the outlet face, where the next element starts
    out_dir: Vec  # flow direction on leaving the block
    pipes: Dict[str, Pipe]
    inner_links: List[Tuple[str, str]]

    def shifted(self, delta: Vec) -> "Block":
        """Return a copy of the block translated by delta."""
        boxes = {n: Box(b.x + delta[0], b.y + delta[1], b.w, b.h, b.rotation) for n, b in self.boxes.items()}
        return Block(boxes, self.entries, self.exits, _add(self.exit_point, delta), self.out_dir, self.pipes, self.inner_links)

    def extent(self, axis: Vec) -> Tuple[float, float]:
        """Return the (min, max) projection of the block onto an axis."""
        values = [_dot(c, axis) for b in self.boxes.values() for c in b.corners()]
        return min(values), max(values)


def _layout_pipe(pipe: Pipe, origin: Vec, direction: Vec) -> Block:
    """Places a pipe whose inlet face is centred on ``origin`` (flow along ``direction``)."""
    length, thick = pipe.length * PX_PER_UNIT, pipe.height * PX_PER_UNIT
    if pipe.turn:
        # Elbow: leave room for the pipe's own thickness so it does not overlap the
        # previous element, then flow in the new direction.
        origin = _add(origin, direction, thick / 2)
        direction = _turned(direction, pipe.turn)
    centre = _add(origin, direction, length / 2)
    horizontal = direction[1] == 0
    w, h = (length, thick) if horizontal else (thick, length)
    box = Box(centre[0] - w / 2, centre[1] - h / 2, w, h, _ANGLE[direction])
    return Block({pipe.name: box}, [pipe.name], [pipe.name], _add(origin, direction, length), direction, {pipe.name: pipe}, [])


def _layout_series(series: Series, origin: Vec, direction: Vec) -> Block:
    """Lay out a series of elements one after another along the flow."""
    blocks: List[Block] = []
    cursor, current = origin, direction
    for item in series.items:
        block = _layout(item, cursor, current)
        if blocks:  # connect every exit of the previous block to every entry of this one
            previous = blocks[-1]
            block.inner_links.extend((a, b) for a in previous.exits for b in block.entries)
        blocks.append(block)
        cursor = _add(block.exit_point, block.out_dir, GAP_ALONG)
        current = block.out_dir
    boxes, pipes, links = {}, {}, []
    for b in blocks:
        boxes.update(b.boxes)
        pipes.update(b.pipes)
        links.extend(b.inner_links)
    last = blocks[-1]
    return Block(boxes, blocks[0].entries, last.exits, last.exit_point, last.out_dir, pipes, links)


def _layout_parallel(parallel: Parallel, origin: Vec, direction: Vec) -> Block:
    """Lay out parallel branches side by side across the flow."""
    lanes = [_layout(branch, origin, direction) for branch in parallel.branches]
    out_dirs = {lane.out_dir for lane in lanes}
    if len(out_dirs) != 1:
        raise ValueError("All branches of a Parallel must leave in the same direction.")
    out_dir = out_dirs.pop()

    # Stack the lanes across the flow, centred on the incoming flow line.
    across = _cw(direction)
    base = _dot(origin, across)
    spans = [(lo - base, hi - base) for lo, hi in (lane.extent(across) for lane in lanes)]
    total = sum(hi - lo for lo, hi in spans) + GAP_ACROSS * (len(lanes) - 1)
    cursor = -total / 2
    placed = []
    for lane, (lo, hi) in zip(lanes, spans):
        placed.append(lane.shifted(_add((0.0, 0.0), across, cursor - lo)))
        cursor += (hi - lo) + GAP_ACROSS

    boxes, pipes, links, entries, exits = {}, {}, [], [], []
    for lane in placed:
        boxes.update(lane.boxes)
        pipes.update(lane.pipes)
        links.extend(lane.inner_links)
        entries.extend(lane.entries)
        exits.extend(lane.exits)

    # The next element starts past the furthest edge, on the middle of the block.
    union = Block(boxes, [], [], origin, out_dir, {}, [])
    _, far = union.extent(out_dir)
    cross = _cw(out_dir)
    lo, hi = union.extent(cross)
    exit_point = _add(_add((0.0, 0.0), out_dir, far), cross, (lo + hi) / 2)
    return Block(boxes, entries, exits, exit_point, out_dir, pipes, links)


def _layout(element: Element, origin: Vec, direction: Vec) -> Block:
    """Dispatch the layout of a pipe, series or parallel element."""
    if isinstance(element, Pipe):
        return _layout_pipe(element, origin, direction)
    if isinstance(element, Series):
        return _layout_series(element, origin, direction)
    if isinstance(element, Parallel):
        return _layout_parallel(element, origin, direction)
    raise TypeError(f"Unsupported element: {element!r}")


# --------------------------------------------------------------------------- #
# Data of one pipe
# --------------------------------------------------------------------------- #
def pipe_data(pipe: Pipe) -> Data2D:
    """A thin rectangle of the pipe's size, split in segments whose values go from
    the inlet value to the outlet value."""
    cells = []
    y_values = np.arange(0, pipe.height, pipe.height/(len(pipe.values) + 1))
    for k in range(len(pipe.values)):
        x0 = y_values[k]
        x1 = y_values[k+1]
        ext = PolygonCoords(
            [0, 0, pipe.length, pipe.length],
            [x0, x1, x1, x0], 
        )
        cells.append(PolygonElement(ext, [], f"{pipe.name}_seg_{k}"))
    data = Data2D.from_polygon_list(cells)
    data.cell_values = np.array(pipe.values)
    return data


# --------------------------------------------------------------------------- #
# Hierarchy -> DataGraph
# --------------------------------------------------------------------------- #
def build_graph(network: Element) -> DataGraph:
    """Lays the network out (flow starts to the right at the origin) and builds the
    data set of the graph from it."""
    block = _layout(network, (0.0, 0.0), (1, 0))

    datas = {name: pipe_data(pipe) for name, pipe in block.pipes.items()}
    locations = {
        "Core inlet": (0, 0),
        "Core outlet": (0, -520),
        **{
            f"Ring {i}": (i*50, -400)
            for i in range(len(data))
        }
    }
    rotations = {name: box.rotation for name, box in block.boxes.items() if box.rotation}
    labels = {
        name: f"{name}"# ({pipe.length:g}x{pipe.height:g}"
        # + (f", {pipe.turn:+d} deg elbow" if pipe.turn else "")
        # + ")"
        for name, pipe in block.pipes.items()
    }
    # Directions are expressed in each node's own frame: the flow always enters a
    # pipe through its left plug and leaves through its right plug, whatever the
    # rotation of the node on the canvas.
    links = [
        Link(source_data=a, target_data=b, source_direction="top", target_direction="bottom")
        for a, b in block.inner_links
    ]
    return DataGraph(datas, locations, links, labels=labels, rotations=rotations)


def make_graph() -> DataGraph:
    """Builds the example graph."""
    return build_graph(make_network())