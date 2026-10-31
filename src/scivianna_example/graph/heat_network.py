
from dataclasses import dataclass, field

import numpy as np
from scivianna.data.data2d import Data2D
from scivianna.data.data_graph import DataGraph, Link
from scivianna.utils.polygonize_tools import PolygonCoords, PolygonElement
from scivianna.panel.panel_graph import PanelGraph
from scivianna.slave import ComputeSlave
from scivianna_example.graph.heat_network_interface import HeatNetworkInterface

ROTATION_ORIGIN = (0.5, 1.0)
"""Rotation pivot, as fractions of the figure: the inlet, middle of the bottom edge."""


@dataclass
class Pipe:
    """A straight pipe along +y: the flow goes from its inlet (y = 0, the data origin)
    to its outlet (y = ``length``), before rotation.

    ``turn`` is the rotation of the pipe, in degrees clockwise.
    ``values`` holds one value per segment, from the inlet to the outlet.
    """

    name: str
    length: float = 8.0
    width: float = 1.0
    turn: float = 0.0
    values: np.ndarray = field(default_factory=lambda: np.arange(10, dtype=float))


def pipe_data(pipe: Pipe) -> Data2D:
    """A thin rectangle centred on x = 0, from y = 0 (inlet) to y = ``length``
    (outlet), split in one segment per value."""
    values = np.asarray(pipe.values, dtype=float)
    # n segments need n + 1 edges, the last one being the end of the pipe.
    edges = np.linspace(0.0, pipe.length, len(values) + 1)

    cells = []
    for k in range(len(values)):
        y0, y1 = edges[k], edges[k + 1]
        ext = PolygonCoords(
            [-pipe.width / 2, -pipe.width / 2, pipe.width / 2, pipe.width / 2],
            [y0, y1, y1, y0],
        )
        cells.append(PolygonElement(ext, [], f"{pipe.name}_seg_{k}"))

    data = Data2D.from_polygon_list(cells)
    data.cell_values = values
    return data

flame_polys = [
    [
        (397,83),(366,97),(329,123),(312,142),(297,169),
        (291,192),(291,215),(295,233),(315,283),(315,303),
        (310,316),(299,328),(291,331),(280,331),(266,325),
        (258,318),(250,304),(245,270),(226,293),(205,327),
        (189,367),(180,412),(180,446),(188,489),(203,525),
        (223,554),(245,575),(279,596),(309,607),(353,615),
        (416,613),(451,604),(490,586),(512,570),(538,541),
        (551,516),(559,482),(559,453),(550,418),(537,394),
        (518,374),(502,364),(495,363),(499,375),(499,392),
        (496,400),(488,407),(471,405),(460,392),(458,378),
        (461,365),(487,313),(488,272),(476,239),(464,222),
        (449,207),(409,178),(388,152),(383,136),(383,112),
        (386,101)
    ],
    [
        (372,261),(358,279),(346,310),(344,380),
        (336,410),(320,429),(304,436),(282,432),
        (271,421),(268,410),(251,451),(247,475),
        (247,501),(251,523),(262,550),(276,571),
        (295,590),(327,607),(326,611),(384,616),
        (399,615),(398,611),(407,614),(402,614),
        (401,611),(442,593),(474,565),(486,548),
        (496,525),(500,485),(490,443),(485,460),
        (471,475),(454,478),(440,472),(432,456),
        (432,428),(426,400),(413,375),(387,344),
        (370,311),(366,280)
    ],
    [
        (372,416),(353,435),(335,469),(330,504),
        (336,529),(323,528),(312,522),(296,503),
        (293,537),(301,568),(321,597),(343,610),
        (344,614),(384,616),(385,613),(409,604),
        (424,589),(434,570),(436,541),(423,512),
        (376,459),(369,440)
    ]
]

FLAME_LOGO = {
    "0": {
        "outer": flame_polys[0],
        "holes": [
            flame_polys[1]
        ]
    },
    "1": {
        "outer": flame_polys[1],
        "holes": [
            flame_polys[2]
        ]
    },
    "2": {
        "outer": flame_polys[2],
        "holes": []
    }
}
polys = [
    [
        (107,682),(146,682),(146,482),(309,365),
        (309,475),(463,365),(463,475),(617,365),
        (617,557),(660,557),(684,162),(752,162),
        (776,557),(834,557),(834,682),(873,682),
        (873,722),(107,722)
    ],
    [
        (195,554),(260,554),(260,614),(195,614)
    ],
    [
        (349,554),(413,554),(413,614),(349,614)
    ],
    [
        (503,554),(568,554),(568,614),(503,614)
    ]
]

FACTORY_LOGO = {
    "0": {
        "outer": polys[0],
        "holes": [
            polys[1],
            polys[2],
            polys[3]
        ]
    }
}

LOGO_SCALE = 1/1000

def make_generator(name: str):
    data = Data2D.from_polygon_list(
        [
            PolygonElement(
                exterior_polygon = PolygonCoords(
                    x_coords=[e[0] * LOGO_SCALE for e in FLAME_LOGO[color_name]["outer"]],
                    y_coords=[-e[1] * LOGO_SCALE for e in FLAME_LOGO[color_name]["outer"]]
                ),
                holes=[
                    PolygonCoords(
                        x_coords=[e[0] * LOGO_SCALE for e in hole],
                        y_coords=[-e[1] * LOGO_SCALE for e in hole]
                    )
                    for hole in FLAME_LOGO[color_name]["holes"]
                ],
                cell_id=f'{name}_{color_name}'
            )
            for color_name in ["0", "1", "2"]
        ]
    )
    data.cell_values = np.array([140, 165, 190])
    return data

def make_factory(name: str, value: float):
    data = Data2D.from_polygon_list(
        [
            PolygonElement(
                exterior_polygon = PolygonCoords(
                    x_coords=[e[0] * LOGO_SCALE for e in color_dict["outer"]],
                    y_coords=[-e[1] * LOGO_SCALE for e in color_dict["outer"]]
                ),
                holes=[
                    PolygonCoords(
                        x_coords=[e[0] * LOGO_SCALE for e in hole],
                        y_coords=[-e[1] * LOGO_SCALE for e in hole]
                    )
                    for hole in color_dict["holes"]
                ],
                cell_id=f'{name}_{color_name}'
            )
            for color_name, color_dict in FACTORY_LOGO.items()
        ]
    )
    data.cell_values = np.array([value])
    return data

def build_graph():
    generator = make_generator("generator")
    
    pipe_1 = pipe_data(Pipe("pipe_1", 3., .15, 90, np.linspace(190, 170, 10)))
    pipe_2 = pipe_data(Pipe("pipe_2", 3., .15, 90, np.linspace(170, 150, 10)))
    pipe_3 = pipe_data(Pipe("pipe_3", 3., .15, 90, np.linspace(170, 150, 10)))
    pipe_4 = pipe_data(Pipe("pipe_4", 3., .15, 90, np.linspace(150, 140, 10)))
    
    factory_1 = make_factory("factory_1", 150)
    factory_2 = make_factory("factory_2", 140)

    datas = {
        "generator": generator,
        "pipe_1": pipe_1,
        "pipe_2": pipe_2,
        "pipe_3": pipe_3,
        "pipe_4": pipe_4,
        "factory_1": factory_1,
        "factory_2": factory_2,
    }
    locations = {
        "generator": (0, -130),
        "pipe_1": (150, -170),
        "pipe_2": (570, -250),
        "pipe_3": (650, -170),
        "pipe_4": (1100, -250),
        "factory_1": (570, -700),
        "factory_2": (1100, -700),
    }
    labels = {
        "generator": "Generator",
        "pipe_1": "",
        "pipe_2": "",
        "pipe_3": "",
        "pipe_4": "",
        "factory_1": "Factory 1",
        "factory_2": "Factory 2",
    }
    rotations = {
        "generator": 0.,        
        "pipe_1": 90,
        "pipe_2": 0,
        "pipe_3": 90,
        "pipe_4": 0,
        "factory_1": 0,
        "factory_2": 0,
    }
    links = [
        Link(source_data="generator", target_data="pipe_1", source_direction="right", target_direction="bottom"),

        Link(source_data="pipe_1", target_data="pipe_2", source_direction="top", target_direction="bottom"),
        Link(source_data="pipe_1", target_data="pipe_3", source_direction="top", target_direction="bottom"),
        Link(source_data="pipe_3", target_data="pipe_4", source_direction="top", target_direction="bottom"),
        
        Link(source_data="pipe_2", target_data="factory_1", source_direction="top", target_direction="bottom"),
        Link(source_data="pipe_4", target_data="factory_2", source_direction="top", target_direction="bottom"),
    ]

    return DataGraph(datas, locations, links, labels=labels, rotations=rotations)

def make_graph_panel(**plotter_kwargs) -> PanelGraph:
    """Builds the graph panel of the example.

    The graph is requested to a :class:`ComputeSlave` running the
    :class:`HeatNetworkInterface`, in the same way :class:`Panel2D` requests its
    frames to a :class:`Geometry2D` interface.

    Parameters
    ----------
    **plotter_kwargs
        Forwarded to :class:`ReactFlowPlotter` (e.g. ``max_size``, ``scale``,
        ``color_map_name``). ``rotation_origin`` defaults to :data:`ROTATION_ORIGIN`.

    Returns
    -------
    PanelGraph
        The graph panel displaying the example DataGraph.
    """
    plotter_kwargs.setdefault("rotation_origin", ROTATION_ORIGIN)
    slave = ComputeSlave(HeatNetworkInterface)
    return PanelGraph(name="Graph example", slave=slave, **plotter_kwargs)


if __name__ == "__main__":
    from scivianna.notebook_tools import _serve_panel
    _serve_panel(make_graph_panel())
