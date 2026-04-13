import matplotlib.pyplot as plt
import matplotlib.axes
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb
import numpy as np

from typing import Dict, Tuple, Any, Union

from scivianna.interface.generic_interface import Geometry2D
from scivianna.slave import ComputeSlave
from scivianna.extension.field_selector import set_colors_list
from scivianna.plotter_2d.polygon.matplotlib import Matplotlib2DPolygonPlotter
from scivianna.plotter_2d.grid.matplotlib import Matplotlib2DGridPlotter
from scivianna.utils.color_tools import get_edges_colors
from scivianna.utils.polygon_sorter import PolygonSorter
from scivianna.enums import VisualizationMode
from scivianna.constants import X, Y


def plot_frame_in_axes(
    slave: Union[ComputeSlave, Geometry2D],
    coloring_label: str,
    axes: matplotlib.axes.Axes,
    u: Tuple[float, float, float] = X,
    v: Tuple[float, float, float] = Y,
    u_min: float = 0.,
    u_max: float = 1.,
    v_min: float = 0.,
    v_max: float = 1.,
    w_value: float = 0.0,
    color_map: str = "BuRd",
    display_colorbar: bool = False,
    edge_width: float = 1.,
    options={},
    plot_options={},
    custom_colors: Dict[str, Dict[str, str]] = {},
    rename_values: Dict[str, Dict[str, str]] = {},
    legend_options: Dict[str, Any] = {},
    polygonize: bool = True
):
    """Creates a figure and plots a geometry in it from an already initialized compute slave

    Parameters
    ----------
    slave : Union[ComputeSlave, Geometry2D]
        Slave from which get the geometry polygons or Geometry2D interface object
    coloring_label : str
        Field to display in the plot
    axes : matplotlib.axes.Axes
        matplotlib axis object (obtainable from a plt.subplot call)
    u : Tuple[float, float, float], optional
        Horizontal coordinate director vector, by default X
    v : Tuple[float, float, float], optional
        Vertical coordinate director vector, by default Y
    u_min : float, optional
        Lower bound value along the u axis, by default 0.
    u_max : float, optional
        Upper bound value along the u axis, by default 1.
    v_min : float, optional
        Lower bound value along the v axis, by default 0.
    v_max : float, optional
        Upper bound value along the v axis, by default 1.
    w_value : float, optional
        Value along the u ^ v axis, by default "BuRd"
    color_map : str, optional
        Colormap used to color the polygons, by default "BuRd"
    edge_width : float, optional
        Cells edge width, by default 1.
    display_colorbar : bool, optional
        Display the polygons, by default False
    options : dict, optional
        Extra coloring options, by default {}
    plot_options : dict, optional
        Extra plot options given to geopandas, by default {}
    custom_colors : Dict[str, Dict[str, str]], optional
        HTML color code per field value. Expects a dictionnary such as:
        {
            MATERIAL:{"iron" : #ffffff, "water" : #aa6868ff}
        }
        where MATERIAL is a displayed field name, and iron/water are its values. This parameter is only used for string based field.
    rename_values : Dict[str, Dict[str, str]], optional
        Renames the values present in the data for the legend. Expects a dictionnary such as:
        {
            MATERIAL:{"iron" : "Iron", "water" : "Water"}
        }
        where MATERIAL is a displayed field name, and iron/water are its values. This parameter is only used for string based field.
    legend_options : Dict[str, Any]
        Dictionnary of options provided when creating the legend
    polygonize : bool
        Plot the geometry as polygons

    Returns
    ----------
        plt.Figure
            Figure containing the axes
        matplotlib.axes.Axes
            Axes in which the geometry was plotted
    """
    data, _ = slave.compute_2D_data(
        u=u,
        v=v,
        u_min=u_min,
        u_max=u_max,
        v_min=v_min,
        v_max=v_max,
        w_value=w_value,
        coloring_label=coloring_label,
        q_tasks=None,
        options=options,
    )

    set_colors_list(
        data,
        slave,
        coloring_label,
        color_map,
        False,
        options,
    )

    pw = PolygonSorter()
    pw.sort_from_value(
        data
    )

    if polygonize:
        plotter = Matplotlib2DPolygonPlotter()
        plotter.line_width = edge_width
    else:
        plotter = Matplotlib2DGridPlotter()
        plotter.line_width = edge_width

    # Replacing provided colors
    if (
        slave.get_label_coloring_mode(coloring_label) == VisualizationMode.FROM_STRING
        and coloring_label in custom_colors
    ):
        compo_list = data.cell_values
        cell_color_list = data.cell_colors

        compos = np.unique(compo_list)

        for compo in compos:
            if compo in custom_colors[coloring_label]:
                color_array = np.array(
                    [
                        [
                            int(c * 255)
                            for c in to_rgb(custom_colors[coloring_label][compo])
                        ]
                        + [255]
                    ]
                    * len(compo_list)
                )
                compo_array = np.repeat(np.expand_dims(compo_list, axis=1), 4, axis=1)
                cell_color_list = np.where(
                    compo_array == compo, color_array, cell_color_list
                )

        data.cell_values = compo_list
        data.cell_colors = cell_color_list
        data.cell_edge_colors = get_edges_colors(cell_color_list)

    if display_colorbar:
        compo_list = data.cell_values
        cell_color_list = data.cell_colors
        if (
            slave.get_label_coloring_mode(coloring_label)
            == VisualizationMode.FROM_VALUE
        ):

            values = np.array(compo_list).astype(float)

            plotter.set_color_map(color_map)
            plotter.update_colorbar(True, (np.nanmin(values), np.nanmin(values)))
        elif (
            slave.get_label_coloring_mode(coloring_label) == VisualizationMode.FROM_STRING
        ):
            compos = np.unique(compo_list)
            cell_color_list = np.array(cell_color_list).astype(float)

            edge_color_list = np.array(data.cell_edge_colors).astype(float)

            colors = []
            edge_colors = []
            legend_compos = []

            cell_color_list /= 255.0
            edge_color_list /= 255.0

            for compo in compos:
                location = compo_list.index(compo)

                add_in_legend = True

                if coloring_label in rename_values:
                    if compo in rename_values[coloring_label]:
                        legend_compos.append(rename_values[coloring_label][compo])
                    else:
                        add_in_legend = False
                else:
                    legend_compos.append(compo)

                if add_in_legend:
                    colors.append(cell_color_list[location])
                    edge_colors.append(edge_color_list[location])
            
            legend_elements = [
                Patch(facecolor=c, edgecolor=ce, label=label)
                for c, ce, label in zip(colors, edge_colors, legend_compos)
            ]

            axes.legend(handles=legend_elements, **legend_options)
        else:
            raise ValueError(
                f"Can't display the colorbar of a field whose visualisation mode is not FROM_VALUE or FROM_STRING, found {slave.get_label_coloring_mode(coloring_label)}"
            )

    plotter.plot_2d_frame_in_axes(data, axes=axes, plot_options=plot_options)


def plot_frame(
    slave: Union[ComputeSlave, Geometry2D],
    coloring_label: str,
    u: Tuple[float, float, float] = X,
    v: Tuple[float, float, float] = Y,
    u_min: float = 0.,
    u_max: float = 1.,
    v_min: float = 0.,
    v_max: float = 1.,
    w_value: float = 0.0,
    color_map: str = "BuRd",
    display_colorbar: bool = False,
    edge_width: float = 1.,
    options={},
    plot_options={},
    custom_colors: Dict[str, Dict[str, str]] = {},
    rename_values: Dict[str, Dict[str, str]] = {},
    legend_options: Dict[str, Any] = {},
    polygonize: bool = True
) -> Tuple[plt.Figure, matplotlib.axes.Axes]:
    """Creates a figure and plots a geometry in it from an already initialized compute slave

    Parameters
    ----------
    slave : Union[ComputeSlave, Geometry2D]
        Slave from which get the geometry polygons or Geometry2D interface object
    coloring_label : str
        Field to display in the plot
    u : Tuple[float, float, float], optional
        Horizontal coordinate director vector, by default X
    v : Tuple[float, float, float], optional
        Vertical coordinate director vector, by default Y
    u_min : float, optional
        Lower bound value along the u axis, by default 0.
    u_max : float, optional
        Upper bound value along the u axis, by default 1.
    v_min : float, optional
        Lower bound value along the v axis, by default 0.
    v_max : float, optional
        Upper bound value along the v axis, by default 1.
    w_value : float, optional
        Value along the u ^ v axis, by default "BuRd"
    color_map : str, optional
        Colormap used to color the polygons, by default "BuRd"
    display_colorbar : bool, optional
        Display the polygons, by default False
    edge_width : float, optional
        Cells edge width, by default 1.
    options : dict, optional
        Extra coloring options, by default {}
    plot_options : dict, optional
        Extra plot options given to geopandas, by default {}
    custom_colors : Dict[str, Dict[str, str]], optional
        HTML color code per field value. Expects a dictionnary such as:
        {
            MATERIAL:{"iron" : #ffffff, "water" : #aa6868ff}
        }
        where MATERIAL is a displayed field name, and iron/water are its values. This parameter is only used for string based field.
    rename_values : Dict[str, Dict[str, str]], optional
        Renames the values present in the data for the legend. Expects a dictionnary such as:
        {
            MATERIAL:{"iron" : "Iron", "water" : "Water"}
        }
        where MATERIAL is a displayed field name, and iron/water are its values. This parameter is only used for string based field.
    legend_options : Dict[str, Any]
        Dictionnary of options provided when creating the legend
    polygonize : bool
        Plot the geometry as polygons

    Returns
    ----------
        plt.Figure
            Figure containing the axes
        matplotlib.axes.Axes
            Axes in which the geometry was plotted
    """
    fig, axes = plt.subplots(1, 1)
    plot_frame_in_axes(
        slave=slave,
        u=u,
        v=v,
        u_min=u_min,
        u_max=u_max,
        v_min=v_min,
        v_max=v_max,
        coloring_label=coloring_label,
        axes=axes,
        w_value=w_value,
        color_map=color_map,
        display_colorbar=display_colorbar,
        edge_width=edge_width,
        options=options,
        plot_options=plot_options,
        custom_colors=custom_colors,
        rename_values=rename_values,
        legend_options=legend_options,
        polygonize=polygonize
    )
    return fig, axes
