"""Public matplotlib plotting API: render a scivianna frame."""

from random import randint
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from scivianna.constants import X, Y
from scivianna.data.data2d import Data2D
from scivianna.enums import VisualizationMode
from scivianna.extension.field_selector import set_colors_list
from scivianna.interface.generic_interface import Geometry2D
from scivianna.plotter_2d.grid.matplotlib import Matplotlib2DGridPlotter
from scivianna.plotter_2d.polygon.matplotlib import Matplotlib2DPolygonPlotter
from scivianna.slave import ComputeSlave
from scivianna.utils.color_tools import get_edges_colors
from scivianna.utils.polygon_sorter import PolygonSorter


def get_data(
    slave: Union[ComputeSlave, Geometry2D],
    coloring_label: str,
    u: Tuple[float, float, float] = X,
    v: Tuple[float, float, float] = Y,
    origin: Tuple[float, float, float] = None,
    size_u: float = 1.0,
    size_v: float = 1.0,
    color_map: str = "BuRd",
    options={},
    custom_colors: Dict[str, Dict[str, str]] = {},
    caller: str = "MatplotlibAPI",
    color_seed: int = randint(0, 1000),
    log_scale: bool = False,
) -> Data2D:
    """Computes a 2D slice of a geometry and colors its cells according to a field

    Parameters
    ----------
    slave : Union[ComputeSlave, Geometry2D]
        Slave from which get the geometry polygons or Geometry2D interface object
    coloring_label : str
        Field used to color the cells
    u : Tuple[float, float, float], optional
        Horizontal coordinate director vector, by default X
    v : Tuple[float, float, float], optional
        Vertical coordinate director vector, by default Y
    origin : Tuple[float, float, float], optional
        Physical 3D position of the slice center, by default (0, 0, 0)
    size_u : float, optional
        Size of the slice along the u axis, by default 1.
    size_v : float, optional
        Size of the slice along the v axis, by default 1.
    color_map : str, optional
        Colormap used to color the polygons, by default "BuRd"
    options : dict, optional
        Extra coloring options, by default {}
    custom_colors : Dict[str, Dict[str, str]], optional
        HTML color code per field value. Expects a dictionnary such as:
        {
            MATERIAL:{"iron" : #ffffff, "water" : #aa6868ff}
        }
        where MATERIAL is a displayed field name, and iron/water are its values. This parameter is only used for string based field.
    caller : str, optional
        Name of the caller, by default "MatplotlibAPI"
    color_seed : int
        Seed used to randomize the colors in FROM_STRING display mode. By default randomized from 0 to 1000.
    log_scale : bool
        Use logarithmic scale. By default False.

    Returns
    ----------
        Data2D
            Sliced geometry, with cell values and colors set
    """
    # If origin is None, default to (0, 0, 0)
    if origin is None:
        origin = (0.0, 0.0, 0.0)

    data, _ = slave.compute_2D_data(
        u=tuple(u),
        v=tuple(v),
        origin=tuple(origin),
        size_u=size_u,
        size_v=size_v,
        q_tasks=None,
        coloring_label=coloring_label,
        options=options,
        caller=caller,
    )

    set_colors_list(
        data=data,
        slave=slave,
        coloring_label=coloring_label,
        color_map=color_map,
        center_colormap_on_zero=False,
        options=options,
        seed=color_seed,
        log_scale=log_scale,
    )

    pw = PolygonSorter()
    pw.sort_from_value(data)

    # Replacing provided colors
    if (
        coloring_label in custom_colors
        and slave.get_label_coloring_mode(coloring_label) == VisualizationMode.FROM_STRING
    ):
        compo_list = data.cell_values
        cell_color_list = data.cell_colors

        compos = np.unique(compo_list)

        for compo in compos:
            if compo in custom_colors[coloring_label]:
                color_array = np.array(
                    [[int(c * 255) for c in to_rgb(custom_colors[coloring_label][compo])] + [255]]
                    * len(compo_list)
                )
                compo_array = np.repeat(np.expand_dims(compo_list, axis=1), 4, axis=1)
                cell_color_list = np.where(compo_array == compo, color_array, cell_color_list)

        data.cell_values = compo_list
        data.cell_colors = cell_color_list
        data.cell_edge_colors = get_edges_colors(cell_color_list)

    return data


def _guess_coloring_mode(data: Data2D) -> VisualizationMode:
    """Guesses the coloring mode of a Data2D from its cell values: numeric
    values are displayed with a colorbar, other values with a legend."""
    try:
        np.array(data.cell_values).astype(float)
        return VisualizationMode.FROM_VALUE
    except (TypeError, ValueError):
        return VisualizationMode.FROM_STRING


def plot_data_in_axes(
    data: Data2D,
    axes: matplotlib.axes.Axes,
    color_map: str = "BuRd",
    display_colorbar: bool = False,
    edge_width: float = 1.0,
    plot_options={},
    rename_values: Optional[Dict[str, str]] = None,
    legend_options: Dict[str, Any] = {},
    polygonize: bool = True,
    log_scale: bool = False,
    coloring_mode: Optional[VisualizationMode] = None,
):
    """Plots an already computed and colored Data2D (see get_data) in a matplotlib axes

    Parameters
    ----------
    data : Data2D
        Geometry data, with cell values and colors set
    axes : matplotlib.axes.Axes
        matplotlib axis object (obtainable from a plt.subplot call)
    color_map : str, optional
        Colormap displayed in the colorbar, by default "BuRd". Should match the one used to color the data.
    display_colorbar : bool, optional
        Display the colorbar (FROM_VALUE) or legend (FROM_STRING), by default False
    edge_width : float, optional
        Cells edge width, by default 1.
    plot_options : dict, optional
        Extra plot options given to geopandas, by default {}
    rename_values : Optional[Dict[str, str]], optional
        Renames the values present in the data for the legend, such as {"iron" : "Iron", "water" : "Water"}.
        If provided, only the values present in this dictionnary are added to the legend. Only used for string based field.
    legend_options : Dict[str, Any]
        Dictionnary of options provided when creating the legend
    polygonize : bool
        Plot the geometry as polygons
    log_scale : bool
        Use logarithmic scale. By default False.
    coloring_mode : Optional[VisualizationMode]
        Coloring mode of the displayed field, used for the colorbar/legend.
        By default guessed from the cell values (numeric: FROM_VALUE, otherwise FROM_STRING).
    """
    if polygonize:
        plotter = Matplotlib2DPolygonPlotter()
    else:
        plotter = Matplotlib2DGridPlotter()
    plotter.line_width = edge_width

    if display_colorbar:
        if coloring_mode is None:
            coloring_mode = _guess_coloring_mode(data)

        compo_list = data.cell_values
        cell_color_list = data.cell_colors

        if coloring_mode == VisualizationMode.FROM_VALUE:
            values = np.array(compo_list).astype(float)

            plotter.set_color_map(color_map)
            if np.isnan(values).all():
                plotter.update_colorbar(False, (np.nan, np.nan), log_scale=log_scale)
            else:
                plotter.update_colorbar(True, (np.nanmin(values), np.nanmax(values)), log_scale=log_scale)

        elif coloring_mode == VisualizationMode.FROM_STRING:
            compos = np.unique(compo_list)
            cell_color_list = np.array(cell_color_list).astype(float) / 255.0
            edge_color_list = np.array(data.cell_edge_colors).astype(float) / 255.0

            colors = []
            edge_colors = []
            legend_compos = []

            for compo in compos:
                if rename_values is not None:
                    if compo not in rename_values:
                        continue
                    legend_compos.append(rename_values[compo])
                else:
                    legend_compos.append(compo)

                location = list(compo_list).index(compo)
                colors.append(cell_color_list[location])
                edge_colors.append(edge_color_list[location])

            legend_elements = [
                Patch(facecolor=c, edgecolor=ce, label=label)
                for c, ce, label in zip(colors, edge_colors, legend_compos)
            ]

            axes.legend(handles=legend_elements, **legend_options)
        else:
            raise ValueError(
                f"Can't display the colorbar of a field whose visualisation mode is not FROM_VALUE or FROM_STRING, found {coloring_mode}"
            )

    plotter.plot_2d_frame_in_axes(data, axes=axes, plot_options=plot_options)


def plot_data(
    data: Data2D,
    color_map: str = "BuRd",
    display_colorbar: bool = False,
    edge_width: float = 1.0,
    plot_options={},
    rename_values: Optional[Dict[str, str]] = None,
    legend_options: Dict[str, Any] = {},
    polygonize: bool = True,
    log_scale: bool = False,
    coloring_mode: Optional[VisualizationMode] = None,
) -> Tuple[plt.Figure, matplotlib.axes.Axes]:
    """Creates a figure and plots an already computed and colored Data2D (see get_data) in it

    Parameters
    ----------
    See plot_data_in_axes.

    Returns
    ----------
        plt.Figure
            Figure containing the axes
        matplotlib.axes.Axes
            Axes in which the geometry was plotted
    """
    fig, axes = plt.subplots(1, 1)
    plot_data_in_axes(
        data=data,
        axes=axes,
        color_map=color_map,
        display_colorbar=display_colorbar,
        edge_width=edge_width,
        plot_options=plot_options,
        rename_values=rename_values,
        legend_options=legend_options,
        polygonize=polygonize,
        log_scale=log_scale,
        coloring_mode=coloring_mode,
    )
    return fig, axes


def plot_frame_in_axes(
    slave: Union[ComputeSlave, Geometry2D],
    coloring_label: str,
    axes: matplotlib.axes.Axes,
    u: Tuple[float, float, float] = X,
    v: Tuple[float, float, float] = Y,
    origin: Tuple[float, float, float] = None,
    size_u: float = 1.0,
    size_v: float = 1.0,
    color_map: str = "BuRd",
    display_colorbar: bool = False,
    edge_width: float = 1.0,
    options={},
    plot_options={},
    custom_colors: Dict[str, Dict[str, str]] = {},
    rename_values: Dict[str, Dict[str, str]] = {},
    legend_options: Dict[str, Any] = {},
    polygonize: bool = True,
    caller: str = "MatplotlibAPI",
    color_seed: int = randint(0, 1000),
    log_scale: bool = False,
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
    origin : Tuple[float, float, float], optional
        Physical 3D position of the slice center, by default (0, 0, 0)
    size_u : float, optional
        Size of the slice along the u axis, by default 1.
    size_v : float, optional
        Size of the slice along the v axis, by default 1.
    color_map : str, optional
        Colormap used to color the polygons, by default "BuRd"
    edge_width : float, optional
        Cells edge width, by default 1.
    display_colorbar : bool, optional
        Display the colorbar, by default False
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
    color_seed : int
        Seed used to randomize the colors in FROM_STRING display mode. By default randomized from 0 to 1000.
    log_scale : bool
        Use logarithmic scale. By default False.

    Returns
    ----------
        plt.Figure
            Figure containing the axes
        matplotlib.axes.Axes
            Axes in which the geometry was plotted
    """
    data = get_data(
        slave=slave,
        coloring_label=coloring_label,
        u=u,
        v=v,
        origin=origin,
        size_u=size_u,
        size_v=size_v,
        color_map=color_map,
        options=options,
        custom_colors=custom_colors,
        caller=caller,
        color_seed=color_seed,
        log_scale=log_scale,
    )

    plot_data_in_axes(
        data=data,
        axes=axes,
        color_map=color_map,
        display_colorbar=display_colorbar,
        edge_width=edge_width,
        plot_options=plot_options,
        rename_values=rename_values.get(coloring_label) if coloring_label in rename_values else None,
        legend_options=legend_options,
        polygonize=polygonize,
        log_scale=log_scale,
        coloring_mode=slave.get_label_coloring_mode(coloring_label) if display_colorbar else None,
    )


def plot_frame(
    slave: Union[ComputeSlave, Geometry2D],
    coloring_label: str,
    u: Tuple[float, float, float] = X,
    v: Tuple[float, float, float] = Y,
    origin: Tuple[float, float, float] = None,
    size_u: float = 1.0,
    size_v: float = 1.0,
    color_map: str = "BuRd",
    display_colorbar: bool = False,
    edge_width: float = 1.0,
    options={},
    plot_options={},
    custom_colors: Dict[str, Dict[str, str]] = {},
    rename_values: Dict[str, Dict[str, str]] = {},
    legend_options: Dict[str, Any] = {},
    polygonize: bool = True,
    caller: str = "MatplotlibAPI",
    color_seed: int = randint(0, 1000),
    log_scale: bool = False,
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
    origin : Tuple[float, float, float], optional
        Physical 3D position of the slice center, by default (0, 0, 0)
    size_u : float, optional
        Size of the slice along the u axis, by default 1.
    size_v : float, optional
        Size of the slice along the v axis, by default 1.
    color_map : str, optional
        Colormap used to color the polygons, by default "BuRd"
    display_colorbar : bool, optional
        Display the colorbar, by default False
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
    color_seed : int
        Seed used to randomize the colors in FROM_STRING display mode. By default randomized from 0 to 1000.
    log_scale : bool
        Use logarithmic scale. By default False.

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
        origin=origin,
        size_u=size_u,
        size_v=size_v,
        coloring_label=coloring_label,
        axes=axes,
        color_map=color_map,
        display_colorbar=display_colorbar,
        edge_width=edge_width,
        options=options,
        plot_options=plot_options,
        custom_colors=custom_colors,
        rename_values=rename_values,
        legend_options=legend_options,
        polygonize=polygonize,
        caller=caller,
        color_seed=color_seed,
        log_scale=log_scale
    )
    return fig, axes
