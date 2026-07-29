from typing import Tuple

import numpy as np

from scivianna.data.data2d import Data2D


def get_grids(
    data: Data2D, display_edges: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Provides 2D grids and color 3D grid from a Data2D, darkens the edges if requested

    Parameters
    ----------
    data : Data2D
        Data2D to display
    display_edges : bool
        Darken edges

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Bokeh readable 2D image, 3D color grid, id grid, value grid
    """
    grid = data.get_grid()
    flat_grid = grid.flatten()
    vals, inv = np.unique(flat_grid, return_inverse=True)

    value_map = dict(zip(data.cell_ids, data.cell_values))
    color_map = dict(zip(data.cell_ids, data.cell_colors))

    value_array = np.array([value_map[val] for val in vals])
    color_array = np.array([color_map[val] for val in vals])

    colors = color_array[inv]  # shape (n, m, 4)

    if display_edges:
        flat_data = grid.flatten()
        roll_1_0 = np.where(flat_data == np.roll(flat_data, -1), 1, 0)
        roll_1_1 = np.where(flat_data == np.roll(flat_data, 1), 1, 0)
        contour_1_0 = roll_1_0.reshape(grid.shape)
        contour_1_1 = roll_1_1.reshape(grid.shape)

        flat_data_2 = grid.T.flatten()
        roll_2_0 = np.where(flat_data_2 == np.roll(flat_data_2, -1), 1, 0)
        roll_2_1 = np.where(flat_data_2 == np.roll(flat_data_2, 1), 1, 0)

        contour_2_0 = roll_2_0.reshape(grid.T.shape).T
        contour_2_1 = roll_2_1.reshape(grid.T.shape).T

        borders = np.expand_dims(
            np.minimum(
                np.minimum(contour_1_0, contour_2_0),
                np.minimum(contour_1_1, contour_2_1),
            ).flatten(),
            axis=-1,
        )

        borders = np.concatenate([borders, borders, borders, borders], axis=1)

        color_edge_map = dict(zip(data.cell_ids, data.cell_edge_colors))
        edge_color_array = np.array([color_edge_map[val] for val in vals])

        edge_colors = edge_color_array[inv]  # shape (n, m, 4)

        colors = np.where(borders == (1, 1, 1, 1), colors, edge_colors).reshape((*grid.shape, 4))
    else:
        colors = colors.reshape((*grid.shape, 4))

    val_grid = value_array[inv].reshape(grid.shape)

    img = np.empty(grid.shape, dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape(colors.shape)
    view[:, :, :] = colors[:, :, :]

    return img, view, grid, val_grid
