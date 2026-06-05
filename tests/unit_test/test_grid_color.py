import numpy as np
import pytest

from scivianna.data.data2d import Data2D
from scivianna.plotter_2d.grid.grid_tools import get_grids
from scivianna.utils.color_tools import get_edges_colors

@pytest.mark.default
def test_get_grid():
    size = 10
    arr = np.zeros((size, size))

    arr[:, :int(size/2)] += 2
    arr[:int(size/2), :] += 1

    data = Data2D.from_grid(arr, u_values=np.arange(size+1), v_values=np.arange(size+1))

    data.cell_values = data.cell_ids.copy()
    data.cell_colors = np.array([
        [0, 255, 0, 255.],
        [255, 125, 0, 255.],
        [255, 0, 255, 255.],
        [0, 255, 255, 255.],
    ])
    data.cell_edge_colors = get_edges_colors(data.cell_colors)

    _, view, _, _ = get_grids(data, True)

    # Test on RED : 
    # -     first line ok : figure borders darkened
    # -     second line ok : borders darkened only
    # -     int(size/2)-1 line ok : borders darkened, also inside
    # -     second column : function also works in vertical
    np.testing.assert_almost_equal(view[0, :, 0], [0, 0, 0, 0, 0, 235, 235, 235, 235, 235])
    np.testing.assert_almost_equal(view[1, :, 0], [0, 0, 0, 0, 0, 235, 255, 255, 255, 235])
    np.testing.assert_almost_equal(view[int(size/2)-1, :, 0], [0, 0, 0, 0, 0, 235, 235, 235, 235, 235])
    np.testing.assert_almost_equal(view[:, 1, 0], [0, 0, 0, 0, 0, 235, 255, 255, 255, 235])

    # Test on other colors
    np.testing.assert_almost_equal(view[1, :, 1], [235, 255, 255, 255, 235, 105, 125, 125, 125, 105])
    np.testing.assert_almost_equal(view[1, :, 2], [235, 255, 255, 255, 235, 0, 0, 0, 0, 0])

if __name__ == "__main__":
    test_get_grid()