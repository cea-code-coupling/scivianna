import numpy as np
import pytest

from scivianna.data.data2d import Data2D
from scivianna.utils.polygonize_tools import PolygonCoords, PolygonElement

def build_data2d_polygons(order_shift=0):
    """Adding order shift to change the cells order, the operators sorts the cells before the operations"""
    data_2d = Data2D.from_polygon_list(
        [
            PolygonElement(
                exterior_polygon = PolygonCoords(
                    x_coords=[0 + i, 1 + i, 1 + i, 0 + i],
                    y_coords=[0, 0, 1, 1]
                ),
                holes = [],
                cell_id = f"{(i + order_shift)%3}"
            )
            for i in range(3)
        ]
    )
    data_2d.cell_values = np.array([0., 1., 2.])
    return data_2d

def build_data2d_grid():
    data_2d = Data2D.from_grid(
        grid=np.array([
            [0, 1],
            [2, 3]
        ]),
        u_values=[0, 1, 2],
        v_values=[0, 1, 2]
    )
    data_2d.cell_values = np.array([0., 1., 2., 3.])
    return data_2d

@pytest.mark.default
def test_add_grid():
    data1 = build_data2d_grid()
    data2 = build_data2d_grid()
    result = data1 + data2
    np.testing.assert_equal(result.cell_values, [0., 2., 4., 6.])

@pytest.mark.default
def test_radd_grid():
    data = build_data2d_grid()
    result = 1 + data
    np.testing.assert_equal(result.cell_values, [1, 2, 3, 4])

@pytest.mark.default
def test_sub_grid():
    data1 = build_data2d_grid()
    data2 = build_data2d_grid()
    result = data1 - data2
    np.testing.assert_equal(result.cell_values, [0, 0, 0, 0])

@pytest.mark.default
def test_rsub():
    data = build_data2d_grid()
    result = 1 - data
    np.testing.assert_equal(result.cell_values, [1, 0, -1, -2])

@pytest.mark.default
def test_sub_float():
    data = build_data2d_grid()
    result = data - 1
    np.testing.assert_equal(result.cell_values, [-1, 0, 1, 2])

@pytest.mark.default
def test_mul_grid():
    data1 = build_data2d_grid()
    data2 = build_data2d_grid()
    result = data1 * data2
    np.testing.assert_equal(result.cell_values, [0, 1, 4, 9])

@pytest.mark.default
def test_rmul_grid():
    data = build_data2d_grid()
    result = 2 * data
    np.testing.assert_equal(result.cell_values, [0, 2, 4, 6])

@pytest.mark.default
def test_truediv_grid():
    data1 = build_data2d_grid()
    data2 = build_data2d_grid()
    result = data1 / data2
    np.testing.assert_equal(result.cell_values, [np.nan, 1, 1, 1])

@pytest.mark.default
def test_rtruediv_grid():
    data = build_data2d_grid()
    result = 1 / data
    np.testing.assert_equal(result.cell_values, [np.inf, 1, .5, 1/3])

@pytest.mark.default
def test_add_polygons():
    data1 = build_data2d_polygons()
    data2 = build_data2d_polygons()
    result = data1 + data2
    np.testing.assert_equal(result.cell_values, [0, 2, 4])

@pytest.mark.default
def test_add_polygons_order_shift():
    data1 = build_data2d_polygons(order_shift=1)
    data2 = build_data2d_polygons(order_shift=2)
    result = data1 + data2
    np.testing.assert_equal(result.cell_values, [0, 2, 4])

@pytest.mark.default
def test_radd_polygons():
    data = build_data2d_polygons()
    result = 1 + data
    np.testing.assert_equal(result.cell_values, [1, 2, 3])

@pytest.mark.default
def test_radd_polygons_order_shift():
    data = build_data2d_polygons(order_shift=1)
    result = 1 + data
    np.testing.assert_equal(result.cell_values, [1, 2, 3])

@pytest.mark.default
def test_sub_polygons():
    data1 = build_data2d_polygons()
    data2 = build_data2d_polygons()
    result = data1 - data2
    np.testing.assert_equal(result.cell_values, [0, 0, 0])

@pytest.mark.default
def test_sub_polygons_order_shift():
    data1 = build_data2d_polygons(order_shift=1)
    data2 = build_data2d_polygons(order_shift=2)
    result = data1 - data2
    np.testing.assert_equal(result.cell_values, [0, 0, 0])

@pytest.mark.default
def test_rsub_polygons():
    data = build_data2d_polygons()
    result = 1 - data
    np.testing.assert_equal(result.cell_values, [1, 0, -1])

@pytest.mark.default
def test_rsub_polygons_order_shift():
    data = build_data2d_polygons(order_shift=1)
    result = 1 - data
    np.testing.assert_equal(result.cell_values, [1, 0, -1])

@pytest.mark.default
def test_mul_polygons():
    data1 = build_data2d_polygons()
    data2 = build_data2d_polygons()
    result = data1 * data2
    np.testing.assert_equal(result.cell_values, [0, 1, 4])

@pytest.mark.default
def test_mul_polygons_order_shift():
    data1 = build_data2d_polygons(order_shift=1)
    data2 = build_data2d_polygons(order_shift=2)
    result = data1 * data2
    np.testing.assert_equal(result.cell_values, [0, 1, 4])

@pytest.mark.default
def test_rmul_polygons():
    data = build_data2d_polygons()
    result = 2 * data
    np.testing.assert_equal(result.cell_values, [0, 2, 4])

@pytest.mark.default
def test_rmul_polygons_order_shift():
    data = build_data2d_polygons(order_shift=1)
    result = 2 * data
    np.testing.assert_equal(result.cell_values, [0, 2, 4])

@pytest.mark.default
def test_truediv_polygons():
    data1 = build_data2d_polygons()
    data2 = build_data2d_polygons()
    result = data1 / data2
    np.testing.assert_equal(result.cell_values, [np.nan, 1, 1])

@pytest.mark.default
def test_truediv_polygons_order_shift():
    data1 = build_data2d_polygons(order_shift=1)
    data2 = build_data2d_polygons(order_shift=2)
    result = data1 / data2
    print(data1.cell_ids)
    print(data2.cell_ids)
    np.testing.assert_equal(result.cell_values, [np.nan, 1, 1])

@pytest.mark.default
def test_rtruediv_polygons():
    data = build_data2d_polygons()
    result = 1 / data
    np.testing.assert_equal(result.cell_values, [np.inf, 1, 0.5])

@pytest.mark.default
def test_rtruediv_polygons_order_shift():
    data = build_data2d_polygons(order_shift=1)
    result = 1 / data
    np.testing.assert_equal(result.cell_values, [np.inf, 1, 0.5])
