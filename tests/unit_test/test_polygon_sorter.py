from typing import List, Union

import numpy as np
import pytest

from scivianna.data.data2d import Data2D, DataType
from scivianna.utils.polygon_sorter import PolygonSorter
from scivianna.utils.polygonize_tools import PolygonCoords, PolygonElement


@pytest.fixture
def mock_data()->Data2D:
    """Returns a simple Data2D mock object for testing."""
    data = Data2D.from_polygon_list([
        PolygonElement(
            exterior_polygon=PolygonCoords(
                x_coords=[0.0, 1.0, 1.0],
                y_coords=[0.0, 0.0, 1.0]
            ),
            holes=[],
            cell_id=101
        ),
        PolygonElement(
            exterior_polygon=PolygonCoords(
                x_coords=[2.0, 3.0, 3.0],
                y_coords=[0.0, 0.0, 1.0]
            ),
            holes=[],
            cell_id=102
        ),
        PolygonElement(
            exterior_polygon=PolygonCoords(
                x_coords=[4.0, 5.0, 5.0],
                y_coords=[0.0, 0.0, 1.0]
            ),
            holes=[],
            cell_id=103
        ),
        PolygonElement(
            exterior_polygon=PolygonCoords(
                x_coords=[6.0, 7.0, 7.0],
                y_coords=[0.0, 0.0, 1.0]
            ),
            holes=[],
            cell_id=104
        )
    ])
    data.cell_values = ["C", "A", "B", "D"]
    data.cell_colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
    data.cell_edge_colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
    return data


@pytest.mark.default
def test_polygon_sorter_init():
    """Test that PolygonSorter initializes correctly."""
    sorter = PolygonSorter()
    assert sorter.sort_indexes is None


@pytest.mark.default
def test_sort_polygon_list_sorts_by_cell_values(mock_data:Data2D):
    """Test that sort_polygon_list sorts by cell_values in ascending order."""
    sorter = PolygonSorter()
    sorter.sort_from_value(mock_data)

    # After sorting: A, B, C, D
    expected_order = ["A", "B", "C", "D"]
    np.testing.assert_equal(mock_data.cell_values, expected_order)

    # Check that associated data is sorted in same order
    np.testing.assert_equal(mock_data.cell_colors, [(255, 0, 0), (0, 255, 0), (255, 255, 255), (0, 0, 255)])
    np.testing.assert_equal(mock_data.cell_edge_colors, [(255, 0, 0), (0, 255, 0), (255, 255, 255), (0, 0, 255)])
    np.testing.assert_equal(mock_data.cell_ids, [102, 103, 101, 104])
    np.testing.assert_equal(len(mock_data.polygons), 4)


@pytest.mark.default
def test_sort_polygon_list_preserves_polygons(mock_data:Data2D):
    """Test that polygons are sorted in same order as cell_values."""
    _original_polygons = mock_data.polygons

    sorter = PolygonSorter()
    sorter.sort_from_value(mock_data)

    # Original order of polygons: [0, 1, 2, 3] → after sort: [1, 2, 0, 3]
    expected_polygons_order = [1, 2, 0, 3]
    for i, idx in enumerate(expected_polygons_order):
        assert mock_data.polygons[i] == _original_polygons[idx]


@pytest.mark.default
def test_sort_polygon_list_invalid_length_raises(mock_data:Data2D):
    """Test that mismatched lengths raise AssertionError."""
    sorter = PolygonSorter()
    mock_data.cell_values = ["A", "B", "C"]
    mock_data.cell_colors = ["red", "green", "blue", "yellow"]  # Mismatched

    with pytest.raises(AssertionError):
        sorter.sort_from_value(mock_data)


@pytest.mark.default
def test_sort_polygon_list_invalid_polygon_length_raises(mock_data:Data2D):
    """Test that mismatched polygon count raises AssertionError."""
    sorter = PolygonSorter()
    mock_data.cell_values = ["A", "B", "C"]
    mock_data.polygons = mock_data.polygons[:2]  # Remove one polygon

    with pytest.raises(AssertionError):
        sorter.sort_from_value(mock_data)


@pytest.mark.default
def test_sort_list_preserves_order(mock_data:Data2D):
    """Test that sort_list applies the same sort order to new data."""
    sorter = PolygonSorter()
    sorter.sort_from_value(mock_data)

    # Create new data with same structure but different order
    new_data = Data2D.from_polygon_list([
        PolygonElement(exterior_polygon=PolygonCoords(x_coords=[0.0, 1.0, 1.0], y_coords=[0.0, 0.0, 1.0]), holes=[], cell_id=101),
        PolygonElement(exterior_polygon=PolygonCoords(x_coords=[2.0, 3.0, 3.0], y_coords=[0.0, 0.0, 1.0]), holes=[], cell_id=102),
        PolygonElement(exterior_polygon=PolygonCoords(x_coords=[4.0, 5.0, 5.0], y_coords=[0.0, 0.0, 1.0]), holes=[], cell_id=103),
        PolygonElement(exterior_polygon=PolygonCoords(x_coords=[6.0, 7.0, 7.0], y_coords=[0.0, 0.0, 1.0]), holes=[], cell_id=104)
    ])
    new_data.cell_values = ["D", "C", "B", "A"]
    new_data.cell_colors = [(0, 0, 255), (255, 255, 255), (0, 255, 0), (255, 0, 0)]
    new_data.cell_edge_colors = [(0, 0, 255), (255, 255, 255), (0, 255, 0), (255, 0, 0)]
    new_data.cell_ids = [104, 101, 103, 102]

    # Apply sort_list
    sorter.sort_list(new_data)

    # Should now be sorted: A, B, C, D
    np.testing.assert_equal(new_data.cell_values, ['C', 'B', 'D', 'A'])
    np.testing.assert_equal(new_data.cell_colors, [(255, 255, 255), (0, 255, 0), (0, 0, 255), (255, 0, 0)])
    np.testing.assert_equal(new_data.cell_edge_colors, [(255, 255, 255), (0, 255, 0), (0, 0, 255), (255, 0, 0)])
    np.testing.assert_equal(new_data.cell_ids, [101, 103, 104, 102])


@pytest.mark.default
def test_sort_list_with_different_length_raises(mock_data:Data2D):
    """Test that sort_list raises error if data length doesn't match sort_indexes."""
    sorter = PolygonSorter()
    sorter.sort_from_value(mock_data)

    # Create data with mismatched length
    new_data = Data2D.from_polygon_list([PolygonElement(exterior_polygon=PolygonCoords(x_coords=[0.0, 1.0, 1.0], y_coords=[0.0, 0.0, 1.0]), holes=[], cell_id=101)])
    new_data.cell_values = ["A"]
    new_data.cell_colors = [(255, 0, 0)]
    new_data.cell_edge_colors = [(255, 0, 0)]
    new_data.cell_ids = [101]

    with pytest.raises(AssertionError):
        sorter.sort_list(new_data)


        
@pytest.mark.default
def test_sort_polygon_list_with_nan_in_values(mock_data:Data2D):
    """Test that values are sorted with a nan in the values."""
    mock_data.cell_values = [0., 3., np.nan, 1.]

    sorter = PolygonSorter()
    sorter.sort_from_value(mock_data)  

    # Should preserve the previous order
    np.testing.assert_equal(mock_data.cell_values, [0.0, 1.0, 3.0, np.nan]) 
    np.testing.assert_equal(mock_data.cell_colors, [(255, 255, 255), (0, 0, 255), (255, 0, 0), (0, 255, 0)])
    np.testing.assert_equal(mock_data.cell_edge_colors, [(255, 255, 255), (0, 0, 255), (255, 0, 0), (0, 255, 0)])


        
@pytest.mark.default
def test_sort_polygon_list_with_float_values(mock_data:Data2D):
    """Test that values are sorted with a nan in the values."""
    mock_data.cell_values = [0., 3., 12., 1.]

    sorter = PolygonSorter()
    sorter.sort_from_value(mock_data)  

    # Should preserve the previous order
    np.testing.assert_equal(mock_data.cell_values, [0.0, 1.0, 3.0, 12.])  
    np.testing.assert_equal(mock_data.cell_colors, [(255, 255, 255), (0, 0, 255), (255, 0, 0), (0, 255, 0)])
    np.testing.assert_equal(mock_data.cell_edge_colors, [(255, 255, 255), (0, 0, 255), (255, 0, 0), (0, 255, 0)])

    mock_data.cell_values = [2., 7., 3., 1.]
    mock_data.cell_colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
    mock_data.cell_edge_colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
    sorter.sort_list(mock_data)  
    
    np.testing.assert_equal(mock_data.cell_values, [2., 1., 7., 3.])  
    np.testing.assert_equal(mock_data.cell_colors, [(255, 255, 255), (0, 0, 255), (255, 0, 0), (0, 255, 0)])
    np.testing.assert_equal(mock_data.cell_edge_colors, [(255, 255, 255), (0, 0, 255), (255, 0, 0), (0, 255, 0)])

        
@pytest.mark.default
def test_sort_polygon_list_with_both_string_float(mock_data:Data2D):
    """Test that values are sorted with a nan in the values."""
    mock_data.cell_values = ["0.", 3., np.nan, 1.]

    sorter = PolygonSorter()
    
    with pytest.raises(AssertionError):
        sorter.sort_list(mock_data)
