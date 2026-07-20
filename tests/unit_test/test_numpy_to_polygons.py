# tests/test_polygonize_tools.py

import pytest
import numpy as np
from scivianna.utils.polygonize_tools import numpy_2D_array_to_polygons, PolygonCoords, PolygonElement
from scivianna.constants import OUTSIDE

@pytest.mark.grid
def test_numpy_2D_array_to_polygons_basic():
    """Test basic functionality with a simple 2x2 grid."""
    x = np.array([0, 1])
    y = np.array([0, 1])
    arr = np.array([[1, 1], [1, 2]])

    result = numpy_2D_array_to_polygons(x, y, arr, simplify=False)

    assert len(result) == 2
    assert set((result[0].cell_id, result[1].cell_id)) == set((1, 2))


@pytest.mark.grid
def test_numpy_2D_array_to_polygons_with_holes():
    """Test with a 3x3 array where one cell is a hole."""
    x = np.array([0, 1, 2])
    y = np.array([0, 1, 2])
    # Create a square with a hole in the center
    arr = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])

    result = numpy_2D_array_to_polygons(x, y, arr, simplify=False)

    assert len(result) == 2
    assert set((result[0].cell_id, result[1].cell_id)) == set((1, 0))

    # One of them has a hole
    assert len(result[0].holes) + len(result[1].holes) == 1


@pytest.mark.grid
def test_numpy_2D_array_to_polygons_simplify():
    """Test that simplify=True reduces polygon complexity."""
    x = np.array([0, 1, 2])
    y = np.array([0, 1, 2])
    arr = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    result = numpy_2D_array_to_polygons(x, y, arr, simplify=True)

    assert len(result) == 1
    assert result[0].cell_id == 1

@pytest.mark.grid
def test_numpy_2D_array_to_polygons_with_outside():
    """Test with a single value across entire array."""
    x = np.array([0, 1])
    y = np.array([0, 1])
    arr = np.array([[5, 5], [5, OUTSIDE]])

    result = numpy_2D_array_to_polygons(x, y, arr, simplify=False)

    assert len(result) == 2
    assert set((result[0].cell_id, result[1].cell_id)) == set((5, OUTSIDE))


@pytest.mark.grid
def test_numpy_2D_array_to_polygons_non_integer_values():
    """Test with non-integer values (strings or floats)."""
    x = np.array([0, 1])
    y = np.array([0, 1])
    arr = np.array([[1.5, 1.5], [1.5, 2.0]])

    result = numpy_2D_array_to_polygons(x, y, arr, simplify=False)

    assert len(result) == 2
    assert result[0].cell_id == 1.5
    assert result[1].cell_id == 2.0

@pytest.mark.grid
def test_numpy_2D_array_to_polygons_non_number_values():
    """Test with non-integer values (strings or floats)."""
    x = np.array([0, 1])
    y = np.array([0, 1])
    arr = np.array([["1.5", "1.5"], ["1.5", "2.0"]])

    result = numpy_2D_array_to_polygons(x, y, arr, simplify=False)

    assert len(result) == 2
    assert result[0].cell_id == "1.5"
    assert result[1].cell_id == "2.0"


@pytest.mark.grid
def test_numpy_2D_array_to_polygons_invalid_input():
    """Test invalid inputs."""
    x = np.array([0, 1])
    y = np.array([0, 1])
    arr = np.array([[1, 2], [3, 4]])

    # Test with non-array-like x or y
    with pytest.raises(TypeError):
        numpy_2D_array_to_polygons("not a list", y, arr, False)

    # Test with mismatched dimensions
    with pytest.raises(ValueError):
        numpy_2D_array_to_polygons(x, y, np.array([[1, 2, 3], [4, 5, 6]]), False)

    # Test with invalid array type
    with pytest.raises(TypeError):
        numpy_2D_array_to_polygons(x, y, "not an array", False)
