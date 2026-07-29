import numpy as np
import pytest

from scivianna.utils.polygonize_tools import PolygonCoords


@pytest.fixture
def square_coords():
    """Fixture: returns a square polygon (0,0), (1,0), (1,1), (0,1)"""
    return PolygonCoords(x_coords=[0.0, 1.0, 1.0, 0.0], y_coords=[0.0, 0.0, 1.0, 1.0])


@pytest.fixture
def triangle_coords():
    """Fixture: returns a triangle polygon"""
    return PolygonCoords(x_coords=[0.0, 2.0, 1.0], y_coords=[0.0, 0.0, 2.0])


@pytest.mark.default
def test_initialization_with_list():
    """Test initialization with Python lists."""
    poly = PolygonCoords([0, 1], [0, 1])
    assert isinstance(poly.x_coords, np.ndarray)
    assert isinstance(poly.y_coords, np.ndarray)
    np.testing.assert_array_equal(poly.x_coords, [0, 1])
    np.testing.assert_array_equal(poly.y_coords, [0, 1])


@pytest.mark.default
def test_initialization_with_numpy_array():
    """Test initialization with NumPy arrays."""
    x = np.array([0, 1])
    y = np.array([0, 1])
    poly = PolygonCoords(x, y)
    np.testing.assert_array_equal(poly.x_coords, x)
    np.testing.assert_array_equal(poly.y_coords, y)


@pytest.mark.default
def test_initialization_mismatched_length():
    """Test that mismatched x and y lengths raise ValueError."""
    with pytest.raises(ValueError):
        PolygonCoords([0, 1], [0])


@pytest.mark.default
def test_initialization_invalid_type():
    """Test that non-array/list types raise TypeError."""
    with pytest.raises(TypeError):
        PolygonCoords("invalid", [0, 1])
    with pytest.raises(TypeError):
        PolygonCoords([0, 1], "invalid")


@pytest.mark.default
def test_translate(square_coords):
    """Test translation by (dx, dy)."""
    dx, dy = 2.5, -1.0
    square_coords.translate(dx, dy)

    expected_x = [2.5, 3.5, 3.5, 2.5]
    expected_y = [-1.0, -1.0, 0.0, 0.0]

    np.testing.assert_allclose(square_coords.x_coords, expected_x, atol=1e-10)
    np.testing.assert_allclose(square_coords.y_coords, expected_y, atol=1e-10)


@pytest.mark.default
def test_translate_inplace(square_coords):
    """Test that translation modifies object in place."""
    original_x = square_coords.x_coords.copy()
    original_y = square_coords.y_coords.copy()

    square_coords.translate(1.0, 1.0)

    np.testing.assert_allclose(square_coords.x_coords, original_x + 1.0, atol=1e-10)
    np.testing.assert_allclose(square_coords.y_coords, original_y + 1.0, atol=1e-10)


@pytest.mark.parametrize("angle, origin, expected_x, expected_y", [
    # 90° CCW around center (0.5, 0.5)
    (np.pi / 2, (0.5, 0.5), [1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]),

    # -90° (clockwise) around center
    (-np.pi / 2, (0.5, 0.5), [0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]),

    # 180° around center
    (np.pi, (0.5, 0.5), [1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0]),

    # 90° CCW around corner (0,0)
    (np.pi / 2, (0.0, 0.0), [0.0, 0.0, -1.0, -1.0], [0.0, 1.0, 1.0, 0.0]),

    # 0° rotation (no change)
    (0.0, (0.5, 0.5), [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]),])

@pytest.mark.default
def test_rotate(square_coords, angle, origin, expected_x, expected_y):
    """Parametrized test for rotation around various origins and angles."""
    # Reset to original
    square_coords.x_coords = np.array([0.0, 1.0, 1.0, 0.0])
    square_coords.y_coords = np.array([0.0, 0.0, 1.0, 1.0])

    square_coords.rotate(origin, angle)

    np.testing.assert_allclose(square_coords.x_coords, expected_x, atol=1e-10)
    np.testing.assert_allclose(square_coords.y_coords, expected_y, atol=1e-10)



@pytest.mark.default
def test_rotate_non_square(triangle_coords):
    """Test rotation on a non-square polygon."""
    # Triangle: (0,0), (2,0), (1,2)
    # Rotate 90° CCW around (1, 0) → should go to (1, -1), (1, 1), (-1, 0)
    origin = (1.0, 0.0)
    angle = np.pi / 2

    triangle_coords.rotate(origin, angle)
    
    expected_x = [1.0, 1.0, -1.0]
    expected_y = [-1.0, 1.0, 0.0]

    np.testing.assert_allclose(triangle_coords.x_coords, expected_x, atol=1e-10)
    np.testing.assert_allclose(triangle_coords.y_coords, expected_y, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])