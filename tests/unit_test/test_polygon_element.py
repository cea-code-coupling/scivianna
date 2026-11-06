"""Unit tests for PolygonElement (construction, validation and transforms)."""

import numpy as np
import pytest

from scivianna.utils.polygonize_tools import PolygonCoords, PolygonElement


# Helper: Create a simple polygon for testing
def create_test_polygon_coords():
    """A unit square as PolygonCoords."""
    x_coords = [0.0, 1.0, 1.0, 0.0]
    y_coords = [0.0, 0.0, 1.0, 1.0]
    return PolygonCoords(x_coords, y_coords)

def create_test_hole_coords():
    """A centered square hole as PolygonCoords."""
    x_coords = [0.25, 0.75, 0.75, 0.25]
    y_coords = [0.25, 0.25, 0.75, 0.75]
    return PolygonCoords(x_coords, y_coords)


@pytest.mark.default
def test_init_valid():
    """A valid PolygonElement is constructed from coords and a cell id."""
    exterior = create_test_polygon_coords()
    holes = [create_test_hole_coords()]
    pe = PolygonElement(exterior_polygon=exterior, holes=holes, cell_id="test_vol")
    assert pe.exterior_polygon is exterior
    assert pe.holes == holes
    assert pe.cell_id == "test_vol"
    assert pe.compo == ""

@pytest.mark.default
def test_init_invalid_exterior_type():
    """A non-PolygonCoords exterior raises TypeError."""
    with pytest.raises(TypeError):
        PolygonElement(exterior_polygon="not a PolygonCoords", holes=[], cell_id="test")

@pytest.mark.default
def test_init_invalid_holes_type():
    """A non-list of holes raises TypeError."""
    with pytest.raises(TypeError):
        PolygonElement(exterior_polygon=create_test_polygon_coords(), holes="not a list", cell_id="test")

@pytest.mark.default
def test_init_invalid_hole_element_type():
    """A hole that is not a PolygonCoords raises TypeError."""
    exterior = create_test_polygon_coords()
    holes = [create_test_hole_coords(), "not a PolygonCoords"]
    with pytest.raises(TypeError):
        PolygonElement(exterior_polygon=exterior, holes=holes, cell_id="test")

@pytest.mark.default
def test_translate():
    """Translating a polygon shifts both the exterior and the holes."""
    exterior = create_test_polygon_coords()
    holes = [create_test_hole_coords()]
    pe = PolygonElement(exterior_polygon=exterior, holes=holes, cell_id="test")
    pe.translate(1.0, 2.0)
    # Check exterior
    assert np.array_equal(pe.exterior_polygon.x_coords, np.array([1.0, 2.0, 2.0, 1.0]))
    assert np.array_equal(pe.exterior_polygon.y_coords, np.array([2.0, 2.0, 3.0, 3.0]))
    # Check hole
    assert np.array_equal(pe.holes[0].x_coords, np.array([1.25, 1.75, 1.75, 1.25]))
    assert np.array_equal(pe.holes[0].y_coords, np.array([2.25, 2.25, 2.75, 2.75]))

@pytest.mark.default
def test_rotate():
    """Rotating a polygon rotates both the exterior and the holes."""
    exterior = create_test_polygon_coords()
    holes = [create_test_hole_coords()]
    pe = PolygonElement(exterior_polygon=exterior, holes=holes, cell_id="test")
    pe.rotate(origin=(0.5, 0.5), angle=np.pi / 2)  # 90 deg CCW
    # Expected: (0,0) -> (1,0), (1,0) -> (1,1), (1,1) -> (0,1), (0,1) -> (0,0)
    expected_exterior_x = np.array([1.0, 1.0, 0.0, 0.0])
    expected_exterior_y = np.array([0.0, 1.0, 1.0, 0.0])
    assert np.allclose(pe.exterior_polygon.x_coords, expected_exterior_x, atol=1e-10)
    assert np.allclose(pe.exterior_polygon.y_coords, expected_exterior_y, atol=1e-10)
    # Hole should rotate similarly
    expected_hole_x = np.array([0.75, 0.75, 0.25, 0.25])
    expected_hole_y = np.array([0.25, 0.75, 0.75, 0.25])
    assert np.allclose(pe.holes[0].x_coords, expected_hole_x, atol=1e-10)
    assert np.allclose(pe.holes[0].y_coords, expected_hole_y, atol=1e-10)