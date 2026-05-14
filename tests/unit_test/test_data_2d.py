from typing import Tuple

import pytest
import numpy as np
from scivianna.data.data2d import Data2D
from scivianna.enums import DataType


# Helper function to create a simple Data2D with specific cell_ids and values
def create_data2d(cell_ids: list, values: list, default_color: Tuple[int, int, int] = (100, 100, 100), default_edge_color: Tuple[int, int, int] = (50, 50, 50)) -> Data2D:
    """Create a Data2D object with given cell_ids and values.
    
    Parameters
    ----------
    cell_ids : list
        List of cell ids
    values : list
        List of cell values
    default_color : tuple
        Default color for all cells (overridable via cell_colors attribute)
    default_edge_color : tuple
        Default edge color for all cells (overridable via cell_edge_colors attribute)
    """
    data = Data2D()
    data.data_type = DataType.POLYGONS
    data.cell_ids = cell_ids.copy()
    data.cell_values = values.copy()
    data.cell_colors = [default_color] * len(cell_ids)
    data.cell_edge_colors = [default_edge_color] * len(cell_ids)
    data.polygons = []
    return data


# ============================================================
# Tests for _reorder_data2d_to_match
# ============================================================

@pytest.mark.default
class TestReorderData2D:
    def test_reorder_same_order(self):
        """Test reordering when cell_ids are already in the same order."""
        data = create_data2d([1, 2, 3], [10.0, 20.0, 30.0])
        result = Data2D._reorder_data2d_to_match(data, [1, 2, 3])
        assert result.cell_ids == [1, 2, 3]
        assert result.cell_values == [10.0, 20.0, 30.0]

    def test_reorder_different_order(self):
        """Test reordering when cell_ids are in a different order."""
        data = create_data2d([1, 2, 3], [10.0, 20.0, 30.0])
        result = Data2D._reorder_data2d_to_match(data, [3, 1, 2])
        assert result.cell_ids == [3, 1, 2]
        assert result.cell_values == [30.0, 10.0, 20.0]

    def test_reorder_subset(self):
        """Test reordering with a subset of cell_ids."""
        data = create_data2d([1, 2, 3], [10.0, 20.0, 30.0])
        result = Data2D._reorder_data2d_to_match(data, [2, 1])
        assert result.cell_ids == [2, 1]
        assert result.cell_values == [20.0, 10.0]

    def test_reorder_superset(self):
        """Test reordering with a superset of cell_ids (some missing in original)."""
        data = create_data2d([1, 2], [10.0, 20.0])
        result = Data2D._reorder_data2d_to_match(data, [1, 2, 3])
        assert result.cell_ids == [1, 2, 3]
        assert result.cell_values == [10.0, 20.0, np.nan]

    def test_reorder_string_cell_ids(self):
        """Test reordering with string cell_ids."""
        data = create_data2d(["a", "b", "c"], [1.0, 2.0, 3.0])
        result = Data2D._reorder_data2d_to_match(data, ["c", "a"])
        assert result.cell_ids == ["c", "a"]
        assert result.cell_values == [3.0, 1.0]


# ============================================================
# Tests for __add__
# ============================================================

@pytest.mark.default
class TestAdd:
    def test_add_scalar(self):
        """Test adding a float to Data2D."""
        data = create_data2d([1, 2, 3], [10.0, 20.0, 30.0])
        result = data + 5.0
        assert result.cell_ids == [1, 2, 3]
        assert result.cell_values == [15.0, 25.0, 35.0]

    def test_add_same_cell_ids(self):
        """Test adding two Data2D with same cell_ids and same order."""
        data1 = create_data2d([1, 2, 3], [10.0, 20.0, 30.0])
        data2 = create_data2d([1, 2, 3], [1.0, 2.0, 3.0])
        result = data1 + data2
        assert result.cell_ids == [1, 2, 3]
        assert result.cell_values == [11.0, 22.0, 33.0]

    def test_add_different_order(self):
        """Test adding two Data2D with same cell_ids but different order."""
        data1 = create_data2d([1, 2, 3], [10.0, 20.0, 30.0])
        data2 = create_data2d([3, 2, 1], [1.0, 2.0, 3.0])
        result = data1 + data2
        # Union order: [1, 2, 3] (from data1)
        assert result.cell_ids == [1, 2, 3]
        # After reordering data2: values should be [3.0, 2.0, 1.0]
        assert result.cell_values == [13.0, 22.0, 31.0]

    def test_add_different_cell_ids(self):
        """Test adding two Data2D with different cell_ids."""
        data1 = create_data2d([1, 2], [10.0, 20.0])
        data2 = create_data2d([2, 3], [2.0, 3.0])
        result = data1 + data2
        # Union order: [1, 2, 3]
        assert result.cell_ids == [1, 2, 3]
        # Cell 1: 10 + nan = nan, Cell 2: 20 + 2 = 22, Cell 3: nan + 3 = nan
        assert result.cell_values[1] == 22.0
        assert np.isnan(result.cell_values[0])
        assert np.isnan(result.cell_values[2])

    def test_add_float_on_left(self):
        """Test adding a float on the left side."""
        data = create_data2d([1, 2, 3], [10.0, 20.0, 30.0])
        result = 5.0 + data
        assert result.cell_ids == [1, 2, 3]
        assert result.cell_values == [15.0, 25.0, 35.0]

    def test_add_in_place(self):
        """Test in-place addition (+=)."""
        data = create_data2d([1, 2, 3], [10.0, 20.0, 30.0])
        data += 5.0
        assert data.cell_values == [15.0, 25.0, 35.0]


# ============================================================
# Tests for __sub__
# ============================================================

@pytest.mark.default
class TestSub:
    def test_sub_scalar(self):
        """Test subtracting a float from Data2D."""
        data = create_data2d([1, 2, 3], [10.0, 20.0, 30.0])
        result = data - 5.0
        assert result.cell_ids == [1, 2, 3]
        assert result.cell_values == [5.0, 15.0, 25.0]

    def test_sub_same_cell_ids(self):
        """Test subtracting two Data2D with same cell_ids and same order."""
        data1 = create_data2d([1, 2, 3], [10.0, 20.0, 30.0])
        data2 = create_data2d([1, 2, 3], [1.0, 2.0, 3.0])
        result = data1 - data2
        assert result.cell_ids == [1, 2, 3]
        assert result.cell_values == [9.0, 18.0, 27.0]

    def test_sub_different_order(self):
        """Test subtracting two Data2D with same cell_ids but different order."""
        data1 = create_data2d([1, 2, 3], [10.0, 20.0, 30.0])
        data2 = create_data2d([3, 2, 1], [1.0, 2.0, 3.0])
        result = data1 - data2
        # Union order: [1, 2, 3] (from data1)
        assert result.cell_ids == [1, 2, 3]
        # After reordering data2: values should be [3.0, 2.0, 1.0]
        assert result.cell_values == [7.0, 18.0, 29.0]

    def test_sub_different_cell_ids(self):
        """Test subtracting two Data2D with different cell_ids."""
        data1 = create_data2d([1, 2], [10.0, 20.0])
        data2 = create_data2d([2, 3], [2.0, 3.0])
        result = data1 - data2
        # Union order: [1, 2, 3]
        assert result.cell_ids == [1, 2, 3]
        assert result.cell_values[1] == 18.0
        assert np.isnan(result.cell_values[0])
        assert np.isnan(result.cell_values[2])

    def test_rsub(self):
        """Test reverse subtraction (float - Data2D)."""
        data = create_data2d([1, 2, 3], [10.0, 20.0, 30.0])
        result = 100.0 - data
        assert result.cell_ids == [1, 2, 3]
        assert result.cell_values == [90.0, 80.0, 70.0]

    def test_sub_in_place(self):
        """Test in-place subtraction (-=)."""
        data = create_data2d([1, 2, 3], [10.0, 20.0, 30.0])
        data -= 5.0
        assert data.cell_values == [5.0, 15.0, 25.0]


# ============================================================
# Tests for __mul__
# ============================================================

@pytest.mark.default
class TestMul:
    def test_mul_scalar(self):
        """Test multiplying Data2D by a float."""
        data = create_data2d([1, 2, 3], [10.0, 20.0, 30.0])
        result = data * 2.0
        assert result.cell_ids == [1, 2, 3]
        assert result.cell_values == [20.0, 40.0, 60.0]

    def test_mul_same_cell_ids(self):
        """Test multiplying two Data2D with same cell_ids."""
        data1 = create_data2d([1, 2, 3], [10.0, 20.0, 30.0])
        data2 = create_data2d([1, 2, 3], [2.0, 3.0, 4.0])
        result = data1 * data2
        assert result.cell_ids == [1, 2, 3]
        assert result.cell_values == [20.0, 60.0, 120.0]

    def test_mul_different_order(self):
        """Test multiplying two Data2D with same cell_ids but different order."""
        data1 = create_data2d([1, 2, 3], [10.0, 20.0, 30.0])
        data2 = create_data2d([3, 2, 1], [1.0, 2.0, 3.0])
        result = data1 * data2
        # Union order: [1, 2, 3] (from data1)
        assert result.cell_ids == [1, 2, 3]
        # After reordering data2: values should be [3.0, 2.0, 1.0]
        assert result.cell_values == [30.0, 40.0, 30.0]

    def test_mul_different_cell_ids(self):
        """Test multiplying two Data2D with different cell_ids."""
        data1 = create_data2d([1, 2], [10.0, 20.0])
        data2 = create_data2d([2, 3], [2.0, 3.0])
        result = data1 * data2
        # Union order: [1, 2, 3]
        assert result.cell_ids == [1, 2, 3]
        assert result.cell_values[1] == 40.0
        assert np.isnan(result.cell_values[0])
        assert np.isnan(result.cell_values[2])

    def test_mul_int_scalar(self):
        """Test multiplying Data2D by an int."""
        data = create_data2d([1, 2, 3], [10.0, 20.0, 30.0])
        result = data * 3
        assert result.cell_ids == [1, 2, 3]
        assert result.cell_values == [30.0, 60.0, 90.0]

    def test_rmul(self):
        """Test reverse multiplication (float * Data2D)."""
        data = create_data2d([1, 2, 3], [10.0, 20.0, 30.0])
        result = 2.0 * data
        assert result.cell_ids == [1, 2, 3]
        assert result.cell_values == [20.0, 40.0, 60.0]

    def test_mul_in_place(self):
        """Test in-place multiplication (*=)."""
        data = create_data2d([1, 2, 3], [10.0, 20.0, 30.0])
        data *= 2.0
        assert data.cell_values == [20.0, 40.0, 60.0]


# ============================================================
# Tests for __truediv__
# ============================================================

@pytest.mark.default
class TestDiv:
    def test_div_scalar(self):
        """Test dividing Data2D by a float."""
        data = create_data2d([1, 2, 3], [10.0, 20.0, 30.0])
        result = data / 2.0
        assert result.cell_ids == [1, 2, 3]
        assert result.cell_values == [5.0, 10.0, 15.0]

    def test_div_same_cell_ids(self):
        """Test dividing two Data2D with same cell_ids."""
        data1 = create_data2d([1, 2, 3], [10.0, 20.0, 30.0])
        data2 = create_data2d([1, 2, 3], [2.0, 4.0, 6.0])
        result = data1 / data2
        assert result.cell_ids == [1, 2, 3]
        assert result.cell_values == [5.0, 5.0, 5.0]

    def test_div_different_order(self):
        """Test dividing two Data2D with same cell_ids but different order."""
        data1 = create_data2d([1, 2, 3], [10.0, 20.0, 30.0])
        data2 = create_data2d([3, 2, 1], [1.0, 2.0, 3.0])
        result = data1 / data2
        # Union order: [1, 2, 3] (from data1)
        assert result.cell_ids == [1, 2, 3]
        # After reordering data2: values should be [3.0, 2.0, 1.0]
        assert result.cell_values == [10.0 / 3.0, 20.0 / 2.0, 30.0 / 1.0]

    def test_div_by_zero(self):
        """Test dividing by zero returns nan."""
        data = create_data2d([1, 2, 3], [10.0, 20.0, 30.0])
        result = data / 0.0
        assert all(np.isnan(v) for v in result.cell_values)

    def test_div_zero_values_in_data(self):
        """Test dividing Data2D containing zero values."""
        data1 = create_data2d([1, 2, 3], [10.0, 0.0, 30.0])
        data2 = create_data2d([1, 2, 3], [2.0, 0.0, 6.0])
        result = data1 / data2
        assert not np.isnan(result.cell_values[0])  # 10/2 = 5
        assert np.isnan(result.cell_values[1])  # 0/0 = nan
        assert not np.isnan(result.cell_values[2])  # 30/6 = 5

    def test_div_different_cell_ids(self):
        """Test dividing two Data2D with different cell_ids."""
        data1 = create_data2d([1, 2], [10.0, 20.0])
        data2 = create_data2d([2, 3], [2.0, 5.0])
        result = data1 / data2
        # Union order: [1, 2, 3]
        assert result.cell_ids == [1, 2, 3]
        assert result.cell_values[1] == 10.0  # 20/2
        assert np.isnan(result.cell_values[0])
        assert np.isnan(result.cell_values[2])

    def test_rdiv(self):
        """Test reverse division (float / Data2D)."""
        data = create_data2d([1, 2, 3], [2.0, 4.0, 5.0])
        result = 20.0 / data
        assert result.cell_ids == [1, 2, 3]
        assert result.cell_values == [10.0, 5.0, 4.0]

    def test_rdiv_by_zero(self):
        """Test reverse division by zero returns nan."""
        data = create_data2d([1], [0.0])
        result = 20.0 / data
        assert np.isnan(result.cell_values[0])

    def test_div_in_place(self):
        """Test in-place division (/=)."""
        data = create_data2d([1, 2, 3], [10.0, 20.0, 30.0])
        data /= 2.0
        assert data.cell_values == [5.0, 10.0, 15.0]


# ============================================================
# Tests for type safety and edge cases
# ============================================================

@pytest.mark.default
class TestEdgeCases:
    def test_unsupported_type(self):
        """Test that unsupported operand types raise TypeError."""
        data = create_data2d([1, 2, 3], [10.0, 20.0, 30.0])
        with pytest.raises(TypeError):
            data + "invalid"

    def test_empty_data2d_add(self):
        """Test operations on empty Data2D."""
        data1 = create_data2d([], [])
        data2 = create_data2d([1, 2], [10.0, 20.0])
        result = data1 + data2
        assert result.cell_ids == [1, 2]
        assert np.isnan(result.cell_values[0])
        assert np.isnan(result.cell_values[1])

    def test_single_cell_operations(self):
        """Test operations with single cell."""
        data = create_data2d([1], [10.0])
        result = data + 5.0
        assert result.cell_values == [15.0]

    def test_result_has_default_colors_for_data2d_operation(self):
        """Test that result has default white colors when operating two Data2D objects."""
        data1 = create_data2d([1, 2], [1.0, 2.0], default_color=(100, 100, 100))
        data2 = create_data2d([1, 2], [3.0, 4.0], default_color=(200, 200, 200))
        # Manually set different colors to test they are NOT preserved
        data1.cell_colors = [[100, 100, 100], [150, 150, 150]]
        data2.cell_colors = [[200, 200, 200], [250, 250, 250]]
        result = data1 + data2
        # Result should have white colors for cells and (200, 200, 200, 255) for edges (as tuples)
        assert result.cell_colors == [(255, 255, 255), (255, 255, 255)]
        assert result.cell_edge_colors == [(200, 200, 200, 255), (200, 200, 200, 255)]

    def test_result_preserves_colors_for_scalar_operation(self):
        """Test that scalar operations preserve colors from the original Data2D."""
        data = create_data2d([1, 2], [1.0, 2.0], default_color=(100, 100, 100))
        data.cell_colors = [[100, 100, 100], [150, 150, 150]]
        data.cell_edge_colors = [[50, 50, 50, 255], [75, 75, 75, 255]]
        result = data + 5.0
        # Colors are preserved as lists after copy()
        assert result.cell_colors == [[100, 100, 100], [150, 150, 150]]
        assert result.cell_edge_colors == [[50, 50, 50, 255], [75, 75, 75, 255]]

    def test_original_data_unchanged(self):
        """Test that original Data2D is not modified by operations."""
        data = create_data2d([1, 2, 3], [10.0, 20.0, 30.0])
        _ = data + 5.0
        assert data.cell_values == [10.0, 20.0, 30.0]

    def test_integer_cell_ids(self):
        """Test with integer cell_ids."""
        data1 = create_data2d([1, 2, 3], [10.0, 20.0, 30.0])
        data2 = create_data2d([3, 1, 2], [1.0, 2.0, 3.0])
        result = data1 + data2
        assert result.cell_ids == [1, 2, 3]
        assert result.cell_values == [12.0, 23.0, 31.0]

    def test_mixed_string_cell_ids(self):
        """Test with string cell_ids."""
        data1 = create_data2d(["cell_a", "cell_b"], [10.0, 20.0])
        data2 = create_data2d(["cell_b", "cell_a"], [1.0, 2.0])
        result = data1 + data2
        assert result.cell_ids == ["cell_a", "cell_b"]
        assert result.cell_values == [12.0, 21.0]