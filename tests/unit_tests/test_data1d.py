"""Tests for scivianna.data.data1d module."""

import pytest
import numpy as np
import pandas as pd


class TestData1D:
    """Test suite for Data1D class."""

    @pytest.fixture
    def data1d(self):
        """Create an empty Data1D instance for testing."""
        from scivianna.data.data1d import Data1D
        return Data1D()

    @pytest.fixture
    def data1d_with_data(self):
        """Create a Data1D instance with pre-populated data via from_serie_dict."""
        from scivianna.data.data1d import Data1D
        
        series = {
            'line_a': pd.Series([1.0, 2.0, 3.0]),
            'line_b': pd.Series([4.0, 5.0, 6.0]),
        }
        
        return Data1D.from_serie_dict(series)

    # ---- __init__ tests ----

    def test_init_empty_data1d(self, data1d):
        """Test default initialization creates empty Data1D."""
        assert data1d.line_ids == []
        assert data1d.line_values == []
        assert data1d.line_colors == []
        assert data1d.line_styles == []

    def test_init_attribute_types(self, data1d):
        """Test that initialized attributes are correct types (lists)."""
        assert isinstance(data1d.line_ids, list)
        assert isinstance(data1d.line_values, list)
        assert isinstance(data1d.line_colors, list)
        assert isinstance(data1d.line_styles, list)

    def test_init_independent_instances(self):
        """Test that two Data1D instances are independent."""
        from scivianna.data.data1d import Data1D
        
        d1 = Data1D()
        d2 = Data1D()
        
        d1.line_ids.append('test')
        
        assert d2.line_ids == []

    # ---- from_dataframe tests ----

    def test_from_dataframe_basic(self):
        """Test from_dataframe with basic dataframe."""
        from scivianna.data.data1d import Data1D
        
        df = pd.DataFrame({
            'x': [1.0, 2.0, 3.0],
            'y': [4.0, 5.0, 6.0],
        })
        
        data1d = Data1D.from_dataframe(df)
        
        assert data1d.line_ids == ['x', 'y']
        assert len(data1d.line_values) == 2
        assert isinstance(data1d.line_values[0], pd.Series)
        assert isinstance(data1d.line_values[1], pd.Series)
        pd.testing.assert_series_equal(data1d.line_values[0], df['x'])
        pd.testing.assert_series_equal(data1d.line_values[1], df['y'])

    def test_from_dataframe_colors_styles_none(self):
        """Test from_dataframe sets colors and styles to None."""
        from scivianna.data.data1d import Data1D
        
        df = pd.DataFrame({
            'a': [1.0, 2.0],
            'b': [3.0, 4.0],
            'c': [5.0, 6.0],
        })
        
        data1d = Data1D.from_dataframe(df)
        
        assert data1d.line_colors == [None, None, None]
        assert data1d.line_styles == [None, None, None]

    def test_from_dataframe_single_column(self):
        """Test from_dataframe with single column dataframe."""
        from scivianna.data.data1d import Data1D
        
        df = pd.DataFrame({'only_col': [10.0, 20.0, 30.0]})
        
        data1d = Data1D.from_dataframe(df)
        
        assert data1d.line_ids == ['only_col']
        assert len(data1d.line_values) == 1
        assert data1d.line_colors == [None]
        assert data1d.line_styles == [None]

    def test_from_dataframe_empty_dataframe(self):
        """Test from_dataframe with empty dataframe."""
        from scivianna.data.data1d import Data1D
        
        df = pd.DataFrame()
        
        data1d = Data1D.from_dataframe(df)
        
        assert data1d.line_ids == []
        assert data1d.line_values == []
        assert data1d.line_colors == []
        assert data1d.line_styles == []

    def test_from_dataframe_preserves_series_index(self):
        """Test from_dataframe preserves Series index."""
        from scivianna.data.data1d import Data1D
        
        df = pd.DataFrame({
            'col1': [100.0, 200.0],
        }, index=[0.5, 1.5])
        
        data1d = Data1D.from_dataframe(df)
        
        assert data1d.line_values[0].index.tolist() == [0.5, 1.5]

    def test_from_dataframe_numeric_column_names(self):
        """Test from_dataframe with numeric column names."""
        from scivianna.data.data1d import Data1D
        
        df = pd.DataFrame({
            1: [10.0, 20.0],
            2: [30.0, 40.0],
        })
        
        data1d = Data1D.from_dataframe(df)
        
        assert 1 in data1d.line_ids
        assert 2 in data1d.line_ids

    def test_from_dataframe_mixed_column_types(self):
        """Test from_dataframe with mixed column name types."""
        from scivianna.data.data1d import Data1D
        
        df = pd.DataFrame({
            'string_col': [1.0, 2.0],
            123: [3.0, 4.0],
        })
        
        data1d = Data1D.from_dataframe(df)
        
        assert 'string_col' in data1d.line_ids
        assert 123 in data1d.line_ids

    def test_from_dataframe_returns_data1d_instance(self):
        """Test from_dataframe returns Data1D instance."""
        from scivianna.data.data1d import Data1D
        
        df = pd.DataFrame({'a': [1.0]})
        result = Data1D.from_dataframe(df)
        
        assert isinstance(result, Data1D)

    # ---- from_serie_dict tests ----

    def test_from_serie_dict_basic(self):
        """Test from_serie_dict with valid dict of Series."""
        from scivianna.data.data1d import Data1D
        
        series = {
            'line_1': pd.Series([1.0, 2.0, 3.0]),
            'line_2': pd.Series([4.0, 5.0, 6.0]),
        }
        
        data1d = Data1D.from_serie_dict(series)
        
        assert data1d.line_ids == ['line_1', 'line_2']
        assert len(data1d.line_values) == 2
        pd.testing.assert_series_equal(data1d.line_values[0], series['line_1'])
        pd.testing.assert_series_equal(data1d.line_values[1], series['line_2'])

    def test_from_serie_dict_colors_styles_none(self):
        """Test from_serie_dict sets colors and styles to None."""
        from scivianna.data.data1d import Data1D
        
        series = {
            'a': pd.Series([1.0]),
            'b': pd.Series([2.0]),
        }
        
        data1d = Data1D.from_serie_dict(series)
        
        assert data1d.line_colors == [None, None]
        assert data1d.line_styles == [None, None]

    def test_from_serie_dict_empty_dict(self):
        """Test from_serie_dict with empty dict."""
        from scivianna.data.data1d import Data1D
        
        data1d = Data1D.from_serie_dict({})
        
        assert data1d.line_ids == []
        assert data1d.line_values == []
        assert data1d.line_colors == []
        assert data1d.line_styles == []

    def test_from_serie_dict_single_series(self):
        """Test from_serie_dict with single Series."""
        from scivianna.data.data1d import Data1D
        
        series = {'only_one': pd.Series([42.0, 84.0])}
        
        data1d = Data1D.from_serie_dict(series)
        
        assert data1d.line_ids == ['only_one']
        assert len(data1d.line_values) == 1
        assert data1d.line_colors == [None]
        assert data1d.line_styles == [None]

    def test_from_serie_dict_non_dict_raises_assertion(self):
        """Test from_serie_dict raises AssertionError for non-dict input."""
        from scivianna.data.data1d import Data1D
        
        with pytest.raises(AssertionError, match="must be a dictionnary"):
            Data1D.from_serie_dict([1.0, 2.0])

    def test_from_serie_dict_list_raises(self):
        """Test from_serie_dict raises AssertionError for list input."""
        from scivianna.data.data1d import Data1D
        
        with pytest.raises(AssertionError):
            Data1D.from_serie_dict([[1.0, 2.0]])

    def test_from_serie_dict_series_raises(self):
        """Test from_serie_dict raises AssertionError for Series input."""
        from scivianna.data.data1d import Data1D
        
        with pytest.raises(AssertionError):
            Data1D.from_serie_dict(pd.Series([1.0, 2.0]))

    def test_from_serie_dict_preserves_series_index(self):
        """Test from_serie_dict preserves Series index."""
        from scivianna.data.data1d import Data1D
        
        series = {
            'x': pd.Series([10.0, 20.0], index=[0.1, 0.2]),
        }
        
        data1d = Data1D.from_serie_dict(series)
        
        assert data1d.line_values[0].index.tolist() == [0.1, 0.2]

    def test_from_serie_dict_preserves_order(self):
        """Test from_serie_dict preserves dict key order."""
        from scivianna.data.data1d import Data1D
        
        series = {
            'z_first': pd.Series([1.0]),
            'a_second': pd.Series([2.0]),
            'm_middle': pd.Series([3.0]),
        }
        
        data1d = Data1D.from_serie_dict(series)
        
        assert data1d.line_ids == ['z_first', 'a_second', 'm_middle']

    def test_from_serie_dict_returns_data1d_instance(self):
        """Test from_serie_dict returns Data1D instance."""
        from scivianna.data.data1d import Data1D
        
        series = {'a': pd.Series([1.0])}
        result = Data1D.from_serie_dict(series)
        
        assert isinstance(result, Data1D)

    def test_from_serie_dict_integer_keys(self):
        """Test from_serie_dict with integer keys."""
        from scivianna.data.data1d import Data1D
        
        series = {
            1: pd.Series([10.0]),
            2: pd.Series([20.0]),
        }
        
        data1d = Data1D.from_serie_dict(series)
        
        assert 1 in data1d.line_ids
        assert 2 in data1d.line_ids

    def test_from_serie_dict_none_value(self):
        """Test from_serie_dict raises for None input."""
        from scivianna.data.data1d import Data1D
        
        with pytest.raises(AssertionError):
            Data1D.from_serie_dict(None)

    # ---- copy tests ----

    def test_copy_empty(self):
        """Test copy method on empty Data1D."""
        from scivianna.data.data1d import Data1D
        
        original = Data1D()
        copied = original.copy()
        
        # After copy, attributes become numpy arrays
        assert copied.line_ids.size == 0
        assert copied.line_values.size == 0 or len(copied.line_values) == 0
        assert copied.line_colors.size == 0 or len(copied.line_colors) == 0
        assert copied.line_styles.size == 0 or len(copied.line_styles) == 0

    def test_copy_preserves_data(self, data1d_with_data):
        """Test copy preserves all data."""
        copied = data1d_with_data.copy()
        
        # line_ids becomes numpy array after copy
        assert isinstance(copied.line_ids, np.ndarray)
        assert len(copied.line_ids) == len(data1d_with_data.line_ids)
        
        # line_values becomes numpy array after copy
        assert isinstance(copied.line_values, np.ndarray)
        assert len(copied.line_values) == len(data1d_with_data.line_values)
        
        # line_colors and line_styles also become numpy arrays
        assert isinstance(copied.line_colors, np.ndarray)
        assert isinstance(copied.line_styles, np.ndarray)

    def test_copy_creates_new_instance(self, data1d_with_data):
        """Test copy creates a new independent instance."""
        copied = data1d_with_data.copy()
        
        assert copied is not data1d_with_data

    def test_copy_line_ids_as_array(self):
        """Test that copy converts line_ids to numpy array."""
        from scivianna.data.data1d import Data1D
        
        series = {'a': pd.Series([1.0])}
        original = Data1D.from_serie_dict(series)
        copied = original.copy()
        
        # copy() uses np.array() on line_ids
        assert isinstance(copied.line_ids, np.ndarray)

    def test_copy_line_values_as_array(self):
        """Test that copy converts line_values to numpy array."""
        from scivianna.data.data1d import Data1D
        
        series = {'a': pd.Series([1.0])}
        original = Data1D.from_serie_dict(series)
        copied = original.copy()
        
        # copy() uses np.array() on line_values
        assert isinstance(copied.line_values, np.ndarray)

    def test_copy_line_colors_as_array(self):
        """Test that copy converts line_colors to numpy array."""
        from scivianna.data.data1d import Data1D
        
        series = {'a': pd.Series([1.0])}
        original = Data1D.from_serie_dict(series)
        copied = original.copy()
        
        assert isinstance(copied.line_colors, np.ndarray)

    def test_copy_line_styles_as_array(self):
        """Test that copy converts line_styles to numpy array."""
        from scivianna.data.data1d import Data1D
        
        series = {'a': pd.Series([1.0])}
        original = Data1D.from_serie_dict(series)
        copied = original.copy()
        
        assert isinstance(copied.line_styles, np.ndarray)

    def test_copy_returns_data1d(self):
        """Test copy returns Data1D instance."""
        from scivianna.data.data1d import Data1D
        
        original = Data1D()
        copied = original.copy()
        
        assert isinstance(copied, Data1D)

    # ---- check_valid tests ----

    def test_check_valid_basic(self):
        """Test check_valid passes for valid Data1D."""
        from scivianna.data.data1d import Data1D
        
        series = {
            'a': pd.Series([1.0]),
            'b': pd.Series([2.0]),
        }
        
        data1d = Data1D.from_serie_dict(series)
        
        # Should not raise
        data1d.check_valid()

    def test_check_valid_empty(self):
        """Test check_valid passes for empty Data1D."""
        from scivianna.data.data1d import Data1D
        
        data1d = Data1D()
        
        # Should not raise
        data1d.check_valid()

    def test_check_valid_mismatched_colors_raises(self):
        """Test check_valid raises when line_ids and line_colors lengths differ."""
        from scivianna.data.data1d import Data1D
        
        series = {
            'a': pd.Series([1.0]),
            'b': pd.Series([2.0]),
        }
        
        data1d = Data1D.from_serie_dict(series)
        # Manually mismatch the lengths
        data1d.line_colors = [None]  # Only 1 color for 2 lines
        
        with pytest.raises(AssertionError, match="same number of line id and colors"):
            data1d.check_valid()

    def test_check_valid_mismatched_values_colors_raises(self):
        """Test check_valid raises when line_values and line_colors lengths differ."""
        from scivianna.data.data1d import Data1D
        
        series = {
            'a': pd.Series([1.0]),
            'b': pd.Series([2.0]),
        }
        
        data1d = Data1D.from_serie_dict(series)
        # Set matching line_ids and line_colors (2 each), but mismatch values vs colors
        data1d.line_colors = [None, None, None]  # 3 colors for 2 values
        
        with pytest.raises(AssertionError, match="same number of line id and colors"):
            data1d.check_valid()

    def test_check_valid_mismatched_values_styles_raises(self):
        """Test check_valid raises when line_values and line_styles lengths differ."""
        from scivianna.data.data1d import Data1D
        
        series = {
            'a': pd.Series([1.0]),
            'b': pd.Series([2.0]),
        }
        
        data1d = Data1D.from_serie_dict(series)
        # Manually mismatch the lengths
        data1d.line_styles = [None, None, None]  # 3 styles for 2 lines
        
        with pytest.raises(AssertionError, match="same number of line values and styles"):
            data1d.check_valid()

    def test_check_valid_all_matching(self):
        """Test check_valid passes when all lists have matching lengths."""
        from scivianna.data.data1d import Data1D
        
        series = {
            'a': pd.Series([1.0]),
            'b': pd.Series([2.0]),
            'c': pd.Series([3.0]),
        }
        
        data1d = Data1D.from_serie_dict(series)
        # Set matching colors and styles
        data1d.line_colors = [None, None, None]
        data1d.line_styles = ['-', '--', '-.']
        
        # Should not raise
        data1d.check_valid()

    # ---- from_dataframe edge cases ----

    def test_from_dataframe_with_nan(self):
        """Test from_dataframe handles NaN values."""
        from scivianna.data.data1d import Data1D
        
        df = pd.DataFrame({
            'with_nan': [1.0, np.nan, 3.0],
        })
        
        data1d = Data1D.from_dataframe(df)
        
        assert len(data1d.line_ids) == 1
        assert pd.isna(data1d.line_values[0].iloc[1])

    def test_from_dataframe_with_negative(self):
        """Test from_dataframe handles negative values."""
        from scivianna.data.data1d import Data1D
        
        df = pd.DataFrame({
            'neg': [-1.0, -2.0, -3.0],
        })
        
        data1d = Data1D.from_dataframe(df)
        
        assert data1d.line_values[0].iloc[0] == -1.0

    def test_from_dataframe_many_columns(self):
        """Test from_dataframe with many columns."""
        from scivianna.data.data1d import Data1D
        
        data = {f'col_{i}': [float(i)] for i in range(50)}
        df = pd.DataFrame(data)
        
        data1d = Data1D.from_dataframe(df)
        
        assert len(data1d.line_ids) == 50
        assert len(data1d.line_values) == 50
        assert len(data1d.line_colors) == 50
        assert len(data1d.line_styles) == 50

    # ---- from_serie_dict edge cases ----

    def test_from_serie_dict_mixed_dtypes(self):
        """Test from_serie_dict with mixed Series dtypes."""
        from scivianna.data.data1d import Data1D
        
        series = {
            'int_series': pd.Series([1, 2, 3]),
            'float_series': pd.Series([1.0, 2.0, 3.0]),
        }
        
        data1d = Data1D.from_serie_dict(series)
        
        assert len(data1d.line_ids) == 2

    def test_from_serie_dict_datetime_index(self):
        """Test from_serie_dict with datetime index."""
        from scivianna.data.data1d import Data1D
        
        dates = pd.date_range('2024-01-01', periods=3)
        series = {
            'time_series': pd.Series([1.0, 2.0, 3.0], index=dates),
        }
        
        data1d = Data1D.from_serie_dict(series)
        
        assert len(data1d.line_values[0]) == 3
        assert isinstance(data1d.line_values[0].index, pd.DatetimeIndex)

    def test_from_serie_dict_boolean_series(self):
        """Test from_serie_dict with boolean Series."""
        from scivianna.data.data1d import Data1D
        
        series = {
            'bool_col': pd.Series([True, False, True]),
        }
        
        data1d = Data1D.from_serie_dict(series)
        
        assert len(data1d.line_values[0]) == 3

    def test_from_serie_dict_large_dict(self):
        """Test from_serie_dict with many Series."""
        from scivianna.data.data1d import Data1D
        
        series = {f'serie_{i}': pd.Series([float(i)]) for i in range(100)}
        
        data1d = Data1D.from_serie_dict(series)
        
        assert len(data1d.line_ids) == 100

    # ---- copy edge cases ----

    def test_copy_from_from_dataframe(self):
        """Test copy works on Data1D created from from_dataframe."""
        from scivianna.data.data1d import Data1D
        
        df = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
        original = Data1D.from_dataframe(df)
        copied = original.copy()
        
        assert len(copied.line_values) == 2

    def test_copy_deep_series_data(self):
        """Test that Series data is accessible after copy."""
        from scivianna.data.data1d import Data1D
        
        series = {
            'a': pd.Series([1.0, 2.0, 3.0]),
        }
        
        original = Data1D.from_serie_dict(series)
        copied = original.copy()
        
        # The Series inside should still be accessible and valid
        assert len(copied.line_values[0]) == 3

    def test_copy_preserves_none_colors(self):
        """Test copy preserves None values in colors."""
        from scivianna.data.data1d import Data1D
        
        series = {'a': pd.Series([1.0])}
        original = Data1D.from_serie_dict(series)
        copied = original.copy()
        
        # After np.array([None]), the value should still be None-like
        assert copied.line_colors[0] is None or str(copied.line_colors[0]) == 'None'


class TestData1DIntegration:
    """Integration tests for Data1D class."""

    def test_full_workflow(self):
        """Test a complete workflow: create, copy, validate."""
        from scivianna.data.data1d import Data1D
        
        # Create
        series = {
            'x': pd.Series([1.0, 2.0, 3.0]),
            'y': pd.Series([4.0, 5.0, 6.0]),
        }
        data1d = Data1D.from_serie_dict(series)
        
        # Validate
        data1d.check_valid()
        
        # Copy
        copied = data1d.copy()
        copied.check_valid()
        
        # Verify data
        assert len(copied.line_ids) == 2

    def test_dataframe_to_serie_dict_conversion(self):
        """Test round-trip via dataframe and serie_dict."""
        from scivianna.data.data1d import Data1D
        
        # From dataframe
        df = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
        data1d_df = Data1D.from_dataframe(df)
        
        # From serie_dict  
        series = {'a': pd.Series([1.0, 2.0]), 'b': pd.Series([3.0, 4.0])}
        data1d_series = Data1D.from_serie_dict(series)
        
        # Should be equivalent
        assert data1d_df.line_ids == data1d_series.line_ids
        assert len(data1d_df.line_values) == len(data1d_series.line_values)

    def test_multiple_copies(self):
        """Test creating multiple copies."""
        from scivianna.data.data1d import Data1D
        
        series = {'a': pd.Series([1.0])}
        original = Data1D.from_serie_dict(series)
        
        copy1 = original.copy()
        copy2 = original.copy()
        copy3 = copy1.copy()
        
        assert len(copy1.line_ids) == 1
        assert len(copy2.line_ids) == 1
        assert len(copy3.line_ids) == 1

    def test_chained_operations(self):
        """Test chaining from_serie_dict with copy and check_valid."""
        from scivianna.data.data1d import Data1D
        
        result = (
            Data1D.from_serie_dict({'a': pd.Series([1.0])})
            .copy()
        )
        
        # Should be valid
        result.check_valid()